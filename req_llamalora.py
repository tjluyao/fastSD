from transformers import LlamaConfig, AutoTokenizer
from punica import KvCache,LlamaForCausalLMWithLora, KvPool, LoraWeight, BatchedKvCache, BatchLenInfo, BatchedLlamaLoraWeight
import torch, transformers, time, peft

class LlamaLoraWeight:
    def __init__(
        self,
        config: LlamaConfig,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.q = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.k = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.o = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.gate = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.up = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.down = LoraWeight(
            config.num_hidden_layers,
            config.intermediate_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        if ts["q.A"] is not None:
            self.q.copy_from_tensor(ts["q.A"], ts["q.B"])
        if ts["k.A"] is not None:
            self.k.copy_from_tensor(ts["k.A"], ts["k.B"])
        if ts["v.A"] is not None:
            self.v.copy_from_tensor(ts["v.A"], ts["v.B"])
        if ts["o.A"] is not None:
            self.o.copy_from_tensor(ts["o.A"], ts["o.B"])
        if ts["gate.A"] is not None:
            self.gate.copy_from_tensor(ts["gate.A"], ts["gate.B"])
        if ts["up.A"] is not None:
            self.up.copy_from_tensor(ts["up.A"], ts["up.B"])
        if ts["down.A"] is not None:
            self.down.copy_from_tensor(ts["down.A"], ts["down.B"])

def init_Llama_lora(model_path, device):
    model_config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = LlamaForCausalLMWithLora.from_pretrained(model_path,
                                                              low_cpu_mem_usage=True,
                                                              torch_dtype=torch.float16,
                                                              config=model_config)
    model.to(device=device, dtype=torch.float16)
    kvpool = KvPool(
        num_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        page_len=16,
        dtype=torch.float16,
        device=device,
    )
    return model, tokenizer, model_config, kvpool

lora_paths = {
    'fin':'lora_weights/fingpt-forecaster_dow30_llama2-7b_lora',
    'Chinese':'lora_weights/Chinese-Llama-2-LoRA-7B',
}

def init_lora(lora_ids,
              model_config,
              device=torch.device("cuda:0"),
              dtype=torch.float16,
              ):
    lora_weights = {}
    defalut_rank = 16
    lora_weights["empty"] = LlamaLoraWeight(
            model_config, defalut_rank, dtype, device
        )
    for lora in lora_ids:
        path = lora_paths[lora]
        model_path = path+'/adapter_model.bin'
        tmp = torch.load(
                model_path, map_location=device, weights_only=True
            )
        lora_rank = peft.config.PeftConfigMixin.from_json_file(path+'/adapter_config.json')['r']
        if lora_rank < 16:
            lora_weight = LlamaLoraWeight(model_config, lora_rank*2, dtype, device)
        else:
            lora_weight = LlamaLoraWeight(model_config, lora_rank, dtype, device)
        tmp = weight_convert(tmp,lora_rank)
        lora_weight.copy_from_tensors(tmp)
        del tmp
        lora_weights[lora] = lora_weight
    return lora_weights

class llama_lora_req:
    def __init__(self, 
                    prompt, 
                    kvpool, 
                    tokenizer, 
                    lora_id, 
                    temperature=0.9,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    top_k=-1,
                    max_new_tokens=500,
                    ):
        self.time = time.time()
        self.state = 0

        input_ids = tokenizer.encode(prompt)
        self.generator = TextGeneration(
            input_ids=input_ids,
            kvpool=kvpool,
            lora_id=lora_id,
            tokenizer=tokenizer,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            maxlen=max_new_tokens,
            stop_token_id=tokenizer.eos_token_id,
        )
       

class TextGeneration:
    def __init__(
        self,
        input_ids: list[int],
        kvpool: KvPool,
        lora_id: str,
        tokenizer,
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.kvcache = KvCache(kvpool, self.prompt_len)
        self.lora_id = lora_id
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> int:
        if len(self.output_ids) >= self.maxlen:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len

    def decode_tokens(self) -> str:
        # Adapted from: https://github.com/huggingface/text-generation-inference/blob/a5def7c222174e03d815f890093584f3e815c5ce/server/text_generation_server/models/model.py#L68
        prefix_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\uFFFD"):
            new_text = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.output_ids)
            return new_text
        else:
            return ""
        
def weight_convert(weights,rank):
    qA = []
    qB = []
    kA = []
    kB = []
    vA = []
    vB = []
    oA = []
    oB = []
    gateA = []
    gateB = []
    upA = []
    upB = []
    downA = []
    downB = []
    for key in weights.keys():
        if 'q_proj' in key:
            if 'A' in key:
                qA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                qB.append(weights[key].unsqueeze(0))
        if 'k_proj' in key:
            if 'A' in key:
                kA.append(weights[key].unsqueeze(0))    
            if 'B' in key:
                kB.append(weights[key].unsqueeze(0))
        if 'v_proj' in key:
            if 'A' in key:
                vA.append(weights[key].unsqueeze(0))    
            if 'B' in key:
                vB.append(weights[key].unsqueeze(0))
        if 'o_proj' in key:
            if 'A' in key:
                oA.append(weights[key].unsqueeze(0)) 
            if 'B' in key:
                oB.append(weights[key].unsqueeze(0))
        if 'gate_proj' in key:
            if 'A' in key:
                gateA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                gateB.append(weights[key].unsqueeze(0))
        if 'up_proj' in key:
            if 'A' in key:
                upA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                upB.append(weights[key].unsqueeze(0))
        if 'down_proj' in key:
            if 'A' in key:
                downA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                downB.append(weights[key].unsqueeze(0))
    weights = {
        'q.A':torch.cat(qA, dim=0) if qA else None,
        'q.B':torch.cat(qB, dim=0) if qB else None,
        'k.A':torch.cat(kA, dim=0) if kA else None,
        'k.B':torch.cat(kB, dim=0) if kB else None,
        'v.A':torch.cat(vA, dim=0) if vA else None,
        'v.B':torch.cat(vB, dim=0) if vB else None,
        'o.A':torch.cat(oA, dim=0) if oA else None,
        'o.B':torch.cat(oB, dim=0) if oB else None,
        'gate.A':torch.cat(gateA, dim=0) if gateA else None,
        'gate.B':torch.cat(gateB, dim=0) if gateB else None,
        'up.A':torch.cat(upA, dim=0) if upA else None,
        'up.B':torch.cat(upB, dim=0) if upB else None,
        'down.A':torch.cat(downA, dim=0) if downA else None,
        'down.B':torch.cat(downB, dim=0) if downB else None,
    }
    if rank == 8:
        for key in weights.keys():
            if weights[key] is not None:
                if 'A' in key:
                    complement = torch.zeros_like(weights[key])
                    weights[key] = torch.cat([weights[key], complement], dim=1)
                if 'B' in key:
                    complement = torch.zeros_like(weights[key])
                    weights[key] = torch.cat([weights[key], complement], dim=2)
    return weights