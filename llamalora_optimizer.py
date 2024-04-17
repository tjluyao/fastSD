import peft
import torch
from llama_optimizer import llama_req, llama_optimizer
from transformers import AutoTokenizer, LlamaConfig
from punica import LlamaForCausalLMWithLora, KvPool, BatchLenInfo, BatchedKvCache, LoraWeight, BatchedLlamaLoraWeight
from huggingface_hub import hf_hub_download

lora_paths = {
    'fin':'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
    'Chinese':'hfl/chinese-alpaca-2-lora-7b',
}

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

class llamalora_optimizer(llama_optimizer):
    def __init__(
        self,
        model_name: str,
        batch_option: int = 1,
        max_batch_size: int = 10,
        seed: int = 49,
        **kwargs
        ):
        lora_ids = kwargs.get('lora_ids',None)
        super().__init__(model_name, batch_option, max_batch_size, seed, **kwargs)
        self.lora_weights = self.init_lora(
            lora_ids, 
            self.model_config, 
            device=self.device,
            )

    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        model_config = LlamaConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLMWithLora.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            config=model_config
            )
        self.model.to(self.device)
        self.model.eval()
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=torch.float16,
            device=self.device,
        )
        self.model_config = model_config
        print('Model initialized.')
 
    def iteration(self,batch):
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []
        for item in batch:
            reqctx = item.generator
            if reqctx.is_prefill():
                prefill_input_ids.extend(reqctx.output_ids)
                prefill_lens.append(len(reqctx.output_ids))
                prefill_kv.append(reqctx.kvcache)
            else:
                decode_input_ids.append(reqctx.output_ids[-1])
                decode_kv.append(reqctx.kvcache)
                reqctx.kvcache.acquire_one()
            if lora_ids and lora_ids[-1] == reqctx.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(reqctx.lora_id)
                lora_lens.append(1)
        input_ids = torch.tensor(
                prefill_input_ids + decode_input_ids,
                dtype=torch.long,
                device=self.device,
            )
        blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
        prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None
        lora = BatchedLlamaLoraWeight(
            [self.lora_weights[id] for id in lora_ids], lora_lens
        )
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora)
        
        if prefill_kv:
            if decode_kv:
                logits = torch.cat(
                    [logits[blen.indptr[1:] - 1], logits[blen.doff :]]
                )
            else:
                logits = logits[blen.indptr[1:] - 1]

        for i, item in enumerate(batch):
            reqctx = item.generator
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            if reqctx.is_stop():
                item.state = 2
                reqctx.kvcache.release()
        return batch
    
    def init_lora(self,
            lora_ids,
            model_config,
            device=torch.device("cuda:0"),
            dtype=torch.float16,
            defalut_rank = 16,
            ):
        lora_weights = {}
        lora_weights["empty"] = LlamaLoraWeight(
                model_config, defalut_rank, dtype, device
            )
        if lora_ids is None:
            return lora_weights
        for lora in lora_ids:
            path = lora_paths[lora]
            ckpt_path = hf_hub_download(path,filename='adapter_model.bin')
            config_path = hf_hub_download(path,filename='adapter_config.json')
            tmp = torch.load(
                    ckpt_path, map_location=device, weights_only=True
                )
            lora_rank = peft.config.PeftConfigMixin.from_json_file(config_path)['r']
            if lora_rank < 16:
                lora_weight = LlamaLoraWeight(model_config, lora_rank*2, dtype, device)
            else:
                lora_weight = LlamaLoraWeight(model_config, lora_rank, dtype, device)
            tmp = weight_convert(tmp,lora_rank)
            lora_weight.copy_from_tensors(tmp)
            del tmp
            lora_weights[lora] = lora_weight
        return lora_weights
    
def weight_convert(weights,rank):
    qA,qB,kA,kB,vA,vB,oA,oB,gateA,gateB,upA,upB,downA,downB = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
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

if __name__ == '__main__':
    optimizer = llamalora_optimizer(
        model_name='llama2-7b-lora',
        model_path='meta-llama/Llama-2-7b-chat-hf',
        lora_ids=['fin','Chinese'],
        )
                                
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = llama_req(
                    usr_input,
                    optimizer.tokenizer,
                    optimizer.kvpool,
                    lora_id='fin',
                    )
                optimizer.wait_runtime.append(req) 

    import threading
    t = threading.Thread(target=get_usr_input)
    t.daemon = True
    t.start()
    while True:
        optimizer.check_prepost()
        optimizer.runtime()
