import time
import torch
from text_generator import TextGeneration
from default_optimizer import default_optimazer
from transformers import AutoTokenizer, LlamaConfig
from punica import LlamaForCausalLM, KvPool, BatchLenInfo, BatchedKvCache

class llama_req():
    def __init__(self,
                 input,
                 tokenizer,
                 kvpool,
                 temperature=0.9,
                 repetition_penalty=1.1,
                 top_p=0.9,
                 top_k=-1,
                 max_new_tokens=500,
                 lora_id=None,
                 additional_init_length=0,
                 ) -> None:
        
        self.input = input
        self.time = time.time()
        self.id = self.time
        self.state = 0
        input_ids = self.get_id(tokenizer)

        self.generator = TextGeneration(
            input_ids=input_ids,
            kvpool=kvpool,
            tokenizer=tokenizer,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            maxlen=max_new_tokens,
            stop_token_id=tokenizer.eos_token_id,
            lora_id=lora_id,
            additional_init_length=additional_init_length,
        )
    
    def get_id(self,tokenizer):
        input_ids = tokenizer.encode(self.input)
        return input_ids
    
class llama_optimizer(default_optimazer):
    def __init__(self,
                 model_name: str,
                 batch_option: int = 1,
                 max_batch_size: int = 10,
                 seed: int = 49,
                 **kwargs
                 ):
        super().__init__(model_name, batch_option, max_batch_size, seed, **kwargs)

    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        model_config = LlamaConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path,
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
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv)
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
                reqctx.kvcache = None
        return batch
    
    def postprocess(self, batch):
        output_ids = []
        for req in batch:
            output_ids.append(req.generator.output_ids)
            del req
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for sentence in output:
            print(sentence)

if __name__ == '__main__':
    optimizer = llama_optimizer('llama2-7b',
                                model_path='checkpoints/Llama-2-7b-chat-hf',
                                )
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = llama_req(usr_input,
                                optimizer.tokenizer,
                                optimizer.kvpool,
                                )
                optimizer.wait_runtime.append(req) 

    import threading
    t = threading.Thread(target=get_usr_input)
    t.daemon = True
    t.start()
    while True:
        optimizer.check_prepost()
        optimizer.runtime()

