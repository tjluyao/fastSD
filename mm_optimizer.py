from llamalora_optimizer import llamalora_optimizer
from llama_optimizer import llama_req
from transformers import AutoTokenizer, AutoConfig
from punica import LlamaForCausalLMWithLora, BatchedKvCache, BatchLenInfo, BatchedLlamaLoraWeight
import torch,os
from PIL import Image

class mm_request(llama_req):
    def __init__(self, 
                 input, 
                 tokenizer, 
                 kvpool, 
                 img_path=None,
                 ):
        super().__init__(
            input=input,
            tokenizer=tokenizer,
            kvpool=kvpool,
            temperature=0.9,
            repetition_penalty=1.1,
            top_p=0.9,
            top_k=-1,
            max_new_tokens=500,
            lora_id=None,
        )
        self.img_path = img_path
        self.img = self.load_img(img_path)
        self.img_tensor = None
    
    def load_img(self, img_path):
        img = Image.open(img_path)
        return img
    
class mm_optimizer(llamalora_optimizer):
    def __init__(self, model_name, device, **kwargs):
        super().__init__(model_name, device, **kwargs)
    
    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        model_config = AutoConfig.from_pretrained(model_path)
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLMWithLora.from_pretrained(model_path,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    config=model_config
                                                    )
        
        self.model.to(self.device).eval()
        self.vision_model = self.build_vision_model(model_config)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(model_config)
        self.projector.to(self.device).eval()


    def build_vision_model(self, model_config, **kwargs):
        pass
    
    def build_projector(self, model_config, **kwargs):
        pass

    def preprocess(self, batch):
        imgs = [req.img for req in batch]
        img_tensors = self.vision_model(imgs)
        img_tensors = self.projector(img_tensors)
        for i, req in enumerate(batch):
            req.img_tensor = img_tensors[i]
        return batch
    
    def interation(self, batch):
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
                reqctx.kvcache = None
        return batch

if __name__ == '__main__':
    optimizer = mm_optimizer(
        'mm',
        device='cuda',
        model_path='checkpoints/llava-v1.5-7b'
    )
    print(optimizer.model)
    print(optimizer.vision_model)
    print(optimizer.projector)