from llama_optimizer import llama_optimizer
from llama_optimizer import llama_req
from transformers import AutoTokenizer, AutoConfig
from punica import BatchedKvCache, BatchLenInfo, BatchedLlamaLoraWeight, KvPool
from embedding_llama import LlamaForCausalLM
import torch,os
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class mm_request(llama_req):
    def __init__(
            self, 
            input, 
            tokenizer, 
            kvpool, 
            device='cuda',
            img_path=None,
            additionl_init_length=576,
            **kwargs,
        ):
        self.device = device
        super().__init__(
            input=input,
            tokenizer=tokenizer,
            kvpool=kvpool,
            temperature=0.9,
            repetition_penalty=1.0,
            top_p=0.9,
            top_k=-1,
            max_new_tokens=500,
            additionl_init_length=additionl_init_length,
        )
        self.img_path = img_path
        self.img = self.load_img(img_path, **kwargs)
        self.img_tensor = None
    
    def load_img(self, img_path, **kwargs):
        image_processor = kwargs.get('image_processor', None)
        img = Image.open(img_path).convert('RGB')
        if image_processor:
            img = image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return img
    
    def get_id(self, tokenizer):
        return super().get_id(tokenizer)
    
class mm_optimizer(llama_optimizer):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        model_config = AutoConfig.from_pretrained(model_path)
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    config=model_config
                                                    )
        
        self.model.to(self.device).eval()
        self.vision_model = self.build_vision_model(model_config)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(model_config)
        self.projector.to(self.device).eval()
        self.id_embedder = self.model.model.embed_tokens
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=torch.float16,
            device=self.device,
        )
        print('Model initialized.')

    def build_vision_model(self, model_config, **kwargs):
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        mm_vision_tower = "openai/clip-vit-large-patch14-336"
        return CLIPVisionTower(mm_vision_tower, args=model_config, **kwargs)
    
    def build_projector(self, model_config, **kwargs):
        from llava.model.multimodal_projector.builder import build_vision_projector
        return build_vision_projector(model_config, **kwargs)

    def preprocess(self, batch):
        imgs = [req.img for req in batch]
        img_tensors = self.vision_model(imgs)
        img_tensors = torch.stack(img_tensors, dim=0)
        if self.projector:
            img_tensors = self.projector(img_tensors)
        for i, req in enumerate(batch):
            req.img_tensor = img_tensors[i]
            self.prefill(req)
            self.wait_runtime.append(req)
        return batch
    
    def prefill(self, req):
        generator = req.generator
        img_tensor = req.img_tensor.half()
        input_ids = torch.tensor(generator.output_ids).to(self.device)
        length = len(input_ids) + generator.additionl_init_length
        input_embeddings = self.id_embedder(input_ids).unsqueeze(0)
        input_embeddings = torch.cat([input_embeddings, img_tensor], dim=1)
        blen = BatchLenInfo([length], 0, self.device)
        prefill_kv = BatchedKvCache([generator.kvcache]) if generator.kvcache else None
        decode_kv = None
        logits, _ = self.model(
            input_ids = None, 
            blen = blen, 
            prefill_kv = prefill_kv, 
            decode_kv = decode_kv, 
            input_embeddings = input_embeddings,
            )
        print(logits.shape)
        next_token_id = generator.get_next_token_id(logits[0])
        generator.append_token(next_token_id)
        generator.next_token_id = next_token_id

    
    def iteration(self, batch):
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
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

        input_ids = torch.tensor(
                prefill_input_ids + decode_input_ids,
                dtype=torch.long,
                device=self.device,
            )

        blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
        prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None

        logits, _ = self.model(
            input_ids = input_ids, 
            blen = blen, 
            prefill_kv = prefill_kv, 
            decode_kv = decode_kv, 
            input_embeddings = None,
            )
        
        if prefill_kv:
            if decode_kv:
                logits = torch.cat(
                    [logits[blen.indptr[1:] - 1], logits[blen.doff :]]
                )
            else:
                logits = logits[blen.indptr[1:] - 1]
        print(logits.shape)
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
        model_name='mm',
        model_path='checkpoints/llava-v1.5-7b'
    )
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = mm_request(
                    usr_input,
                    optimizer.tokenizer,
                    optimizer.kvpool,
                    img_path='inputs/01.jpg',
                    image_processor=optimizer.vision_model.image_processor,
                    )
                optimizer.wait_preprocess.append(req)
    import threading
    t = threading.Thread(target=get_usr_input)
    t.daemon = True
    t.start()
    while True:
        optimizer.check_prepost()
        optimizer.runtime()