from llama_optimizer import llama_optimizer
from llama_optimizer import llama_req
from transformers import AutoTokenizer, AutoConfig
from punica import BatchedKvCache, BatchLenInfo, KvPool
from embedding_llama import LlamaForCausalLM
import torch, json
from PIL import Image
from huggingface_hub import hf_hub_download
from io import BytesIO
import base64

class mm_request(llama_req):
    def __init__(
            self, 
            prompt, 
            tokenizer, 
            kvpool,
            make_generator=False, 
            img_path=None,
            init_length=None,
            **kwargs,
            ):
        super().__init__(
            prompt=prompt,
            tokenizer=tokenizer,
            kvpool=kvpool,
            make_generator=make_generator,
            temperature=0.9,
            repetition_penalty=1.0,
            top_p=0.9,
            top_k=-1,
            max_new_tokens=1024,
            init_length=init_length,
        )
        self.img = self.load_img(img_path, **kwargs)
    
    def load_img(self, img_path, **kwargs):
        with open(img_path, "rb") as image_file:
            img_encoded = base64.b64encode(image_file.read())
        return img_encoded
    
class mm_optimizer(llama_optimizer):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
    
    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        self.model_id = model_path
        model_config = AutoConfig.from_pretrained(model_path)
        self.model_config = model_config
        self.tokenizer = self.build_tokenizer(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            config=model_config
            )
        self.model.to(self.device).eval()
        
        
        with open(hf_hub_download(model_path, filename='config.json')) as f:
            mm_config = json.loads(f.read())
        self.vision_model = self.build_vision_model(mm_config)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(mm_config)
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
        self.additional_init_length = 576
        print('Model initialized.')

    def build_tokenizer(self, model_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def build_vision_model(self, model_config, **kwargs):
        from models.llava.encoder.clip_encoder import CLIPVisionTower
        mm_vision_tower = "openai/clip-vit-large-patch14-336"
        return CLIPVisionTower(mm_vision_tower, args=model_config, **kwargs)
    
    def build_projector(self, model_config, **kwargs):
        from models.llava.projector import build_vision_projector
        projector = build_vision_projector(model_config, **kwargs)
        model_path = hf_hub_download(self.model_id, filename='mm_projector.bin')
        state_dict = torch.load(model_path)
        new_state_dict = {
            '0.weight': state_dict['model.mm_projector.0.weight'],
            '0.bias': state_dict['model.mm_projector.0.bias'],
            '2.weight': state_dict['model.mm_projector.2.weight'],
            '2.bias': state_dict['model.mm_projector.2.bias'],
        }
        projector.load_state_dict(new_state_dict)
        return projector

    def preprocess(self, batch):
        img_features = []
        for r in batch:
            img = Image.open(BytesIO(base64.b64decode(r.img))).convert('RGB')
            img = self.vision_model.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
            img_features.append(img)
        img_features = torch.stack(img_features, dim=0)
        img_features = self.vision_model(img_features)
        if self.projector:
            img_features = self.projector(img_features)

        def get_input(prompt):
            return 'USER: '+ prompt + ' ASSISTANT: '
        
        input_prompts = [get_input(r.prompt) for r in batch]
        input_ids = self.tokenizer.batch_encode_plus(
            input_prompts,
            return_tensors='pt',
            padding=True,
            max_length=1024,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            )['input_ids'].to(self.device)
        for i, item in enumerate(batch):
            item.make_generator(input_ids[i], init_length=len(input_ids[i]) + self.additional_init_length)
            
        input_embeddings = self.id_embedder(input_ids)
        input_embeddings = torch.cat([img_features,input_embeddings], dim=1).half()

        lens = [input_embeddings.size(1) for _ in batch]
        blen = BatchLenInfo(lens, 0, self.device)
        prefill_kv = BatchedKvCache([r.generator.kvcache for r in batch])

        logits, _ = self.model(
            input_ids = None, 
            blen = blen, 
            prefill_kv = prefill_kv, 
            decode_kv = None, 
            input_embeddings = input_embeddings,
            )
        logits = logits[blen.indptr[1:] - 1]
        
        for i, item in enumerate(batch):
            reqctx = item.generator
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            item.state = 1
            self.wait_runtime.append(item)
        return batch

    
    def iteration(self, batch):
        input_ids, decode_kv = [], []

        for item in batch:
            reqctx = item.generator
            input_ids.append(reqctx.output_ids[-1])
            decode_kv.append(reqctx.kvcache)
            reqctx.kvcache.acquire_one()

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        blen = BatchLenInfo([], len(input_ids), self.device)
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None

        # Forward pass
        logits, _ = self.model(input_ids, blen, None, decode_kv, None)

        for i, item in enumerate(batch):
            reqctx = item.generator
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            if reqctx.is_stop():
                item.state = 2
                reqctx.kvcache.release()
        return batch

if __name__ == '__main__':
    optimizer = mm_optimizer(
        model_name='mm',
        model_path='liuhaotian/llava-v1.5-7b'
    )
    mode = 'test'
    if mode == 'serve':
        def get_usr_input():
            while True:
                usr_input = input()
                if usr_input != '\n':
                    req = mm_request(
                        usr_input,
                        optimizer.tokenizer,
                        optimizer.kvpool,
                        img_path='inputs/00.jpg',
                        )
                    optimizer.wait_preprocess.append(req)

        import threading
        t = threading.Thread(target=get_usr_input)
        t.daemon = True
        t.start()
    else:
        req1 = mm_request(
            'what is in the picture?',
            optimizer.tokenizer,
            optimizer.kvpool,
            img_path='inputs/00.jpg',
            )
        req2 = mm_request(
            'what is the color of the sky?',
            optimizer.tokenizer,
            optimizer.kvpool,
            img_path='inputs/00.jpg',
            )
        optimizer.wait_preprocess.append(req1)
        optimizer.wait_preprocess.append(req2)
    while True:
        optimizer.check_prepost()
        optimizer.runtime()