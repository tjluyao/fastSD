from mm_optimizer import mm_optimizer,mm_request
from transformers import AutoConfig, AutoTokenizer
from embedding_llama import LlamaForCausalLM
import torch
from punica import BatchedKvCache, BatchLenInfo, KvPool

class lynx_req(mm_request):
    def __init__(
            self, 
            input, 
            tokenizer, 
            kvpool, 
            device='cuda',
            img_path=None,
            additional_init_length=576,
            **kwargs,
        ):
        super().__init__(
            input=input,
            tokenizer=tokenizer,
            kvpool=kvpool,
            device=device,
            img_path=img_path,
            temperature=0.9,
            repetition_penalty=1.0,
            top_p=0.9,
            top_k=-1,
            max_new_tokens=500,
            additional_init_length=additional_init_length,
            **kwargs,
        )

    def get_id(self, tokenizer):
        input_ids = tokenizer.encode(self.input)
        return input_ids
    
class lynx_optimizer(mm_optimizer):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

    def init_model(self, **kwargs):
        model_path = kwargs.get('model_path',None)
        config_path = kwargs.get('config_path',model_path)
        import yaml
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        model_config = AutoConfig.from_pretrained(model_path)
        self.model_config = model_config
        self.tokenizer = self.build_tokenizer(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path,
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    config=model_config
                                                    )
        
        self.model.to(self.device).eval()
        self.vision_model = self.build_vision_model(config, **kwargs)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(config)
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

    def build_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer

    def build_vision_model(self, config, **kwargs):
        checkpoint_path = kwargs.get('vision_model_path', None)
        from models.lynx.vits.eva_vit import create_eva_vit_g
        model, missing_keys = create_eva_vit_g(
            config['image_res'], 
            config.get('drop_path_rate', 0.0),
            load_params=True,
            checkpoint_path=checkpoint_path,
            )
        return model.half()
    
    def build_projector(self, config):
        text_width = self.model.config.hidden_size
        vision_width = self.vision_model.embed_dim
        if config['bridge'] == 'resampler':
            from models.lynx.resampler import PerceiverResampler
            model = PerceiverResampler(vision_width, text_width,
                                       depth=config["bridge_depth"], num_latents=config["num_bridge_tokens"])
        else:
            raise NotImplementedError
        return model.half()
    
    def preprocess(self, batch):
        imgs = [req.img for req in batch]
        imgs = torch.cat(imgs, dim=0).to(self.device)
        #print(imgs.shape)
        img_tensors = self.vision_model(imgs)
        if self.projector:
            img_tensors,_ = self.projector(img_tensors)
        for i, req in enumerate(batch):
            req.img_tensor = img_tensors[i]
            self.prefill(req)
            self.wait_runtime.append(req)
        return batch
    
    def prefill(self, req):
        generator = req.generator
        img_tensor = req.img_tensor.unsqueeze(0)
        #print(img_tensor.shape)
        input_ids = torch.tensor(generator.output_ids).to(self.device)
        length = len(input_ids) + generator.additional_init_length
        input_embeddings = self.id_embedder(input_ids).unsqueeze(0)
        input_embeddings = torch.cat([img_tensor,input_embeddings], dim=1)
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
        next_token_id = generator.get_next_token_id(logits[0])
        generator.append_token(next_token_id)
        generator.next_token_id = next_token_id
    
if __name__ == '__main__':
    optimizer = lynx_optimizer(
        model_name='lynx',
        model_path='checkpoints/llava-v1.5-7b',
        config_path='configs/LYNX.yaml',
        vision_model_path='checkpoints/EVA-CLIP/EVA01_g_psz14.pt',
        )
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = lynx_req(
                    usr_input,
                    optimizer.tokenizer,
                    optimizer.kvpool,
                    img_path='inputs/02.jpg',
                    resize=420,
                    img_dtype=torch.float16,
                    additional_init_length=32,
                    )
                optimizer.wait_preprocess.append(req)
    import threading
    t = threading.Thread(target=get_usr_input)
    t.daemon = True
    t.start()
    while True:
        optimizer.check_prepost()
        optimizer.runtime()