from mm_optimizer import mm_optimizer,mm_request
from transformers import LlamaConfig, LlamaTokenizer
from embedding_llama import LlamaForCausalLM
import torch
from punica import BatchedKvCache, BatchLenInfo, KvPool
from PIL import Image

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
            top_k=3,
            max_new_tokens=500,
            additional_init_length=additional_init_length,
            **kwargs,
        )

    def get_id(self, tokenizer):
        input = \
        'User: '+\
        self.input+\
        '\nBot:'
        input_ids = tokenizer.encode(input)
        return input_ids
    
    def load_img(self, img_path, **kwargs):
        config = kwargs.get('config', None)
        transform = self.init_image_transform(config)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        if kwargs.get('img_dtype', None):
            img = img.to(kwargs['img_dtype'])
        print(img.shape)
        return img.unsqueeze(0)

    def init_image_transform(self, config):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        normalize = transforms.Normalize(config['image_mean'], config['image_std'])

        def _convert_to_rgb(image):
            return image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(size=config['image_res'], interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=(config['image_res'], config['image_res'])),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])

        return transform

    
class lynx_optimizer(mm_optimizer):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

    def init_model(self, **kwargs):
        config_path = kwargs.get('config_path',None)
        if config_path:
            import yaml
            config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
            self.config = config
        self.tokenizer, num_new_tokens = self.build_tokenizer(config, **kwargs)
        kwargs['num_new_tokens'] = num_new_tokens
        self.model = self.bulid_llm(config, **kwargs)
        model_config = self.model.config
        self.model.to(self.device).eval()
        self.vision_model = self.build_vision_model(config, **kwargs)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(config)
        self.projector.to(self.device).eval()
        if kwargs.get('load_weights', False):
            path = config.get('checkpoint', None)
            self.load_weights(path)
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

    def load_weights(self, path):
        weights = torch.load(path, map_location='cpu')
        model_weights = {}
        visual_weights = {}
        projector_weights = {}
        for key in weights:
            if 'LLM' in key:
                new_key = key.replace('model.LLM.','')
                model_weights[new_key] = weights[key]
            elif 'vision_encoder' in key:
                new_key = key.replace('model.vision_encoder.','')
                visual_weights[new_key] = weights[key]
            elif 'bridge' in key:
                new_key = key.replace('model.bridge.','')
                projector_weights[new_key] = weights[key]
        self.model.load_state_dict(model_weights,strict=False)
        self.vision_model.load_state_dict(visual_weights,strict=True)
        self.projector.load_state_dict(projector_weights)
        print('New weights loaded.')

    def bulid_llm(self, config, **kwargs):
        use_adapter = config.get('use_adapter', False)
        model_path = kwargs.get('model_path', config.get('checkpoint', None))
        model_config = LlamaConfig.from_pretrained(model_path)
        model_config.use_adapter = use_adapter
        model_config.adapter_freq = config.get('adapter_freq', -1)
        model_config.freeze_params = config.get('freeze_params', True)
        model_config.label_smoothing = config.get("label_smoothing", 0.0)
        model = LlamaForCausalLM.from_pretrained(model_path, config=model_config)
        model.model.padding_idx = self.tokenizer.pad_token_id

        if kwargs.get('num_new_tokens',0) > 0:
            num_new_tokens = kwargs['num_new_tokens']
            print("### LLM Vocab Size: ", model.config.vocab_size, flush=True)
            print("### num_new_tokens: ", num_new_tokens, flush=True)
            vocab_size = model.config.vocab_size + num_new_tokens
            assert vocab_size == len(self.tokenizer)

            model.resize_token_embeddings(vocab_size)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        return model.half()

    def build_tokenizer(self, config, **kwargs):
        path = kwargs.get('model_path', None)
        tokenizer = LlamaTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            num_new_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": '[PAD]',
                }
            )

        if tokenizer.bos_token is None:
            TOKEN_NONE_FLAG = "[NONE]"
            print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
            tokenizer.bos_token = TOKEN_NONE_FLAG
        else:
            print("bos_token, ", tokenizer.bos_token)
            print("bos_token_id, ", tokenizer.bos_token_id)

        if config.get('use_left_pad', None):
            tokenizer.pad_token = 'left'
        return tokenizer, num_new_tokens

    def build_vision_model(self, config, **kwargs):
        checkpoint_path = kwargs.get('vision_model_path', None)
        from models.lynx.vits.eva_vit import create_eva_vit_g
        model, missing_keys = create_eva_vit_g(
            config['image_res'], 
            config.get('drop_path_rate', 0.0),
            load_params=True,
            checkpoint_path=checkpoint_path,
            )
        if missing_keys:
            print('Missing keys:', missing_keys)
        return model.half()
    
    def build_projector(self, config):
        text_width = self.model.config.hidden_size
        vision_width = self.vision_model.embed_dim
        if config['bridge'] == 'resampler':
            from models.lynx.resampler import PerceiverResampler
            model = PerceiverResampler(
                vision_width, 
                text_width,
                depth=config["bridge_depth"], 
                num_latents=config["num_bridge_tokens"]
                )
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
        input_embeddings = self.id_embedder(input_ids).unsqueeze(0)
        #print(input_embeddings.shape)
        input_embeddings = torch.cat([img_tensor,input_embeddings], dim=1)
        length = len(input_ids) + generator.additional_init_length
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
        model_path='checkpoints/vicuna-7b-v1.1',
        config_path='configs/LYNX.yaml',
        vision_model_path='checkpoints/EVA-CLIP/EVA01_g_psz14.pt',
        load_weights=True,
        )
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = lynx_req(
                    usr_input,
                    optimizer.tokenizer,
                    optimizer.kvpool,
                    img_path='inputs/03.jpg',
                    config=optimizer.config,
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