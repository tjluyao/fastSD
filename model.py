import torch
import transformers
from transformers import AutoTokenizer,AutoConfig
from embedding_llama import LlamaForCausalLM
from embedding_llamalora import LlamaForCausalLMWithLora
from punica import KvPool, BatchLenInfo, BatchedKvCache, KvCache

class abstract_model:
    def __init__(self,config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config['model_type']
        self.model_mode = config['model_mode']
        self.next_model = config['next_model']
    
class tokenizer_model(abstract_model):
    def __init__(self,config):
        super().__init__(config)
        path = config['checkpoint_path']
        options = config.get('options',{})
        self.model = AutoTokenizer.from_pretrained(path,**options)
        if config['prompt']:
            self.prompt = config['prompt']
        else:
            self.prompt = None
    
    def run(self,reqs,**kwargs):
        inp = [x.prompt for x in reqs]
        if self.prompt:
            inp = [self.prompt.format(input=x) for x in inp]
        outs = self.model.batch_encode_plus(inp)['input_ids']
        for i,out in enumerate(outs):
            reqs[i].token_ids = out
            reqs[i].state = int(self.next_model)
        return reqs
    
class tokenizer_decode_model(abstract_model):
    def __init__(self,config):
        super().__init__(config)
        path = config['checkpoint_path']
        self.model = AutoTokenizer.from_pretrained(path)
    
    def run(self, reqs, **kwargs):
        inp = [x.token_ids for x in reqs]
        outs = self.model.batch_decode(inp, skip_special_tokens=True)
        for i,out in enumerate(outs):
            reqs[i].output = out
            reqs[i].state = self.next_model
            print(out)
        return reqs
    
class llama_model(abstract_model):
    def __init__(self, config):
        super().__init__(config)
        path = config['checkpoint_path']
        options = config.get('options',{})
        if options.get('torch_dtype',None) == 'float16':
            options['torch_dtype'] = torch.float16
        model_config = AutoConfig.from_pretrained(path)
        self.model = LlamaForCausalLM.from_pretrained(path,
                                                      config=model_config,
                                                      **options)
        if options.get('use_adapter',False):
            weights = torch.load(config['adapter_path'],map_location=self.device)
            self.model.load_state_dict(weights,strict=False)
        self.model.half().to(self.device).eval()
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=torch.float16,
            device=self.device,
        )

        self.temperature = config.get('temperature',0.9)
        self.repetition_penalty = config.get('repetition_penalty',1.0)
        self.top_p = config.get('top_p',0.9)
        self.top_k = config.get('top_k',-1)
        self.maxlen = config.get('maxlen',500)
        self.stop_token_id = self.model.config.eos_token_id
        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if self.temperature > 0 and self.temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(self.temperature)
            )
        if self.repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(self.repetition_penalty)
            )
        if 0 < self.top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(self.top_p))
        if self.top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(self.top_k))

    
    def run(self,reqs,**kwargs):
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        for req in reqs:
            if req.is_prefill:
                prefill_input_ids.extend(req.token_ids)
                length = len(req.token_ids)
                prefill_lens.append(length)
                kvcache = KvCache(self.kvpool, length)
                setattr(req, 'kvcache', kvcache)
                prefill_kv.append(req.kvcache)
                req.is_prefill = False
            else:
                decode_input_ids.append(req.token_ids[-1])
                decode_kv.append(req.kvcache)
                req.kvcache.acquire_one()

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

        for i, req in enumerate(reqs):
            stop_info = self.set_next_token_id(req,logits[i].unsqueeze(0))
            if stop_info:
                req.state = self.next_model
                req.kvcache.release()
                req.kvcache = None
        return reqs
    
    def set_next_token_id(
            self, 
            req,
            logits: torch.Tensor,
            ) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([req.token_ids], device=logits.device)
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
        req.token_ids.append(token)
        stop_info = len(req.token_ids) >= self.maxlen or req.token_ids[-1] == self.stop_token_id
        return stop_info
    
from lora import LlamaLoraWeight, weight_convert, lora_paths
import peft
from punica import BatchedLlamaLoraWeight
class llamalora_model(abstract_model):

    def __init__(self, config):
        super().__init__(config)
        path = config['checkpoint_path']
        options = config.get('options',{})
        if options.get('torch_dtype',None) == 'float16':
            options['torch_dtype'] = torch.float16
        model_config = AutoConfig.from_pretrained(path)
        self.model = LlamaForCausalLMWithLora.from_pretrained(path,
                                                      config=model_config,
                                                      **options)
        if options.get('use_adapter',False):
            weights = torch.load(config['adapter_path'],map_location=self.device)
            self.model.load_state_dict(weights,strict=False)
        self.model.half().to(self.device).eval()
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=torch.float16,
            device=self.device,
        )
        lora_ids = config.get('lora_ids',[])
        self.lora_weights = self.init_lora(
            lora_ids, 
            model_config, 
            device=self.device,
            )

        self.temperature = config.get('temperature',0.9)
        self.repetition_penalty = config.get('repetition_penalty',1.0)
        self.top_p = config.get('top_p',0.9)
        self.top_k = config.get('top_k',-1)
        self.maxlen = config.get('maxlen',500)
        self.stop_token_id = self.model.config.eos_token_id
        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if self.temperature > 0 and self.temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(self.temperature)
            )
        if self.repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(self.repetition_penalty)
            )
        if 0 < self.top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(self.top_p))
        if self.top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(self.top_k))
    
    def init_lora(self,
            lora_ids,
            model_config,
            device=torch.device("cuda:0"),
            dtype=torch.float16,
            ):
        lora_weights = {}
        defalut_rank = 16
        lora_weights["empty"] = LlamaLoraWeight(
                model_config, defalut_rank, dtype, device
            )
        if lora_ids is None:
            return lora_weights
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

    
    def run(self,reqs,**kwargs):
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []
        for req in reqs:
            if req.is_prefill:
                prefill_input_ids.extend(req.token_ids)
                length = len(req.token_ids)
                prefill_lens.append(length)
                kvcache = KvCache(self.kvpool, length)
                setattr(req, 'kvcache', kvcache)
                prefill_kv.append(req.kvcache)
                req.is_prefill = False
            else:
                decode_input_ids.append(req.token_ids[-1])
                decode_kv.append(req.kvcache)
                req.kvcache.acquire_one()
            if lora_ids and lora_ids[-1] == req.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(req.lora_id)
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

        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora=lora)

        if prefill_kv:
            if decode_kv:
                logits = torch.cat(
                    [logits[blen.indptr[1:] - 1], logits[blen.doff :]]
                )
            else:
                logits = logits[blen.indptr[1:] - 1]

        for i, req in enumerate(reqs):
            stop_info = self.set_next_token_id(req,logits[i].unsqueeze(0))
            if stop_info:
                req.state = self.next_model
                req.kvcache.release()
                req.kvcache = None
        return reqs
    
    def set_next_token_id(
            self, 
            req,
            logits: torch.Tensor,
            ) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([req.token_ids], device=logits.device)
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
        req.token_ids.append(token)
        stop_info = len(req.token_ids) >= self.maxlen or req.token_ids[-1] == self.stop_token_id
        return stop_info


from PIL import Image 
class visual_model(abstract_model):
    def __init__(self,config):
        super().__init__(config)
        self.model = self.build_vision_model(config['visual_options'])
        self.model.to(self.device).eval()
        self.projector = self.build_projector(config['projector_options'])
        self.projector.to(self.device).eval()

    def build_vision_model(self, config):
        model_name = config['model_name']
        if model_name.startswith('openai'):
            from models.llava.encoder.clip_encoder import CLIPVisionTower
            model = CLIPVisionTower(model_name, args=config)
            self.image_processor = model.image_processor
            return model
        elif model_name.startswith('eva'):
            from models.lynx.vits.eva_vit import create_eva_vit_g
            checkpoint_path = config['checkpoint_path']
            model, missing_keys = create_eva_vit_g(
                config['image_res'], 
                config.get('drop_path_rate', 0.0),
                load_params=True,
                checkpoint_path=checkpoint_path,
                )
            if missing_keys:
                print('Missing keys:', missing_keys)
            model.load_state_dict(torch.load(config['finetune_path']))
            self.image_processor = self.build_processor(config)
            return model.half()
        else:
            raise ValueError('model_name not supported.')

    def build_projector(self, config):
        if config['model_name'] == 'llava':
            from models.llava.projector import build_vision_projector
            projector = build_vision_projector(config)
            checkpoint_path = config['checkpoint_path']
            state_dict = torch.load(checkpoint_path)
            projector.load_state_dict(state_dict)
            return projector
        elif config['model_name'] == 'lynx':
            from models.lynx.resampler import PerceiverResampler
            model = PerceiverResampler(
                vision_width=self.model.embed_dim, 
                text_width=config['text_width'],
                depth=config["bridge_depth"], 
                num_latents=config["num_bridge_tokens"]
                )
            model.load_state_dict(torch.load(config['checkpoint_path']))
            return model.half()
    
    def run(self,reqs,**kwargs):
        language_model = kwargs.get('language_model',None)
        imgs = []
        for req in reqs:
            img = Image.open(req.img_path).convert('RGB')
            img = self.process_image(img)
            imgs.append(img)
        imgs = torch.cat(imgs).to(self.device).half()
        features = self.model(imgs)
        features = self.projector(features)
        for i,req in enumerate(reqs):
            img_tensor = features[i].unsqueeze(0)
            self.prefill(req, img_tensor, language_model)
            req.state = self.next_model
        return reqs
    
    def process_image(self, img):
        if self.config['image_process_type'] == 'processor':
            img = self.image_processor(img,return_tensors='pt')['pixel_values'].squeeze(0)
        elif self.config['image_process_type'] == 'transform':
            img = self.image_processor(img).unsqueeze(0)
        
        return img       

    def build_processor(self, config):
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
    
    def prefill(self, req, img_tensor, language_model):
        img_tensor = img_tensor.squeeze(0)
        text_embedder = language_model.model.get_input_embeddings()
        text_ids = torch.tensor(req.token_ids).to(self.device).to(torch.long)
        input_embeddings = text_embedder(text_ids).unsqueeze(0)
        input_embeddings = torch.cat([img_tensor,input_embeddings], dim=1).to(language_model.model.dtype)
        init_lenth = input_embeddings.shape[1]

        kvcache = KvCache(language_model.kvpool, init_lenth)
        setattr(req, 'kvcache', kvcache)
        blen = BatchLenInfo([init_lenth], 0, self.device)
        prefill_kv = BatchedKvCache([req.kvcache]) if req.kvcache else None
        decode_kv = None

        logits, _ = language_model.model(
            input_ids = None, 
            blen = blen, 
            prefill_kv = prefill_kv, 
            decode_kv = decode_kv, 
            input_embeddings = input_embeddings,
            ) 
        
        stop_flag = language_model.set_next_token_id(req,logits[0])
        req.is_prefill = False
        return
