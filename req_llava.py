from transformers import AutoConfig, AutoModelForCausalLM,LlamaConfig,AutoTokenizer
from punica import  LlamaModel, LlamaForCausalLM, KvPool, KvCache
from llava.model.language_model.cached_llama import LlavaLlamaForCausalLM, LlavaConfig
import torch,transformers
from punica.utils import BatchedKvCache, BatchLenInfo
import os,time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_IMAGE_PATCH_TOKEN
from llava.serve.cli import load_image
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class llava_req():
    def __init__(self,input,img_path,tokenizer,model,config,image_processor,kv_pool) -> None:
        self.input = input
        self.time = time.time()
        self.state = 0
        self.image,self.image_tensor = self.get_img(img_path,config,image_processor,device=model.device)
        self.input_ids = self.get_id(tokenizer,model)
        self.generator = TextGeneration(
            self.input_ids[0].tolist(),
            temperature=0.7,
            repetition_penalty=1.0,
            top_p=0.9,
            top_k=-1,
            max_new_tokens=256,
            stop_token_id=tokenizer.eos_token_id,
        )
        input_length = self.input_ids.shape[-1]+575
        self.kvcache = KvCache(kv_pool, input_length)

        self.next_token_id = self.generator.prefill_kv(model,
                                                        self.input_ids,
                                                        self.kvcache, 
                                                        input_length, 
                                                        model.device,
                                                        image_tensor=self.image_tensor)
        pass

    def get_img(self,img_path,config,image_processor,device):
        image = load_image(img_path)
        image_tensor = process_images([image], image_processor, config)
        if type(image_tensor) is list:
            image_tensor = [image.to(device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)
        return image, image_tensor
    
    def get_id(self,tokenizer,model):
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = self.input
        if self.image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            self.image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        return input_ids

        
        
    
class TextGeneration:
    def __init__(
        self,
        input_ids: list[int],
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        stop_token_id: int,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
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

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            print(logits.shape)
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
        if len(self.output_ids) - self.prompt_len >= self.max_new_tokens:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False
    
    def remove_unvalid(self, ids):
        out = ids.copy()
        for i in out:
            if i<0:
                out.remove(i)
        return out
    
    def prefill_kv(self, model, input_ids, kvcache, input_length, device, image_tensor=None):
        input_ids = input_ids.clone().detach().to(torch.long).to(device)
        logits, _ = model(
            input_ids=input_ids,
            blen=BatchLenInfo([input_length], 0, device),
            prefill_kv=BatchedKvCache([kvcache]),
            decode_kv=None,
            images=image_tensor,
        )
        #print(logits.shape)
        next_token_id = self.get_next_token_id(logits[0])
        self.append_token(next_token_id)
        return next_token_id
        
    
def init_llava(model_path,device = torch.device("cuda:0")):
        model_name = get_model_name_from_path(model_path)
        model_config = LlamaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path,config=model_config)
        model.to(device=device, dtype=torch.float16)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
                vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        return model,tokenizer,image_processor


            
if __name__ == "__main__":
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_IMAGE_PATCH_TOKEN
    model_path = "/root/fastSD/checkpoints/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    model_config = LlamaConfig.from_pretrained("/root/fastSD/checkpoints/llava-v1.5-7b")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained("/root/fastSD/checkpoints/llava-v1.5-7b",config=model_config)
    device = torch.device("cuda:0")
    model.to(device=device, dtype=torch.float16)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
            vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    image = load_image('/root/fastSD/inputs/00.jpg')
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    from llava.conversation import conv_templates, SeparatorStyle
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    inp = 'what is in the picture'
    if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
    else:
            # later messages
            conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    from punica import BatchedKvCache, KvPool, KvCache, BatchLenInfo
    kvpool = KvPool(
        num_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        page_len=16,
        dtype=torch.float16,
        device=model.device,
    )
    input_length = input_ids.shape[-1]+575
    kvcache = KvCache(kvpool, input_length)
    device = torch.device("cuda:0")
    print(input_ids.shape)
    textgen = TextGeneration(
        input_ids[0].tolist(),
        temperature=0.7,
        repetition_penalty=1.0,
        top_p=0.9,
        top_k=-1,
        max_new_tokens=256,
        stop_token_id=tokenizer.eos_token_id,
    )
    logits, _ = model(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        blen=BatchLenInfo([input_length], 0, device),
        prefill_kv=BatchedKvCache([kvcache]),
        decode_kv=None,
        images=image_tensor,
    )
    print(logits.shape)
    next_token_id = textgen.get_next_token_id(logits[0])
    textgen.append_token(next_token_id)
    print(textgen.output_ids)
    text = tokenizer.decode(
            textgen.remove_unvalid(textgen.output_ids),
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
    print(text)
    last_print_len = len(text)
    while not textgen.is_stop():
        kvcache.acquire_one()
        logits, _ = model(
            input_ids=torch.tensor([next_token_id], dtype=torch.long, device=device).unsqueeze(0),
            blen=BatchLenInfo([], 1, device),
            prefill_kv=None,
            decode_kv=BatchedKvCache([kvcache]),
            images=image_tensor,
        )
        next_token_id = textgen.get_next_token_id(logits[0])
        textgen.append_token(next_token_id)

        text = tokenizer.decode(
            textgen.remove_unvalid(textgen.output_ids),
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        print(text[last_print_len:], end="", flush=True)
        last_print_len = len(text)