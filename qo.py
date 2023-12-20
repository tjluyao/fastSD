import time
import threading
import torch
import peft
from pytorch_lightning import seed_everything
from llama.model import ModelArgs as llama_args
from sd import *
import fairscale.nn.model_parallel.initialize as fs_init
from sgm.modules.diffusionmodules.sampling import EDMSampler

class request(object):
    def __init__(self,input,model,cache_size=None) -> None:
        self.input = input
        self.time = time.time()
        self.max_tokens = 128
        self.state = 0   # 0 refers to INITIATION
        self.buffer = []
        if cache_size:
            self.cache_k = torch.zeros(cache_size)
            self.cache_v = torch.zeros(cache_size)

        self.buffer = model.tokenizer.encode(self.input, bos=True, eos=False)

class sd_request(object):
    def __init__(
            self, 
            state: dict,
            steps: int = 10,
            is_video: bool = False,
            prompt: str = None,
            negative_prompt: str = '',
            img_path: str = None,
            lora_pth: str = None,
            num_samples = 1,
            ) -> None:
        self.id = time.time()
        self.time = time.time()
        self.state = 0
        self.steps = steps
        keys = list(set([x.input_key for x in state["model"].conditioner.embedders]))
        self.sampling = {}
        self.num_samples = num_samples
        self.num_frames = state['T'] if is_video else 1
        self.num = self.num_samples * self.num_frames
        W = state['W']
        H = state['H']
        
        self.sample_z = None
        if lora_pth:
            self.lora_dict= self.get_lora(lora_pth)
        if img_path:
            self.img, self.w, self.h = self.load_img(path=img_path, n=self.num)

        self.value_dict = self.get_valuedict(keys, img_path, W, H, is_video=is_video, prompt=prompt, negative_prompt=negative_prompt)
        pass

    def get_valuedict(self,keys,img,W,H,is_video=False, prompt=None, negative_prompt=None):
        if is_video:
            value_dict = init_embedder_options(
            keys,
            {},
            )
            img = load_img_for_prediction(W, H, display=False, key=img)
            cond_aug = 0.02
            value_dict["image_only_indicator"] = 0
            value_dict["cond_frames_without_noise"] = img
            value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
            value_dict["cond_aug"] = cond_aug
            value_dict['num_samples'] = self.num_samples
            value_dict['T'] = self.num_frames
        else:
            init_dict = {"orig_width": W,"orig_height": H,"target_width": W,"target_height": H,}
            value_dict = init_embedder_options(
            keys,
            init_dict,
            prompt=prompt,
            negative_prompt=negative_prompt,
            )
            value_dict['num_samples'] = self.num_samples
            value_dict['T'] = self.num_frames
            value_dict["num"] = self.num
        return value_dict
    
    def load_img(self, path=None, display=False, device="cuda"):
        assert path is not None
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if display:
            print(image)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        width, height = 1024, 1024
        image = image.resize((width, height))
        image = np.array(image)[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        return image.to(device),w ,h 
    
    def get_lora(self, lora_pth):
        lora_dict = load_safetensors(lora_pth)
        try:
            rank = peft.LoraConfig.from_pretrained(lora_pth).r
        except:
            print('Rank not found')
            rank = 8
        return {'weights':lora_dict, 'rank':rank}

class query_optimazer:
    def __init__(self,
                 model_name: str = 'llama',
                 output_path: str = 'outputs/',
                 seed: int = 49,
                 ):
        self.model_name = model_name
        self.model = self.init_model(model_name)
        
        self.output_path = output_path
        self.max_batch_size = 4
        self.n_slots = 102400

        self.n_rsrv = 0
        self.n_scheduled = 0
        self.wait_runtime = []

        self.batch_option = 1
        self.wait_preprocess = []
        self.wait_postprocess = []
        seed_everything(seed)
        pass

    def init_model(self,model_name,**kwargs):
        if model_name == 'llama':
            
            from llama import Llama
            '''
            generator = Llama.build(
                ckpt_dir='Llama-2-7b/',
                tokenizer_path='Llama-2-7b/tokenizer.model',
                max_seq_len=128,
                max_batch_size=4,
            )
            '''
            from llama_generator import Large_model
            generator = Large_model()
            print('Model loaded')
            return generator
        
        elif model_name in VERSION2SPECS:
            version_dict = VERSION2SPECS[model_name]
            state = init_model(version_dict, load_filter=True)
            steps = kwargs.get('steps',10)
            state['W'] = 512
            state['H'] = 512
            state['C'] = version_dict['C']
            state['F'] = version_dict['f']
            state['T'] = 6 if model_name in ['sdv'] else None
            state['options'] = version_dict.get('options', {})
            state['options']["num_frames"] = state['T']
            sampler = init_sampling(options=state['options'],steps=steps)
            state['sampler'] = sampler
            state['saving_fps'] = 6
            self.state = state
            load_model(state['model'].model)
            load_model(state['model'].denoiser)
            print('Model loaded')
            return state.get('model')
        
        else:
            raise NotImplementedError
        
    def add_request(self,req):
        if self.model_name == 'llama':
            self.wait_runtime.append(req)
        elif self.model_name in VERSION2SPECS:
            self.wait_preprocess.append(req)
        else:
            raise NotImplementedError
    
    def select_batch(self, pool):
        batch = []
        pool = sorted(pool, key=lambda req: req.time)

        for item in pool:
            if len(batch) == self.max_batch_size:
                break

            if item.state == 0:
                if self.model_name == 'llama':
                    new_n_rsrv = self.n_rsrv + item.max_tokens
                    if new_n_rsrv > self.n_slots:
                        break
                    self.n_rsrv = new_n_rsrv
            batch.append(item)
        return batch
    
    def runtime(self, model, **kwargs):
        use_cache = kwargs.get('use_cache',True)
        r_batch = []
        batch = self.select_batch(self.wait_runtime)
        if batch and not self.n_scheduled:
            for item in batch:
                self.wait_runtime.remove(item)
            if self.model_name == 'llama':
                r_batch = model.generate_iter_cache(batch) if use_cache else model.generate_iter(batch)
            elif self.model_name in VERSION2SPECS:
                r_batch = self.sample(batch, self.state)
            else:
                raise NotImplementedError
            self.n_scheduled = self.n_scheduled + 1

        if r_batch:
            for item in r_batch:
                if item.state == 3:
                    if self.model_name == 'llama':
                        self.n_rsrv = self.n_rsrv-item.max_tokens
                        print('Finish!',item.time,id(item.cache_k))
                        print(model.tokenizer.decode(item.buffer)+'\n')
                        del item
                    elif self.model_name in VERSION2SPECS:
                        self.wait_postprocess.append(item)
                else:
                    item.state = 2
                    self.wait_runtime.append(item)
            self.n_scheduled = self.n_scheduled - 1 

    def check_prepost(self):
        if len(self.wait_preprocess) >= self.batch_option:
            print('Start encoding')
            encode_process = self.wait_preprocess
            self.wait_preprocess = []
            encode_process = self.preprocess(encode_process, self.state)
            for i in encode_process:
                print('Finish encoding',i.id)
                self.wait_runtime.append(i)

        if len(self.wait_postprocess) >= self.batch_option:
            print('Start decoding')
            decode_process = self.wait_postprocess
            self.wait_postprocess = []
            samples_z = torch.cat([i.sampling['pic'] for i in decode_process], dim=0)
            samples = self.postprocess(samples_z, self.state)
            if self.model_name not in ['svd']:
                perform_save_locally(self.output_path, samples)  #Save to local file
            if self.state['T']:
                save_video_as_grid_and_mp4(samples, self.output_path, self.state['T'], self.state['saving_fps'])
            for i in decode_process:
                print('Saved',i.id)
                decode_process.remove(i)
                del i

    def preprocess(self, encode_process,state,muti_input=True):
        model = state.get('model')
        sampler = state.get('sampler')
        options = state.get('options')
        T = state.get('T')
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    value_dicts = [i.value_dict for i in encode_process]
                    batch2model_input = ["num_video_frames", "image_only_indicator"] if T else []
                    z, c, uc, additional_model_inputs= get_condition(
                        state,
                        value_dicts, 
                        sampler=sampler,
                        T=T,
                        batch2model_input=batch2model_input,
                        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                        muti_input=muti_input,
                        )
                    t = 0
                    t2 = 0
                    for i in encode_process:
                        pic = z[t:t+i.num,]
                        ic = {k: c[k][t:t+i.num,] for k in c}
                        iuc = {k: uc[k][t:t+i.num,] for k in uc}
                        ami = {}
                        for k in additional_model_inputs:
                            if k == "image_only_indicator":
                                ami[k] = additional_model_inputs[k][t2:t2+i.num_samples * 2,]
                            elif k == "num_video_frames":
                                ami[k] = i.value_dict['T']

                        pic, s_in, sigmas, num_sigmas, ic, iuc = sampler.prepare_sampling_loop(x=pic, cond=ic, uc=iuc, num_steps=i.steps)
                        i.sampling = {'pic':pic, 
                                    'step':0,
                                    's_in':s_in,
                                    'sigmas':sigmas,
                                    'num_sigmas':num_sigmas,
                                    'c':ic,
                                    'uc':iuc,
                                    'ami':ami,
                                    }
                        t = t + i.num
                        t2 = t2 + i.num_samples * 2
        return encode_process

    def postprocess(self, samples_z, state, return_latents=False):
        model = state.get('model')
        filter = state.get('filter', None)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    load_model(model.first_stage_model)
                    model.en_and_decode_n_samples_a_time = 2
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    unload_model(model.first_stage_model)

                    if filter is not None:
                        samples = filter(samples)

                    if return_latents:
                        return samples, samples_z
                    return samples
    
    def sample(self, sampling, state):
        model = state.get('model')
        sampler = state.get('sampler')
        T = state.get('T')
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    if isinstance(sampler, EDMSampler):
                        begin = []
                        for i in sampling:
                            gamma = min(sampler.s_churn / (i.sampling['num_sigmas'] - 1), 2**0.5 - 1) if sampler.s_tmin <= i.sampling['sigmas'][i.sampling['step']] <= sampler.s_tmax else 0.0
                            sigma = i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']]
                            sigma_hat = sigma * (gamma + 1.0)
                            if gamma > 0:
                                eps = torch.randn_like(i.sampling['pic']) * sampler.s_noise
                                i.sampling['pic'] = i.sampling['pic'] + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5            
                            begin.append(sigma_hat)
                        begin = torch.cat(begin,dim=0)
                    else: 
                        begin = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']] for i in sampling], dim=0)
                                
                    x = torch.cat([i.sampling['pic'] for i in sampling], dim=0)
                    end = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step'] + 1] for i in sampling], dim=0)

                    dict_list = [d.sampling['c'] for d in sampling]
                    cond = {}
                    for key in dict_list[0]:
                        cond[key] = torch.cat([d[key] for d in dict_list], dim=0)

                    dict_list = [d.sampling['uc'] for d in sampling]
                    uc = {}
                    for key in dict_list[0]:
                        uc[key] = torch.cat([d[key] for d in dict_list], dim=0)

                    dict_list = [d.sampling['ami'] for d in sampling] 
                    additional_model_inputs = {}
                    for key in dict_list[0]:
                        if key == "image_only_indicator":
                            additional_model_inputs[key] = torch.cat([d[key] for d in dict_list], dim=0)
                        elif key == "num_video_frames":
                            additional_model_inputs[key] = T

                    lora_dicts = []
                    for d in sampling:
                        for i in range(d.num):
                            lora_dicts.append(d.lora_dict)  
                    lora_dicts = lora_dicts * 2
                    additional_model_inputs['lora_dicts'] = lora_dicts

                    def denoiser(input, sigma, c):
                        return state["model"].denoiser(state["model"].model, input, sigma, c, **additional_model_inputs)
                    samples = sampler.sampler_step_g(begin,end,denoiser,x,cond,uc) if isinstance(sampler, EDMSampler) else sampler.sampler_step(begin,end,denoiser,x,cond,uc)
                                        
                    t = 0
                    for i in sampling:
                        i.sampling['pic'] = samples[t:t+i.num]
                        print('Finish step ',i.sampling['step'], i.id)
                        i.sampling['step'] = i.sampling['step'] + 1
                        t = t+i.num
                        if i.sampling['step'] >= i.sampling['num_sigmas'] - 1:
                            print('Finish sampling',i.id)
                            i.sample_z = i.sampling['pic']
                            i.state = 3
                    return sampling

def get_usr_input():
    while True:
        usr_input = input()
        if usr_input != '\n':
            if optimatizer.model_name == 'llama':
                cache_size = (llama_args.n_layers ,1, llama_args.max_seq_len, llama_args.n_heads // fs_init.get_model_parallel_world_size(), llama_args.dim // llama_args.n_heads)
                req = request(usr_input,optimatizer.model,cache_size=cache_size)
            elif optimatizer.model_name in VERSION2SPECS:
                #req = sd_request(state=optimatizer.state,img_path='inputs/'+usr_input+'.jpg')
                req = sd_request(state=optimatizer.state,lora_pth='lora_weights/pixel-art-xl.safetensors',prompt=usr_input)
            else:
                raise NotImplementedError
            optimatizer.add_request(req)

if __name__ == "__main__":
    optimatizer = query_optimazer('llama')
    #optimatizer = query_optimazer('svd')
    #optimatizer = query_optimazer('SDXL-lora-1.0')
    input_thread = threading.Thread(target=get_usr_input)
    input_thread.daemon = True
    input_thread.start()
    while True:
        optimatizer.check_prepost()
        optimatizer.runtime(optimatizer.model)