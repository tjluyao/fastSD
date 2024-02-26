import time
import peft
import torch
import numpy as np
from torch import autocast
from PIL import Image
import random
from safetensors.torch import load_file as load_safetensors
from sd_helper import (init_embedder_options, 
                       load_img_for_prediction, 
                       EDMSampler, 
                       append_dims,
                       get_condition, 
                       load_model, 
                       unload_model,
                       save_video_as_grid_and_mp4,
                       perform_save_locally)

from default_optimizer import default_optimazer

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-lora-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_lora.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
    "svd": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_image_decoder": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_xt": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd_xt.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
    "svd_xt_image_decoder": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_xt_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
}

class sd_request():
    def __init__(
            self, 
            state: dict,
            steps: int = 10,
            video_task: bool = False,
            prompt: str = None,
            negative_prompt: str = None,
            image: str|Image.Image|np.ndarray = None,
            lora_pth: str = None,
            output_path: str = './outputs',
            num_samples : int = 2,
            ) -> None:
        self.output = None
        self.output_path = output_path
        self.time = time.time()
        self.id = self.time
        self.state = 0
        self.steps = steps
        keys = list(set([x.input_key for x in state["model"].conditioner.embedders]))
        self.sampling = {}
        self.num_samples = num_samples
        self.num_frames = state['T'] if video_task else 1
        self.num = self.num_samples * self.num_frames
        if lora_pth:
            self.lora_dict= self.get_lora(lora_pth)
        else:
            self.lora_dict = None
        if image and not video_task:
            self.img, self.w, self.h = self.load_img(path=image)
        
        W = self.w if hasattr(self,'w') else state['W']
        H = self.h if hasattr(self,'h') else state['H']
        negative_prompt = negative_prompt if negative_prompt else ''
       
        self.value_dict = self.get_valuedict(keys, 
                                             image, 
                                             W, H, 
                                             video_task=video_task, 
                                             prompt=prompt, 
                                             negative_prompt=negative_prompt)
        
        self.time_list=[self.time]

    def get_valuedict(self,
                      keys,
                      img,
                      W,
                      H,
                      video_task=False, 
                      prompt=None, 
                      negative_prompt=None):
        if video_task:
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
    
    def load_img(self, path=None, display=False, device="cpu", standard = 512):
        if isinstance(path,str):
            image = Image.open(path)
        else:
            image = path
        if display:
            print(image)
        width, height = image.size
        #width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((standard, standard))
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        return image.to(device), width, height
    
    def get_lora(self, lora_pth):
        lora_dict = load_safetensors(lora_pth)
        try:
            rank = peft.LoraConfig.from_pretrained(lora_pth).r
        except:
            print('Rank not found')
            rank = 8
        return {'weights':lora_dict, 'rank':rank}
    
class sd_optimizer(default_optimazer):
    def __init__(self, model_name: str, batch_option: int = 1, max_batch_size: int = 10, seed: int = 49, device: str = 'cuda', **kwargs):
        super().__init__(model_name, batch_option, max_batch_size, seed, device, **kwargs)
    
    def init_model(self, **kwargs):
        version_dict = VERSION2SPECS[self.model_name]
        from sd_helper import init_model as init_sd
        from sd_helper import init_sampling
        state = init_sd(version_dict, load_filter=True)
        steps = kwargs.get('steps',30)
        state['W'] = version_dict.get('W', 1024)
        state['H'] = version_dict.get('H', 1024)
        state['C'] = version_dict['C']
        state['F'] = version_dict['f']
        state['T'] = 6 if self.model_name in ['svd'] else None
        state['options'] = version_dict.get('options', {})
        state['options']["num_frames"] = state['T']
        sampler = init_sampling(options=state['options'],steps=steps)
        state['sampler'] = sampler
        img_sampler = init_sampling(options=state['options'],
                                    img2img_strength=0.75,
                                    steps=steps)
        state['img_sampler'] = img_sampler
        state['saving_fps'] = 6
        self.state = state
        self.model = state['model']
        load_model(state['model'].model)
        load_model(state['model'].denoiser)
        print('Model loaded')

    def iteration(self, sampling, **kwargs):
        model = self.model
        
        sampler = self.state['img_sampler'] if hasattr(sampling[0],'img') else self.state['sampler']
        T = self.state.get('T')
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
                        return self.state["model"].denoiser(self.state["model"].model, input, sigma, c, **additional_model_inputs)
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
                            i.state = 2
                    return sampling
    
    def preprocess(self, encode_process, **kwargs):
        state = self.state
        model = state.get('model')
        is_image = hasattr(encode_process[0],'img')
        sampler = state['img_sampler'] if is_image else state['sampler']
        options = state.get('options')
        T = state.get('T')
        muti_input = state.get('muti_input', True)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    value_dicts = [i.value_dict for i in encode_process]
                    batch2model_input = ["num_video_frames", "image_only_indicator"] if T else []
                    if is_image:
                        imgs = []
                        for req in encode_process:
                            imgs += [req.img]*req.num
                        imgs = torch.cat(imgs, dim=0)
                    z, c, uc, additional_model_inputs= get_condition(
                        state,
                        value_dicts, 
                        sampler=sampler,
                        T=T,
                        batch2model_input=batch2model_input,
                        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                        muti_input=muti_input,
                        imgs=imgs if is_image else None,
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
                        self.wait_runtime.append(i)
        return encode_process
    
    def postprocess(self,decode_process,**kwargs):
        state = self.state
        model = state.get('model')
        filter = state.get('filter', None)
        return_latents = kwargs.get('return_latents', False)
        samples_z = torch.cat([req.sampling['pic'] for req in decode_process], dim=0)
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
        else:
            t = 0
            for req in decode_process:
                if self.state['T']:
                    output = samples[t:t+req.num]
                    save_video_as_grid_and_mp4(output, req.output_path, self.state['T'], self.state['saving_fps'])
                else:
                    output = samples[t:t+req.num]
                    #perform_save_locally(req.output_path, output)
                print('Saved',req.id)
                data_log.append(time.time()-req.time)
                t = t + req.num
                del req
    def update_input(self):
        if len(self.wait_preprocess) < self.batch_option:
            choice = random.choice(sentences)
            req = sd_request(
                        state=optimizer.state,
                        prompt=choice,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/03.jpg',
                        num_samples=1,
                    )
            self.wait_preprocess.append(req) 

    
if __name__ == '__main__':
    optimizer = sd_optimizer(
        model_name='SD-2.1',
        batch_option=32,
        max_batch_size=32,
        )

    mode = 'test'
    if mode == 'server':
        def get_usr_input():
            while True:
                usr_input = input()
                if usr_input != '\n':
                    req = sd_request(
                        state=optimizer.state,
                        prompt=usr_input,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/03.jpg',
                    )
                    optimizer.wait_preprocess.append(req) 

        import threading
        t = threading.Thread(target=get_usr_input)
        t.daemon = True
        t.start()
    elif mode == 'test':
        from datasets import load_dataset
        dataset = load_dataset('lambdalabs/pokemon-blip-captions')
        sentences = dataset['train']['text']
        data_log = []

    while True:
        if mode == 'test':
            optimizer.update_input()
            if len(data_log) > 200:
                break
        optimizer.check_prepost()
        optimizer.runtime()

    with open('data_log.json','w') as f:
        import json
        data = {
                'batch_size':optimizer.batch_option,
                'iteration_batch_size':optimizer.max_batch_size,
                'data':data_log,
        }
        json.dump(data,f,indent=4)