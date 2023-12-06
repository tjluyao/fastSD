from pytorch_lightning import seed_everything
from helpers import *
from sgm.modules.diffusionmodules.sampling import *
import time,peft
import threading
import argparse

wait_to_encode = []
wait_to_sample = []
wait_to_decode = []
wait_to_refine = []

class request(object):
    def __init__(self, 
                 prompt, 
                 n_prompt: str='', 
                 num: int=2, 
                 model: str='SDXL-base-1.0',
                 steps: int=30,
                 img_path: str=None,
                 lora_pth: str=None,
                 ) -> None:
        self.id = time.time()
        self.prompt = prompt
        self.negative_prompt = n_prompt
        self.num = num
        self.steps = steps
        self.value_dict = self.get_valuedict(model, prompt, n_prompt)
        self.sampling = {}
        self.sample_z = None
        if lora_pth:
            self.lora_dict= self.get_lora(lora_pth)
        if img_path:
            self.img, self.w, self.h = self.load_img(path=img_path, n=self.num)
        pass

    def get_lora(self, lora_pth):
        lora_dict = load_safetensors(lora_pth)
        try:
            rank = peft.LoraConfig.from_pretrained(lora_pth).r
        except:
            print('Rank not found')
            rank = 8
        return {'weights':lora_dict, 'rank':rank}

    def get_valuedict(self,model,prompt,negative_prompt):
        keys = list(set([x.input_key for x in state["model"].conditioner.embedders]))
        init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
        value_dict = init_embedder_options(
        keys,
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
        )
        value_dict["num"] = self.num
        return value_dict
    
    def load_img(self, path=None, n=1, display=False, device="cuda"):
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

VERSION2SPECS = {
    "SDXL-lora-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_lora.yaml",
        "ckpt": "/root/fastSD/checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
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
}

samplers = [
            "EulerEDMSampler",
            "HeunEDMSampler",
            "EulerAncestralSampler",
            "DPMPP2SAncestralSampler",
            "DPMPP2MSampler",
            "LinearMultistepSampler",
        ]

discretizations=[
            "LegacyDDPMDiscretization",
            "EDMDiscretization",
        ]

def get_batch_req(conditioner, value_dicts, N: Union[List, ListConfig], T = 1,device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}
    standard_dict = value_dicts[0]
    for key in keys:
        if key == "txt":
            prompts = []
            n_prompts = []
            for dict in value_dicts:
                prompts.append(np.repeat(dict['prompt'], repeats=dict['num']))
                n_prompts.append(np.repeat(dict['negative_prompt'], repeats=dict['num']))
            prompts = np.array(prompts)
            n_prompts = np.array(n_prompts)
            batch["txt"] = prompts.reshape(N).tolist()
            batch_uc["txt"] = n_prompts.reshape(N).tolist()
            
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = []
            for dict in value_dicts:
                batch["original_size_as_tuple"].append(
                    torch.tensor([dict["orig_height"], dict["orig_width"]])
                    .to(device)
                    .repeat(dict['num'], 1)
                )
            batch["original_size_as_tuple"] = torch.cat(batch["original_size_as_tuple"], dim=0)

        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = []
            for dict in value_dicts:
                batch["crop_coords_top_left"].append(
                    torch.tensor(
                        [dict["crop_coords_top"], dict["crop_coords_left"]]
                    )
                    .to(device)
                    .repeat(dict['num'], 1)
                )
            batch["crop_coords_top_left"] = torch.cat(batch["crop_coords_top_left"], dim=0)

        elif key == "aesthetic_score":
            batch["aesthetic_score"] = []
            batch_uc["aesthetic_score"] = []
            for dict in value_dicts:
                batch["aesthetic_score"].append(
                    torch.tensor([dict["aesthetic_score"]]).to(device).repeat(dict['num'], 1)
                )
                batch_uc["aesthetic_score"].append(
                    torch.tensor([dict["negative_aesthetic_score"]]).to(device).repeat(dict['num'], 1)
                )
            batch["aesthetic_score"] = torch.cat(batch["aesthetic_score"], dim=0)
            batch_uc["aesthetic_score"] = torch.cat(batch_uc["aesthetic_score"], dim=0)

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = []
            for dict in value_dicts:
                batch["target_size_as_tuple"].append(
                    torch.tensor([dict["target_height"], dict["target_width"]])
                    .to(device)
                    .repeat(dict['num'], 1)
                )
            batch["target_size_as_tuple"] = torch.cat(batch["target_size_as_tuple"], dim=0)
        else:
            batch[key] = standard_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def get_condition(model, 
                  value_dicts,
                  pics = None,
                  sampler = None,
                  skip_encode=False,
                  force_uc_zero_embeddings=[], 
                  additional_kwargs={},
                  batch2model_input=[],
                  offset_noise_level: int = 0.0,
                  add_noise=True,
                  ):
    print("Getting condition for "+str(len(value_dicts))+" reqs")
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                num_samples = [sum(obj['num'] for obj in value_dicts)]

                load_model(model.conditioner)
                batch, batch_uc = get_batch_req(
                    model.conditioner,
                    value_dicts,
                    num_samples,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)
                if pics is not None:
                    for k in c:
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
                    for k in additional_kwargs:
                        c[k] = uc[k] = additional_kwargs[k]

                    if skip_encode:
                        z = pics
                    else:
                        load_model(model.first_stage_model)
                        z = model.encode_first_stage(pics)
                        unload_model(model.first_stage_model)

                    noise = torch.randn_like(z)
                    sigmas = sampler.discretization(sampler.num_steps).cuda()
                    sigma = sigmas[0]
                    #print(f"all sigmas: {sigmas}")
                    #print(f"noising sigma: {sigma}")
                    if offset_noise_level > 0.0:
                        noise = noise + offset_noise_level * append_dims(
                            torch.randn(z.shape[0], device=z.device), z.ndim
                        )
                    if add_noise:
                        noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                        noised_z = noised_z / torch.sqrt(
                            1.0 + sigmas[0] ** 2.0
                        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                    else:
                        noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)
                else:
                    for k in c:
                        if not k == "crossattn":
                            c[k], uc[k] = map(
                                lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                            )

                    shape = (math.prod(num_samples), C, H // F, W // F)
                    noised_z = torch.randn(shape).to("cuda")

                additional_model_inputs = {}
                for k in batch2model_input:
                        additional_model_inputs[k] = batch[k]
                print('Finish condition')
                return noised_z, c, uc, additional_model_inputs

def collect_input():
    global wait_to_encode
    while True:
        user_input = input()
        if user_input != '\n':
            try:
                req = request(user_input, num=2, model='SDXL-base-1.0', steps=30)
            except:
                print('Invalid input')
                continue
            wait_to_encode.append(req)

def collect_batch(img_batch=False,sampler=None):
    global wait_to_encode
    global wait_to_sample
    global wait_to_decode
    with torch.no_grad():
        with autocast("cuda"):
            with state["model"].ema_scope():
                while True:
                    if wait_to_encode:
                        print('Start encoding')
                        encode_process = wait_to_encode
                        wait_to_encode = []
                        encode_process = encode(state["model"], encode_process, sampler=sampler)
                        for i in encode_process:
                            print('Finish encoding',i.id)
                            wait_to_sample.append(i)

                    if wait_to_decode:
                        print('Start decoding')
                        decode_process = wait_to_decode
                        wait_to_decode = []
                        samples_z = torch.cat([i.sampling['pic'] for i in decode_process], dim=0)
                        samples = decode(state["model"],samples_z,filter=state.get('filter'))
                        perform_save_locally(output, samples)  #Save to local file
                        for i in decode_process:
                            print('Saved',i.id)
                            decode_process.remove(i)
                            del i

max_bs = 3
n_slots = 40 * 1024 * 1024
def orca_select(pool,n_rsrv):
    batch = []
    pool = sorted(pool, key=lambda request: request.id)

    for item in pool:
        if len(batch) == max_bs:
            break

        if False:
            new_n_rsrv = n_rsrv 
            if new_n_rsrv > n_slots:
                break
            n_rsrv = new_n_rsrv
        
        batch.append(item)

    return batch, n_rsrv

def encode(model, encode_process, sampler=None):
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                value_dicts = [i.value_dict for i in encode_process]
                z, c, uc, additional_model_inputs= get_condition(
                    model,
                    value_dicts, 
                    sampler=sampler,
                    )
                t = 0
                for i in encode_process:
                    pic = z[t:t+i.num,]
                    ic = {k: c[k][t:t+i.num,] for k in c}
                    iuc = {k: uc[k][t:t+i.num,] for k in uc}
                    ami = {}

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
    return encode_process

def sample(model,sampling):
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

                lora_dicts = []
                for d in sampling:
                    for i in range(d.num):
                        lora_dicts.append(d.lora_dict)  
                lora_dicts = lora_dicts * 2
                        
                def denoiser(input, sigma, c, lora_dicts):
                    return state["model"].denoiser.run_with_lora(state["model"].model, input, sigma, c, lora_dicts)
                samples = sampler.sampler_step_lora(begin,end,denoiser,x,cond,uc, lora_dicts)
                    
                t = 0
                for i in sampling:
                    i.sampling['pic'] = samples[t:t+i.num]
                    print('Finish step ',i.sampling['step'], i.id)
                    i.sampling['step'] = i.sampling['step'] + 1
                    t = t+i.num
    return sampling
            
def decode(model, samples_z, return_latents=False, filter=None):
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples

def test(model,sampler=None,options=None):
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                t = time.time()
                req = request(prompt='crazy frog',lora_pth='/root/fastSD/lora_weights/pixel-art-xl.safetensors')
                process = [req]
                print('load request:',time.time()-t)
                t = time.time()
                process = encode(model, process, sampler=sampler)
                print('encode:',time.time()-t)
                t = time.time()
                while process[0].sampling['step'] < process[0].sampling['num_sigmas'] - 1:
                    process = sample(model,process)
                print('sample:',time.time()-t)
                t = time.time()
                for i in process:
                    out = decode(model, i.sampling['pic'], return_latents=False, filter=None)
                    perform_save_locally(output, out)  #Save to local file
                print('decode:',time.time()-t)
                exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--version', type=str, default='SDXL-lora-1.0', required=False)
    parser.add_argument('--sampler', type=str, default='EulerEDMSampler', required=False)
    parser.add_argument('--output', type=str, default='outputs/', required=False)
    parser.add_argument('--seed', type=int, default=42, required=False)
    parser.add_argument('--default_steps', type=int, default=30, required=False)

    args = parser.parse_args()
    version = args.version
    sampler = args.sampler
    seed = args.seed
    output = args.output
    steps = args.default_steps

    version_dict = VERSION2SPECS[version]
    seed_everything(seed)
    state = init_model(version_dict, load_filter=True)
    model = state["model"]

    stage2strength = None
    finish_denoising = False
    with_refiner = False
    test_model = True

    W = version_dict["W"]
    H = version_dict["H"]
    C = version_dict["C"]
    F = version_dict["f"] 

    sampler = init_sampling(stage2strength=stage2strength, steps=steps, sampler=sampler)

    input_thread = threading.Thread(target=collect_input)
    input_thread.daemon = True
    batch_thread = threading.Thread(target=collect_batch, args=(model,sampler))
    batch_thread.daemon = True

    load_model(state["model"].denoiser)
    load_model(state["model"].model)
    print('Finish init!')

    if not test_model:
        input_thread.start()
        batch_thread.start()
    else:
        test(model,sampler=sampler)

    n_scheduled = 0
    n_rsrv = 0
    while True:
        r_batch = []
        batch,n_rsrv = orca_select(wait_to_sample,n_rsrv)  #batch on req
        if batch and not n_scheduled:

            for i in batch:
                wait_to_sample.remove(i)
            
            r_batch = sample(batch)
            n_scheduled = n_scheduled + 1

        if r_batch:
            for i in r_batch:
                if i.sampling['step'] >= i.sampling['num_sigmas'] - 1:
                    print('Finish sampling',i.id)
                    i.sample_z = i.sampling['pic']
                    if with_refiner:
                        #wait_to_decode.append(i)
                        wait_to_refine.append(i)
                    else:
                        wait_to_decode.append(i)
                else:
                    wait_to_sample.append(i)
            n_scheduled = n_scheduled - 1 

    unload_model(state["model"].denoiser)
    unload_model(state["model"].model)
