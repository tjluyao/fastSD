from pytorch_lightning import seed_everything
from helpers import *
from sgm.modules.diffusionmodules.sampling import *
import time
import peft
import threading
import argparse
from safetensors.torch import load_file as load_safetensors

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
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
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

def get_batch_req(conditioner, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}
    standard_dict = value_dict[0]
    for key in keys:
        if key == "txt":
            prompts = []
            n_prompts = []
            for dict in value_dict:
                prompts.append(np.repeat(dict['prompt'], repeats=dict['num']))
                n_prompts.append(np.repeat(dict['negative_prompt'], repeats=dict['num']))
            prompts = np.array(prompts)
            n_prompts = np.array(n_prompts)
            batch["txt"] = prompts.reshape(N).tolist()
            batch_uc["txt"] = n_prompts.reshape(N).tolist()
            
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([standard_dict["orig_height"], standard_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [standard_dict["crop_coords_top"], standard_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([standard_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([standard_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([standard_dict["target_height"], standard_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
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
    print("Getting condition")
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
            
def refine(refiner, sampler, refine_process, filter=None):
    input = refine_process[0].sample_z
    refine_dicts = []
    with torch.no_grad():
        with autocast("cuda"):
            with refiner.ema_scope():
                        print('Start refining')
                        input = refine_process[0].sample_z
                        refine_dicts = []
                        for i in refine_process:
                            i.value_dict["orig_width"] = input.shape[3] * 8
                            i.value_dict["orig_height"] = input.shape[2] * 8
                            i.value_dict["target_width"] = input.shape[3] * 8
                            i.value_dict["target_height"] = input.shape[2] * 8
                            i.value_dict["crop_coords_top"] = 0
                            i.value_dict["crop_coords_left"] = 0
                            i.value_dict["aesthetic_score"] = 6.0
                            i.value_dict["negative_aesthetic_score"] = 2.5
                            refine_dicts.append(i.value_dict)
                        
                        samples_z = torch.cat([i.sample_z for i in refine_process], dim=0)
                        
                        randn, c, uc, ami= get_condition(refiner,
                                                    refine_dicts,
                                                    samples_z,
                                                    sampler=sampler,
                                                    skip_encode=True,
                                                    add_noise=not finish_denoising)

                        def refine_denoiser(x, sigma, c):
                            return refiner.denoiser(refiner.model, x, sigma, c)

                        load_model(refiner.denoiser)
                        load_model(refiner.model)
                        samples_z = sampler(refine_denoiser, randn, cond=c, uc=uc)
                        unload_model(refiner.model)
                        unload_model(refiner.denoiser)

                        load_model(refiner.first_stage_model)
                        samples_x = refiner.decode_first_stage(samples_z)
                        unload_model(refiner.first_stage_model)
                        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                        if filter is not None:
                            samples = filter(samples)
                        perform_save_locally(output, samples)  #Save to local file
                        print('Finish refinement ', refine_process)
                        return samples

def collect_input():
    global wait_to_encode
    while True:
        user_input = input()
        if user_input != '\n':
            req = request(prompt=user_input,
                        n_prompt='',
                        num=1,
                        lora_pth='lora_weights/pixel-art-xl.safetensors',
                        )
            wait_to_encode.append(req)
            req2 = request(prompt=user_input,
                        n_prompt='',
                        num=1,
                        lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        )
            wait_to_encode.append(req2)

def collect_batch(img_batch=False,sampler=None):
    global wait_to_encode
    global wait_to_sample
    global wait_to_refine
    global wait_to_decode
    with torch.no_grad():
        with autocast("cuda"):
            with state["model"].ema_scope():
                while True:
                    if wait_to_encode:
                        encode_process = wait_to_encode
                        wait_to_encode = []

                        value_dicts = [i.value_dict for i in encode_process]
                        pics = None
                        if img_batch:
                            pics = torch.cat([repeat(i.img, "1 ... -> n ...", n=i.num) for i in encode_process])
                        z, c, uc, ami= get_condition(state["model"],value_dicts, pics=pics,sampler=sampler)
                        t = 0
                        for i in encode_process:
                            pic = z[t:t+i.num,]
                            ic = {k: c[k][t:t+i.num,] for k in c}
                            iuc = {k: uc[k][t:t+i.num,] for k in uc}

                            pic, s_in, sigmas, num_sigmas, ic, iuc = sampler.prepare_sampling_loop(x=pic, cond=ic, uc=iuc, num_steps=i.steps)
                            i.sampling = {  'pic':pic, 
                                            'step':0,
                                            's_in':s_in,
                                            'sigmas':sigmas,
                                            'num_sigmas':num_sigmas,
                                            'c':ic,
                                            'uc':iuc,
                                            'ami':ami,
                                            }
                            wait_to_sample.append(i)
                            t = t + i.num

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

                    if wait_to_refine:
                        refine_process = wait_to_refine
                        wait_to_refine = []
                        refine(state2['model'],sampler2, refine_process)
                    time.sleep(10)

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

def sample(sampling):
    with torch.no_grad():
        with autocast("cuda"):
            with state["model"].ema_scope():
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
                        
                        lora_dicts = [d.lora_dict for d in sampling] * 2 
                        
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
    with_refiner = False
    version_dict = VERSION2SPECS[version]
    seed_everything(seed)
    state = init_model(version_dict)
    is_legacy = version_dict["is_legacy"]

    stage2strength = None
    finish_denoising = False

    if with_refiner:
        version_dict2 = VERSION2SPECS['SDXL-refiner-1.0']
        state2 = init_model(version_dict2, load_filter=False)
        stage2strength = 0.15 #[0.0,1.0]
        sampler2 = init_sampling(key=2,img2img_strength=stage2strength,sampler=sampler)
        finish_denoising = True #Finish denoising with refiner.
        if not finish_denoising:
            stage2strength = None


    W = version_dict["W"]
    H = version_dict["H"]
    C = version_dict["C"]
    F = version_dict["f"] 

    sampler = init_sampling(stage2strength=stage2strength, steps=steps, sampler=sampler)

    input_thread = threading.Thread(target=collect_input)
    input_thread.daemon = True
    input_thread.start()

    batch_thread = threading.Thread(target=collect_batch, args=(False,sampler))
    batch_thread.daemon = True
    batch_thread.start()

    load_model(state["model"].denoiser)
    load_model(state["model"].model)
    print('Finish init!')

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
                        #wait_to_refine.append(i)
                        refine(state2['model'], sampler2, [i], filter=state.get("filter"))
                    else:
                        wait_to_decode.append(i)
                else:
                    wait_to_sample.append(i)
            n_scheduled = n_scheduled - 1 

    unload_model(state["model"].denoiser)
    unload_model(state["model"].model)
