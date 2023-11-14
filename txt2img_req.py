from pytorch_lightning import seed_everything
from helper_batch import *
import time
import threading

class request(object):
    def __init__(self, 
                 prompt, 
                 n_prompt: str='', 
                 num: int=2, 
                 model: str='SDXL-base-1.0',
                 steps: int=30) -> None:
        self.id = time.time()
        self.prompt = prompt
        self.negative_prompt = n_prompt
        self.num = num
        self.steps = steps
        self.value_dict = self.get_valuedict(model, prompt, n_prompt)
        self.sampling = {}
        pass

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

def get_condition(model, value_dicts, force_uc_zero_embeddings=[], batch2model_input=[]):
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

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

    return randn, c, uc, additional_model_inputs

def sample_batch(model, input, c ,uc, additional_model_inputs):
    print('Start sampling')
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, input, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                return samples_z

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

                grid = torch.stack([samples])
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                if return_latents:
                    return samples, samples_z
                return samples

def collect_input():
    global wait_to_encode
    while True:
        user_input = input()
        if user_input != '\n':
            req = request(prompt=user_input,
                        n_prompt='',
                        num=2,
                        )
            wait_to_encode.append(req)

def collect_batch():
    global wait_to_encode
    global wait_to_decode
    while True:
        if wait_to_encode:
            on_process = wait_to_encode
            wait_to_encode = []

            value_dicts = []
            for i in on_process:
                value_dicts.append(i.value_dict)

            randn, c, uc, ami= get_condition(state["model"],value_dicts)
            t = 0
            for i in on_process:
                pic = randn[t:t+i.num,]
                ic = {k: c[k][t:t+i.num,] for k in c}
                iuc = {k: uc[k][t:t+i.num,] for k in uc}

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
                sampling.append(i)
                t = t + i.num

        if wait_to_decode:
            print('Start decoding')
            samples_z = torch.cat([i.sampling['pic'] for i in wait_to_decode], dim=0)
            samples = decode(state["model"],samples_z)
            perform_save_locally(output, samples)
            for i in wait_to_decode:
                wait_to_decode.remove(i)
                print('Saved',i.id)
                del i

        time.sleep(10)

wait_to_encode = []
sampling = []
wait_to_decode = []
    
if __name__ == '__main__':
    version = 'SDXL-base-1.0'
    seed = 49
    output = 'outputs/'
    steps = 30

    version_dict = VERSION2SPECS[version]
    seed_everything(seed)
    state = init_model(version_dict)
    is_legacy = version_dict["is_legacy"]

    stage2strength = None
    finish_denoising = False
    add_pipeline = False
    
    W = version_dict["W"]
    H = version_dict["H"]
    C = version_dict["C"]
    F = version_dict["f"] 

    sampler = init_sampling(stage2strength=stage2strength, steps=steps)

    input_thread = threading.Thread(target=collect_input)
    input_thread.daemon = True
    input_thread.start()

    batch_thread = threading.Thread(target=collect_batch)
    batch_thread.daemon = True
    batch_thread.start()

    print('Finish init!')
    while True:
        if sampling:
            x = torch.cat([i.sampling['pic'] for i in sampling], dim=0)
            begin = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']] for i in sampling], dim=0)
            end = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step'] + 1] for i in sampling], dim=0)

            dict_list = [d.sampling['c'] for d in sampling]
            cond = {}
            for key in dict_list[0]:
                cond[key] = torch.cat([d[key] for d in dict_list], dim=0)

            dict_list = [d.sampling['uc'] for d in sampling]
            uc = {}
            for key in dict_list[0]:
                uc[key] = torch.cat([d[key] for d in dict_list], dim=0)

            #gamma = min(sampler.s_churn / (i.sampling['num_sigmas'] - 1), 2**0.5 - 1) if sampler.s_tmin <= i.sampling['sigmas'][i.sampling['step']] <= sampler.s_tmax else 0.0
            gamma = 0.0
            print('Start sampling step')
            with torch.no_grad():
                with autocast("cuda"):
                    with state["model"].ema_scope():
                        def denoiser(input, sigma, c):
                            return state["model"].denoiser(state["model"].model, input, sigma, c, **{})

                        load_model(state["model"].denoiser)
                        load_model(state["model"].model)
                        samples = sampler.sampler_step(
                            begin,
                            end,
                            denoiser,
                            x,
                            cond,
                            uc,
                            gamma,
                            )
                        unload_model(state["model"].model)
                        unload_model(state["model"].denoiser)
            t = 0
            for i in sampling:
                i.sampling['pic'] = samples[t:t+i.num]
                i.sampling['step'] = i.sampling['step'] + 1
                if  i.sampling['step'] == i.sampling['num_sigmas'] - 1:
                    print('Finish sampling',i.id)
                    sampling.remove(i)
                    wait_to_decode.append(i)
                t = t+i.num
