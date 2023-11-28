from pytorch_lightning import seed_everything
from helpers import *
from sgm.modules.diffusionmodules.sampling import *
import time
import threading
import argparse

wait_to_encode = []
wait_to_sample = []
wait_to_decode = []
wait_to_refine = []

class request(object):
    def __init__(self, 
                 img_path: str=None,
                 num_samples = 1,
                 num_frames = 6,
                 ) -> None:
        self.id = time.time()
        self.steps = steps
        keys = list(set([x.input_key for x in state["model"].conditioner.embedders]))
        self.sampling = {}
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num = num_samples * num_frames
        self.value_dict = self.get_valuedict(keys, img_path)
        self.sample_z = None
        pass

    def get_valuedict(self,keys,img):
        value_dict = init_embedder_options(
        keys,
        {},
        )
        img = load_img_for_prediction(W, H, display=False, key=img)
        cond_aug = 0.02
        value_dict["cond_frames_without_noise"] = img
        value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
        value_dict["cond_aug"] = cond_aug
        value_dict['num_samples'] = self.num_samples
        value_dict['T'] = self.num_frames
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

def get_batch_v(conditioner, 
                value_dicts, 
                N: Union[List, ListConfig],
                device="cuda",
                T: int = None,
                additional_batch_uc_fields: List[str] = [],
                ):
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
                prompts.append(
                    repeat(dict['prompt'], "1 ... -> b ...", b=dict['num_samples']*dict['T'])
                    )
                n_prompts.append(
                    repeat(dict['negative_prompt'], "1 ... -> b ...", b=dict['num_samples']*dict['T'])
                    )
            batch[key] = rearrange(batch[key],'b t ... -> (b t) ...')
            batch_uc[key] = rearrange(batch_uc[key],'b t ... -> (b t) ...')
            
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = []
            for dict in value_dicts:
                batch["original_size_as_tuple"].append(
                    torch.tensor([dict["orig_height"], dict["orig_width"]])
                    .to(device)
                    .repeat(dict['num_samples']*dict['T'], 1)
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
                    .repeat(dict['num_samples']*dict['T'], 1)
                )
            batch["crop_coords_top_left"] = torch.cat(batch["crop_coords_top_left"], dim=0)

        elif key == "aesthetic_score":
            batch["aesthetic_score"] = []
            batch_uc["aesthetic_score"] = []
            for dict in value_dicts:
                batch["aesthetic_score"].append(
                    torch.tensor([dict["aesthetic_score"]]).to(device).repeat(dict['num_samples']*dict['T'], 1)
                )
                batch_uc["aesthetic_score"].append(
                    torch.tensor([dict["negative_aesthetic_score"]]).to(device).repeat(dict['num_samples']*dict['T'], 1)
                )
            batch["aesthetic_score"] = torch.cat(batch["aesthetic_score"], dim=0)
            batch_uc["aesthetic_score"] = torch.cat(batch_uc["aesthetic_score"], dim=0)

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = []
            for dict in value_dicts:
                batch["target_size_as_tuple"].append(
                    torch.tensor([dict["target_height"], dict["target_width"]])
                    .to(device)
                    .repeat(dict['num_samples']*dict['T'], 1)
                )
            batch["target_size_as_tuple"] = torch.cat(batch["target_size_as_tuple"], dim=0)

        elif key == "fps":
            batch["fps"] = []
            for dict in value_dicts:
                batch["fps"].append(
                    torch.tensor([dict["fps"]]).to(device).repeat(dict['num_samples']*dict['T'])
                )
            batch["fps"] = torch.cat(batch[key], dim=0)

        elif key == "fps_id":
            batch["fps_id"] = []
            for dict in value_dicts:
                batch["fps_id"].append(
                    torch.tensor([dict["fps_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                )
            batch[key] = torch.cat(batch[key], dim=0)

        elif key == "motion_bucket_id":
            batch[key] = []
            for dict in value_dicts:
                batch[key].append(
                    torch.tensor([dict["motion_bucket_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                )
            batch[key] = torch.cat(batch[key], dim=0)

        elif key == "pool_image":
            batch[key] = []
            for dict in value_dicts:
                batch[key].append(
                    repeat(dict[key], "1 ... -> b ...", b=dict['num_samples']*dict['T']).to(
                device, dtype=torch.half)
                )
            batch[key] = rearrange(batch[key],'b t ... -> (b t) ...')

        elif key == "cond_aug":
            batch[key] = []
            for dict in value_dicts:
                batch[key].append(
                    repeat(torch.tensor([dict["cond_aug"]]).to(device), "1 -> b", b=dict['num_samples']*dict['T'])
                )
            batch[key] = torch.cat(batch[key], dim=0)

        elif key == "cond_frames":
            batch[key] = []
            for dict in value_dicts:
                batch[key].append(
                    repeat(dict["cond_frames"], "1 ... -> b ...", b=dict['num_samples'])
                )
            batch[key] = rearrange(batch[key],'b t ... -> (b t) ...')

        elif key == "cond_frames_without_noise":
            batch[key] = []
            for dict in value_dicts:
                batch[key].append(
                    repeat(dict["cond_frames_without_noise"], "1 ... -> b ...", b=dict['num_samples'])
                )
            batch[key] = rearrange(batch[key],'b t ... -> (b t) ...')
        else:
            batch[key] = standard_dict[key]

    T = sum([dict['T'] for dict in value_dicts])
    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc

def get_condition(model, 
                  value_dicts,
                  sampler = None,
                  force_uc_zero_embeddings=[], 
                  force_cond_zero_embeddings=[],
                  batch2model_input=[],
                  ):
    print("Getting condition")
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                num_samples = sum([obj['num_samples'] for obj in value_dicts])
                num_frames = sum([obj['T'] for obj in value_dicts])
                N = [num_samples, num_frames]

                load_model(model.conditioner)
                batch, batch_uc = get_batch_v(
                    model.conditioner,
                    value_dicts,
                    N,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_model(model.conditioner)
                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(N)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        list_c = []
                        list_uc = []
                        t = 0
                        for dict in value_dicts:
                            x = repeat(c[k][t:t+dict['num_samples'],], "b ... -> b t ...", t=dict['T'])
                            x = rearrange(x, "b t ... -> (b t) ...", t=dict['T'])
                            list_c.append(x)
                            x = repeat(uc[k][t:t+dict['num_samples'],], "b ... -> b t ...", t=dict['T'])
                            x = rearrange(x, "b t ... -> (b t) ...", t=dict['T'])
                            list_uc.append(x)
                            t = t + dict['num_samples']
                        c[k] = torch.cat(list_c, dim=0)
                        uc[k] = torch.cat(list_uc, dim=0)
                
                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert num_frames is not None

                        if isinstance(sampler.guider, (VanillaCFG, LinearPredictionGuider)):
                            additional_model_inputs[k] = torch.zeros(N[0] * 2, N[1]).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(N).to("cuda")
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(N), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                return randn, c, uc, additional_model_inputs

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

def collect_input():
    global wait_to_encode
    while True:
        user_input = input()
        if user_input != '\n':
            req = request(img_path='inputs/00.jpg'
                        )
            wait_to_encode.append(req)

def collect_batch(options=None,sampler=None):
    global wait_to_encode
    global wait_to_sample
    global wait_to_decode
    with torch.no_grad():
        with autocast("cuda"):
            with state["model"].ema_scope():
                while True:
                    if wait_to_encode:
                        encode_process = wait_to_encode
                        wait_to_encode = []

                        value_dicts = [i.value_dict for i in encode_process]
                        z, c, uc, additional_model_inputs= get_condition(state["model"],
                                                     value_dicts, 
                                                     sampler=sampler,
                                                     batch2model_input=["num_video_frames", "image_only_indicator"],
                                                     force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                                                     force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                                                     )
                        
                        for key in c:
                            print(key, c[key].shape)
                        for key in uc:
                            print(key, uc[key].shape)
                        for key in additional_model_inputs:
                            print(key, additional_model_inputs[key].shape if isinstance(additional_model_inputs[key], torch.Tensor) else additional_model_inputs[key])

                        t = 0
                        for i in encode_process:
                            pic = z[t:t+i.num,]
                            ic = {k: c[k][t:t+i.num,] for k in c}
                            iuc = {k: uc[k][t:t+i.num,] for k in uc}
                            ami = {}
                            for k in additional_model_inputs:
                                if k == "image_only_indicator":
                                    ami[k] = additional_model_inputs[k][t:t+i.num * 2,]
                                elif k == "num_video_frames":
                                    ami[k] = i.value_dict['T']

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

max_bs = 3
n_slots = 40 * 1024 * 1024
def orca_select(pool,n_rsrv):
    batch = []
    pool = sorted(pool, key=lambda request: request.id)

    for item in pool:
        if len(batch) == max_bs:
            break

        if False:
            pass
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

                        dict_list = [d.sampling['ami'] for d in sampling] 
                        additional_model_inputs = {}
                        for key in dict_list[0]:
                            if key == "image_only_indicator":
                                additional_model_inputs[key] = torch.cat([d[key] for d in dict_list], dim=0)
                            elif key == "num_video_frames":
                                additional_model_inputs[key] = sum([d[key] for d in dict_list])

                        def denoiser(input, sigma, c):
                            return state["model"].denoiser(state["model"].model, input, sigma, c, **additional_model_inputs)
                        samples = sampler.sampler_step_g(begin,end,denoiser,x,cond,uc) if isinstance(sampler, EDMSampler) else sampler.sampler_step(begin,end,denoiser,x,cond,uc)
                                    
                        t = 0
                        for i in sampling:
                            i.sampling['pic'] = samples[t:t+i.num]
                            print('Finish step ',i.sampling['step'], i.id)
                            i.sampling['step'] = i.sampling['step'] + 1
                            t = t+i.num
                        return sampling

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--version', type=str, default='svd', required=False)
    parser.add_argument('--sampler', type=str, default='EulerEDMSampler', required=False)
    parser.add_argument('--output', type=str, default='outputs/', required=False)
    parser.add_argument('--seed', type=int, default=49, required=False)
    parser.add_argument('--default_steps', type=int, default=10, required=False)

    args = parser.parse_args()
    version = args.version
    sampler = args.sampler
    seed = args.seed
    output = args.output
    steps = args.default_steps

    version_dict = VERSION2SPECS[version]
    seed_everything(seed)
    state = init_model(version_dict, load_filter=True)

    stage2strength = None

    W = 512
    H = 512
    C = version_dict["C"]
    F = version_dict["f"] 
    T = 6

    options = version_dict["options"]
    options["num_frames"] = T
    sampler = init_sampling(stage2strength=stage2strength, options=options,steps=10)

    decoding_t = options.get("decoding_t", T)
    saving_fps = 6

    input_thread = threading.Thread(target=collect_input)
    input_thread.daemon = True
    input_thread.start()

    batch_thread = threading.Thread(target=collect_batch, args=(options,sampler))
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
                    wait_to_decode.append(i)
                else:
                    wait_to_sample.append(i)
            n_scheduled = n_scheduled - 1 

    unload_model(state["model"].denoiser)
    unload_model(state["model"].model)
