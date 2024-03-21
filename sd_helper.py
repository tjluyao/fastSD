import torch
import copy
import math
import os, io
from torch import autocast
from torchvision import transforms
from safetensors.torch import load_file as load_safetensors
from einops import rearrange, repeat
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import ListConfig, OmegaConf
from sgm.util import append_dims, default, instantiate_from_config
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from scripts.demo.discretization import (Img2ImgDiscretizationWrapper,
                                         Txt2NoisyDiscretizationWrapper)
from sgm.modules.diffusionmodules.guiders import (LinearPredictionGuider,
                                                  VanillaCFG)
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler,
                                                   DPMPP2SAncestralSampler,
                                                   EulerAncestralSampler,
                                                   EulerEDMSampler,
                                                   HeunEDMSampler,
                                                   LinearMultistepSampler,
                                                   EDMSampler,
                                                   )
from sgm.modules.autoencoding.temporal_ae import VideoDecoder

def load_model(model,location=None):
    if location:
        model.to(location)
    else:
        model.cuda()

def unload_model(model):
    model.cpu()
    torch.cuda.empty_cache()

def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future
    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ''

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]
        
        if key in ["fps_id", "fps"]:
            fps = 6

            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1

        if key == "motion_bucket_id":
            mb_id = 127
            value_dict["motion_bucket_id"] = mb_id

        if key == "pool_image":
            image = load_img(
                key="pool_image_input",
                size=224,
                center_crop=True,
            )
            if image is None:
                image = torch.zeros(1, 3, 224, 224)
            value_dict["pool_image"] = image

    return value_dict

def init_model(version_dict, load_ckpt=True, load_filter=True, verbose=True, dtype_half=True):
    state = dict()
    config = version_dict["config"]
    ckpt = version_dict["ckpt"]
    config = OmegaConf.load(config)

    model = instantiate_from_config(config.model)
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                global_step = pl_sd["global_step"]
                print(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    if dtype_half:
        model.model.half()
        #model.first_stage_model.half()
    
    model.eval()
    state["msg"] = None
    state["model"] = model        
    state["ckpt"] = ckpt if load_ckpt else None
    state["config"] = config
    if load_filter:
        state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state

def load_img(display=True, 
             size:Union[None, int, Tuple[int, int]] = None,
             center_crop: bool = False,
             ):
    image = get_interactive_image()
    if image is None:
        return None
    if display:
        print(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")

    transform = []
    if size is not None:
        transform.append(transforms.Resize(size))
    if center_crop:
        transform.append(transforms.CenterCrop(size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Lambda(lambda x: 2.0 * x - 1.0))

    transform = transforms.Compose(transform)
    img = transform(image)[None, ...]
    print(f"input min/max/mean: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}")
    return img

def get_interactive_image(key=None) -> Image.Image:
    image = key
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        elif isinstance(image, list):
            image = Image.fromarray(np.array(image))
        else:
            raise ValueError(f"unknown image type {type(image)}")
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image
    
def get_guider(options):
    guiders = [
            "VanillaCFG",
            "IdentityGuider",
            "LinearPredictionGuider",
        ]
    guider = guiders[options.get("guider", 0)]
    additional_guider_kwargs = options.pop("additional_guider_kwargs", {})

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = options.get("cfg", 5.0)#cfg-scale, default 5.0, min 0.0
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale,**additional_guider_kwargs,},
        }
    elif guider == "LinearPredictionGuider":
        max_scale = options.get("cfg", 1.5) #cfg-scale, default 1.5, min 1.0
        min_scale = options.get("min_cfg", 1.0) #cfg-scale, default 1.0, min 1.0, max 10.0
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    else:
        raise NotImplementedError
    return guider_config

class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
        ]
        uc = cond
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc
    
def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn

class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)
    
def init_sampling(
    img2img_strength=None,
    stage2strength=None,
    steps = 30,
    sampler = None,
    discretization = None,
    options: Optional[Dict[str, int]] = None,
    turbo : bool = False,
    **kwargs
):
    if turbo:
        sampler = SubstepSampler(
        n_sample_steps=steps,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
            ),
        )
        sampler.noise_sampler = SeededNoise(seed=kwargs.get("seed", 49))
        return sampler

    options = {} if options is None else options
    discretizations = [
        "LegacyDDPMDiscretization",
        "EDMDiscretization",
    ]
    discretization = discretizations[options.get("discretization", 0)]
    discretization_config = get_discretization(discretization,options=options)

    guider_config = get_guider(options=options)
    samplers = [
        "EulerEDMSampler",
        "HeunEDMSampler",
        "EulerAncestralSampler",
        "DPMPP2SAncestralSampler",
        "DPMPP2MSampler",
        "LinearMultistepSampler",
    ]
    sampler = samplers[options.get("sampler", 0)]
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    if img2img_strength is not None:
        #print("Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )
    return sampler


def get_discretization(discretization, options=None):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = options.get("sigma_min", 0.0292)  # 0.0292
        sigma_max = options.get("sigma_max", 14.6146)  # 14.6146
        rho = options.get("rho", 3.0)
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config


def get_sampler(sampler_name, steps, discretization_config, guider_config):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        s_noise = 1.0
        eta = 1.0

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        order = 4
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler

def get_batch(conditioner, 
                value_dicts, 
                N: Union[List, ListConfig] = None,
                device="cuda",
                T: int = None,
                additional_batch_uc_fields: List[str] = [],
                muti_input=False,
                ):
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}
    
    standard_dict = value_dicts[0]
    for key in keys:
        if key == "txt":
            prompts = []
            n_prompts = []
            for dict in value_dicts:
                prompts = prompts + [dict['prompt']] * dict['num']
                n_prompts = n_prompts + [dict['negative_prompt']] * dict['num']
            batch["txt"] = prompts
            batch_uc["txt"] = n_prompts

            
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
            batch[key] = []
            if muti_input:
                batch_uc[key] = []
                for dict in value_dicts:
                    tensor = torch.tensor([dict["fps_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
                    batch_uc[key].append(tensor)
            else:
                for dict in value_dicts:
                    tensor = torch.tensor([dict["fps_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
                batch[key] = torch.cat(batch[key], dim=0)

        elif key == "motion_bucket_id":
            batch[key] = []
            if muti_input:
                batch_uc[key] = []
                for dict in value_dicts:
                    tensor = torch.tensor([dict["motion_bucket_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
                    batch_uc[key].append(tensor)
            else:
                for dict in value_dicts:
                    tensor = torch.tensor([dict["motion_bucket_id"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
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
            if muti_input:
                batch_uc[key] = []
                for dict in value_dicts:
                    tensor = torch.tensor([dict["cond_aug"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
                    batch_uc[key].append(tensor)
            else:
                for dict in value_dicts:
                    tensor = torch.tensor([dict["cond_aug"]]).to(device).repeat(dict['num_samples']*dict['T'])
                    batch[key].append(tensor)
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

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc

def get_condition(
        state, 
        value_dicts,
        sampler = None,
        force_uc_zero_embeddings: Optional[List] = None,
        force_cond_zero_embeddings: Optional[List] = None,
        batch2model_input: List = None,
        T=None,
        C=None,
        H=None,
        W=None,
        F=None,
        additional_batch_uc_fields=None,
        imgs = None,
        skip_encode=False,
        add_noise=True,
        offset_noise_level=0.0,
        lowvram_mode=False,
        turbo=False,
        **kwargs
        ):
    model = state.get("model", None)
    T = state.get("T", T)
    C = state.get("C", C)
    H = state.get("H", H)
    W = state.get("W", W)
    F = state.get("F", F)
    #print("Getting condition for "+str(len(value_dicts))+" videos")
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                num_batch = sum([obj['num_samples'] for obj in value_dicts])
                num_samples = [num_batch, T] if T is not None else [num_batch]
                if lowvram_mode:
                    load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    model.conditioner,
                    value_dicts,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                    muti_input= True if T is not None else False,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                    muti_input= True if T is not None else False,
                )
                if lowvram_mode:
                    unload_model(model.conditioner)
                
                if turbo:
                    shape = (math.prod(num_samples), C, H // F, W // F)
                    randn = seeded_randn(shape, kwargs.get("seed", 49))
                    return randn, c, uc, {}

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
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
                        assert T is not None

                        if isinstance(sampler.guider, (VanillaCFG, LinearPredictionGuider)):
                            additional_model_inputs[k] = torch.zeros(num_batch*2, T).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to("cuda")
                    else:
                        additional_model_inputs[k] = batch[k]
                if imgs is None:
                    shape = (math.prod(num_samples), C, H // F, W // F)
                    randn = torch.randn(shape).to("cuda")
                    return randn, c, uc, additional_model_inputs
                else:
                    if skip_encode:
                        z = imgs
                    else:
                        if lowvram_mode:
                            load_model(model.first_stage_model)
                        imgs = imgs.to(model.first_stage_model.device)
                        z = model.encode_first_stage(imgs)
                        z = z.cuda()
                        if lowvram_mode:
                            unload_model(model.first_stage_model)
                    noise = torch.randn_like(z)
                    sigmas = sampler.discretization(sampler.num_steps).cuda()
                    sigma = sigmas[0]
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
                    return noised_z, c, uc, additional_model_inputs

from imwatermark import WatermarkEncoder
class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor):
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, C, H, W) in range [0, 1]

        Returns:
            same as input but watermarked
        """
        # watermarking libary expects input as cv2 BGR format
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange(
            (255 * image).detach().cpu(), "n b c h w -> (n b) h w c"
        ).numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(
            rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)
        ).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)

import numpy as np
def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watermark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1

def get_resizing_factor(
    desired_shape: Tuple[int, int], current_shape: Tuple[int, int]
) -> float:
    r_bound = desired_shape[1] / desired_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if aspect_r >= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r < 1.0:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)
    else:
        if aspect_r <= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r > 1:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)

    return factor

import torchvision.transforms as TT
def load_img_for_prediction(
    W: int, H: int, display=True, key=None, device="cuda"
) -> torch.Tensor:
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        print(image)
    w, h = image.size

    image = np.array(image).transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)

    rfs = get_resizing_factor((H, W), (h, w))
    resize_size = [int(np.ceil(rfs * s)) for s in (h, w)]
    top = (resize_size[0] - H) // 2
    left = (resize_size[1] - W) // 2

    image = torch.nn.functional.interpolate(
        image, resize_size, mode="area", antialias=False
    )
    image = TT.functional.crop(image, top=top, left=left, height=H, width=W)

    if display:
        numpy_img = np.transpose(image[0].numpy(), (1, 2, 0))
        pil_image = Image.fromarray((numpy_img * 255).astype(np.uint8))
        print(pil_image)
    return image.to(device) * 2.0 - 1.0, w, h

import cv2
from glob import glob
from torchvision.utils import make_grid, save_image
def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5
):
    os.makedirs(save_path, exist_ok=True)
    base_count = len(glob(os.path.join(save_path, "*.mp4")))

    video_batch = rearrange(video_batch, "(b t) c h w -> b t c h w", t=T)
    video_batch = embed_watermark(video_batch)
    for vid in video_batch:
        save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)

        video_path = os.path.join(save_path, f"{base_count:06d}.mp4")

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps,
            (vid.shape[-1], vid.shape[-2]),
        )

        vid = (
            (rearrange(vid, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
        )
        for frame in vid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        video_path_h264 = video_path[:-4] + "_h264.mp4"
        os.system(f"ffmpeg -i {video_path} -c:v libx264 {video_path_h264}")

        with open(video_path_h264, "rb") as f:
            video_bytes = f.read()
        print(video_bytes)

        base_count += 1

def image_to_video(imgs,fps=5,size=None):
    path = "temp.mp4"
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        5,
        (imgs.shape[-1], imgs.shape[-2]),
    )
    imgs = (
        (rearrange(imgs, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    )
    for frame in imgs:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    writer.release()
    return path