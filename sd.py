import copy, time
import math
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TT
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from scripts.demo.discretization import (Img2ImgDiscretizationWrapper,
                                         Txt2NoisyDiscretizationWrapper)
from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.modules.diffusionmodules.guiders import (LinearPredictionGuider,
                                                  VanillaCFG)
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler,
                                                   DPMPP2SAncestralSampler,
                                                   EulerAncestralSampler,
                                                   EulerEDMSampler,
                                                   HeunEDMSampler,
                                                   LinearMultistepSampler)
from sgm.util import append_dims, default, instantiate_from_config

lowvram_mode = True
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
        muti_input=False,
        ):
    model = state.get("model")
    T = state.get("T", T)
    C = state.get("C", C)
    H = state.get("H", H)
    W = state.get("W", W)
    F = state.get("F", F)
    print("Getting condition for "+str(len(value_dicts))+" videos")
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                num_batch = sum([obj['num_samples'] for obj in value_dicts])
                #num_frames = sum([obj['T'] for obj in value_dicts])
                num_samples = [num_batch, T] if T is not None else [num_batch]
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    model.conditioner,
                    value_dicts,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                    muti_input=muti_input,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                    muti_input=muti_input,
                )
                

                unload_model(model.conditioner)
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

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                return randn, c, uc, additional_model_inputs
            
def init_model(version_dict, load_ckpt=True, load_filter=True, verbose=True):
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

    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    
    model.eval()
    state["msg"] = None
    state["model"] = model        
    state["ckpt"] = ckpt if load_ckpt else None
    state["config"] = config
    if load_filter:
        state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state

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
embed_watemark = WatermarkEmbedder(WATERMARK_BITS)

def load_model(model):
    model.cuda()

def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
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


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watemark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        #print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        #print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        #print(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 0.0, original_steps=None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        #print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        #print("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        #print(f"sigmas after pruning: ", sigmas)
        return sigmas


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


def init_sampling(
    img2img_strength=None,
    stage2strength=None,
    steps = 40,
    sampler = None,
    discretization = None,
    options: Optional[Dict[str, int]] = None,
):
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
        print("Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
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


def get_interactive_image(key=None) -> Image.Image:
    image = key
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image


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


def get_init_img(batch_size=1, key=None):
    init_image = load_img(key=key).cuda()
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    return init_image


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    model.conditioner,
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
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
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider, (VanillaCFG, LinearPredictionGuider)
                        ):
                            additional_model_inputs[k] = torch.zeros(
                                num_samples[0] * 2, num_samples[1]
                            ).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                if T is None:
                    grid = torch.stack([samples])
                    grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                else:
                    as_vids = rearrange(samples, "(b t) c h w -> b t c h w", t=T)
                    for i, vid in enumerate(as_vids):
                        grid = rearrange(make_grid(vid, nrow=4), "c h w -> h w c")

                if return_latents:
                    return samples, samples_z
                return samples

def get_batc_old(
    conditioner,
    value_dict: dict,
    N: Union[List, ListConfig],
    device: str = "cuda",
    T: int = None,
    additional_batch_uc_fields: List[str] = [],
):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = [value_dict["prompt"]] * math.prod(N)

            batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc

def get_batch_m(conditioner, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}
    num_input = N[0]//len(value_dict["prompt"])

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat(value_dict["prompt"], repeats=num_input)
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat(value_dict["negative_prompt"], repeats=num_input)
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


@torch.no_grad()
def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: int = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    add_noise=True,
):
    print("Sampling")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    model.conditioner,
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to("cuda"), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    load_model(model.first_stage_model)
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)

                noise = torch.randn_like(z)

                sigmas = sampler.discretization(sampler.num_steps).cuda()
                sigma = sigmas[0]

                print(f"all sigmas: {sigmas}")
                print(f"noising sigma: {sigma}")
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

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                grid = embed_watemark(torch.stack([samples]))
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                if return_latents:
                    return samples, samples_z
                return samples
            
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
    return image.to(device) * 2.0 - 1.0


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
