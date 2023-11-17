import argparse
from pytorch_lightning import seed_everything
from helpers import *

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--model', type=str, default='SDXL-base-1.0', required= False)
    parser.add_argument('--prompt', type=str, default='ID photo, Typical Chinese boy, Undergraduate student, 8k', required= False)
    parser.add_argument('--n_prompt', type=str, default='', required= False)
    parser.add_argument('--seed', type=int, default=49, required= False)
    parser.add_argument('--output', type=str, default='outputs/', required= False)

    args = parser.parse_args()
    version = args.model
    prompt = args.prompt
    negative_prompt = args.n_prompt
    seed = args.seed
    output = args.output

    version_dict = VERSION2SPECS[version]
    seed_everything(seed)

    prompts = ['office lady sitting, black stocking, 8k, high resolution','super girl standing, short skirt, 8k, high resolution']
    negative_prompts = ['','']

    state = init_model(version_dict)
    model = state["model"]

    is_legacy = version_dict["is_legacy"]

    stage2strength = None
    finish_denoising = False
    add_pipeline = False
    
    W = version_dict["W"]
    H = version_dict["H"]
    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    keys = list(set([x.input_key for x in state["model"].conditioner.embedders]))

    states = [keys,init_dict]
    value_dict = init_embedder_options(
        keys,
        init_dict,
        prompt=prompts,
        negative_prompt=negative_prompts,
    )

    sampler = init_sampling(stage2strength=stage2strength, steps=30)
    num_rows = 1
    num_cols = 2 * len(prompts)
    num_samples = num_rows * num_cols

    samples = do_sample(
        state["model"],
        sampler,
        value_dict,            
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=add_pipeline,
        filter=state.get("filter"),
    )

    perform_save_locally(output, samples)