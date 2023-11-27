import argparse
from pytorch_lightning import seed_everything
from helper import *

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

def load_img(key=None, n=1, display=False, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        print(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width//n, height//n))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--model', type=str, default='SDXL-base-1.0', required= False)
    parser.add_argument('--prompt', type=str, default='A typical Chinese student', required= False)
    parser.add_argument('--n_prompt', type=str, default='', required= False)
    parser.add_argument('--seed', type=int, default=49, required= False)
    parser.add_argument('--input_img', type=str, default='inputs/00.jpg', required= False)
    parser.add_argument('--output', type=str, default='outputs/', required= False)

    args = parser.parse_args()
    version = args.model
    prompt = args.prompt
    negative_prompt = args.n_prompt
    seed = args.seed
    input = args.input_img
    output = args.output

    version_dict = VERSION2SPECS[version]
    seed_everything(seed)

    state = init_model(version_dict)
    model = state["model"]

    is_legacy = version_dict["is_legacy"]

    stage2strength = None
    finish_denoising = False
    add_pipeline = False
    
    img = load_img(input,n=2)
    H, W = img.shape[2], img.shape[3]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    value_dict = init_embedder_options(
        state["model"].conditioner,
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    strength = 0.75
    sampler = init_sampling(
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_rows = 1
    num_cols = 2
    num_samples = num_rows * num_cols

    samples = do_img2img(
        repeat(img, "1 ... -> n ...", n=num_samples),
        state["model"],
        sampler,
        value_dict,
        num_samples,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=add_pipeline,
        filter=state.get("filter"),
    )

    perform_save_locally(output, samples)