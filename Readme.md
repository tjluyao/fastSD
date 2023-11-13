Hello, this is the repo of fast SD.  
Prepare the dependency:  
`pip install -r requirements/pt2.txt`  
Download the pretrained weights:  
`wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true`  
and then move the file `.safetensor` to the folder `./checkpoints`  
Run Text to Image:  
`python txt2img.py --prompt <str>`   
Run Image to Image:  
`python img2img.py --prompt <str> --input_img <img_dir>`   
Run Text to Image on batch:  
`python txt2img_batch.py`  
Run Text to Image on req_answer:  
`python txt2img_req.py`  
and text in the prompt one by one.  