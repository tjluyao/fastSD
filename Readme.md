Hello, this is the repo of fast SD.  
Prepare the dependency:  
`pip install -r requirements/pt2.txt`  
Download the pretrained weights:  
`wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true`  
and then move the file `.safetensor` to the folder `./checkpoints`  

Test inference with fastSD:    
`python fastSD.py` and text in the prompt one by one.    

Test inference with fastSD on Multiple-LoRA:    
`python fastSD_lora.py` and text in the prompt one by one, remember to set `with_refine = False`    
