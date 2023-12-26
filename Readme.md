Hello, this is the repo of fast SD.  
Prepare the dependency:  
`pip install -r requirements.txt`  
Download the pretrained weights:  
`wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true`  
and then move the file `.safetensor` to the folder `./checkpoints`  

Test inference with fastSD:    
`python fastSD.py` and text in the prompt one by one.    

Test inference with fastSD on Video:  
`python fastSD_sdv.py` and text in the name of the picture inside the folder '/inputs' one by one.   
