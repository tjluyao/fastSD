# Fast SD  
Hello, this is the repo of QUERY OPTIMIZER for Large Generative Models.  
Prepare the dependency and download the pretrained weights:     
`pip install -r requirements.txt`  
Test inference with Llama-2:    
`python llama_optimizer.py`   
Test inference with Llama-2 with batched-lora input:    
`python llamalora_optimizer.py`   
Test inference with Llava-v1.5:    
`python mm_optimizer.py`  
Test inference with Lynx by ByteDance:    
`python lynx_optimizer.py`    
Test inference with Stable (Video) Diffusion (v1.5/v2.1/XL/XL-turbo):    
`python optimizer_sd.py`   
Test inference with Stable (Video) Diffusion on mutiple GPUs (Still Debugging):    
`python optimizer_multi_sd.py`   