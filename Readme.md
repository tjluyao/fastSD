Hello, this is the repo of QUERY OPTIMIZER for Large Generative Models.  
Prepare the dependency and download the pretrained weights:     
`pip install -r requirements.txt`  
Test inference with Llama-2:    
`python llama_optimizer.py`   
Test inference with Llama-2 with batch-lora:    
`python llamalora_optimizer.py`   
Test inference with Llava:    
`python mm_optimizer.py`  
Test inference with Lynx by ByteDance:    
`python lynx_optimizer.py`    
Test inference with Stable (Video) Diffusion:    
`python optimizer_sd.py`   
Test inference with Stable (Video) Diffusion on mutiple GPUs:    
`python optimizer_multi_sd.py`   