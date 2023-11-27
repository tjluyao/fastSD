This experiment is based on the LLM Llama-2-7b and ORCA.  
To download the pre-trained Llama, use:  
`python model_download.py`  
To run the inference, use:  
`torchrun --nproc_per_node 1 Orca_test.py`  