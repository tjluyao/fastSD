import torch, time
from llama.model import ModelArgs as llama_args
import fairscale.nn.model_parallel.initialize as fs_init

class llama_request(object):
    def __init__(self,input,model,cache_size=None) -> None:
        self.input = input
        self.time = time.time()
        self.max_tokens = 128
        self.state = 0   # 0 refers to INITIATION
        self.buffer = []
        if cache_size:
            self.cache_k = torch.zeros(cache_size)
            self.cache_v = torch.zeros(cache_size)

        self.buffer = model.tokenizer.encode(self.input, bos=True, eos=False)

class hf_llama_request(object):
    def __init__(self,input,tokenizer) -> None:
        self.input = input
        self.time = time.time()
        self.max_tokens = 128
        self.state = 0   # 0 refers to INITIATION
        self.buffer = []

        tokens = tokenizer(self.input, return_tensors="pt")
        self.buffer = tokens.input_ids
        from transformers import DynamicCache
        self.cache = DynamicCache()

def get_cache_size(model):
    size = (llama_args.n_layers ,1, llama_args.max_seq_len, llama_args.n_heads // fs_init.get_model_parallel_world_size(), llama_args.dim // llama_args.n_heads)