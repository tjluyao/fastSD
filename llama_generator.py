import torch.nn as nn
import json, torch, os, sys
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama.model import Attention, ModelArgs, RMSNorm, FeedForward, TransformerBlock, precompute_freqs_cis
from llama.tokenizer import Tokenizer
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
def parse_model_definition(file_path):
    model_dict = []
    current_layer = ['root']
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('('):
            layer_name,layer = line.split(':') 
            layer_name = layer_name.strip('()')
            layer = layer.strip()
            model_dict.append(
                {
                    'name':layer_name,
                    'layer':layer,
                    'parent':current_layer[-1] if current_layer else None
                }
            )
            if line.endswith('('):
                current_layer.append(layer_name)
        elif line.startswith(')'):
            current_layer.pop()
    return model_dict


def get_layer(name,params):
    if name == 'Attention()':
        return Attention(params)
    elif name == 'RMSNorm()':
        return RMSNorm(params.dim, eps=params.norm_eps)
    elif name == 'FeedForward':
        return FeedForward(params)
    elif name == 'TransformerBlock()':
        return TransformerBlock(params)
    elif name == 'ColumnParallelLinear()':
        return ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)
    elif name == 'ParallelEmbedding()':
        return ParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x:x)
    elif name == 'RowParallelLinear()':
        return RowParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)
    else:
        return None
    
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    @torch.inference_mode()
    def forward_cache(self, item):
        if item.state == 0:
            tokens = torch.tensor(item.buffer[:], dtype=torch.long, device="cuda").unsqueeze(dim=0)
            start_pos = 0
            _bsz, seqlen = tokens.shape
        else:
            l = len(item.buffer) - 1
            tokens = torch.tensor(item.buffer[l:], dtype=torch.long, device="cuda").unsqueeze(dim=0)
            start_pos = l
            _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            cache = [item.cache_k[layer.layer_id,:],item.cache_v[layer.layer_id,:]]
            h = layer.forward_cache(h, start_pos, freqs_cis, mask, cache)
        h = self.norm(h)

        output = self.output(h).float()
        return output




class Large_model():
    def __init__(self,
                    model_structure: str = 'Llama-2-7b/llama_def.txt',
                    model_path: str = 'Llama-2-7b/consolidated.00.pth',
                    tokenizer_path: str = 'Llama-2-7b/tokenizer.model',
                    seed: int = 42,
                ):
        self.model_structure = model_structure
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
                initialize_model_parallel(model_parallel_size)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.manual_seed(seed)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        with open("Llama-2-7b/params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
                    max_seq_len=128,
                    max_batch_size=4,
                    **params,
                )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        model_dict = parse_model_definition(model_structure)
        model = Transformer(model_args)
        for item in model_dict:
            if item['parent'] == 'root':
                if item['layer'].startswith('ModuleList'):
                    setattr(model, item['name'], nn.ModuleList())
                    for i in model_dict:
                        if i['parent'] == item['name']:
                            if len(i['name']) == 1:
                                num = int(i['name'])
                                modellist = getattr(model, item['name'])
                                modellist.append(TransformerBlock(j,model_args))
                            else :
                                num = int(i['layer'].split('x')[0])
                                for j in range(num):
                                    modellist = getattr(model, item['name'])
                                    modellist.append(TransformerBlock(j,model_args))
                else:
                    setattr(model, item['name'], get_layer(item['layer'],model_args))

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        self.model = model
        self.tokenizer = tokenizer

    def generate_iter_cache(
        self,
        batch,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ): 
        params = self.model.params
        for item in batch:
            logits = self.model.forward_cache(item)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            if next_token.item() != self.tokenizer.eos_id:
               item.buffer.append(next_token.item())
            else:
                item.state = 3 # 3 refers to End
            if len(item.buffer) == params.max_seq_len:
                item.state = 3                
        return batch
    
        