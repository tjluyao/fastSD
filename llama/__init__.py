# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from .generation import Llama, Dialog
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer
import fire
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate_iter_batch(    #Test failed
        self,
        batch,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ): 
        params = self.model.params
        bsz = len(batch)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        max_prompt_len = max([len(t.buffer) for t in batch])
        min_prompt_len = min([len(t.buffer) for t in batch])
        assert max_prompt_len <= params.max_seq_len
        
        total_len = min(params.max_seq_len, 1 + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(batch):
            tokens[k, max_prompt_len - len(t.buffer): max_prompt_len] = torch.tensor(t.buffer, dtype=torch.long, device="cuda")

        logits = self.model.forward(tokens[:, 0 : max_prompt_len], 0)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)

        for i,item in enumerate(next_token):
            if item != self.tokenizer.eos_id:
                batch[i].buffer.append(item.item())
            else:
                batch[i].state = 3 # 3 refers to End
            if len(batch[i].buffer) == params.max_seq_len:
                batch[i].state = 3                
        return batch
    
    @torch.inference_mode()
    def generate_iter(
        self,
        batch,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ): 
        params = self.model.params
        pad_id = self.tokenizer.pad_id
        for item in batch:
            length = len(item.buffer)
            assert  length < params.max_seq_len
            item.buffer.append(pad_id)
            tokens = torch.tensor(item.buffer, dtype=torch.long, device="cuda")
            tokens = tokens.unsqueeze(dim=0)
            #print(tokens.shape)
            #if item.state == 0:

            logits = self.model.forward(tokens[:, 0 : length], 0)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            if next_token.item() != self.tokenizer.eos_id:
               item.buffer[-1] = next_token.item()
            else:
                item.state = 3 # 3 refers to End
            if len(item.buffer) == params.max_seq_len:
                item.state = 3                
        return batch
    
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
    
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token