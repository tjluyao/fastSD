import torch 
from transformers import LlamaConfig
from punica import LoraWeight

lora_paths = {
    'fin':'lora_weights/fingpt-forecaster_dow30_llama2-7b_lora',
    'Chinese':'lora_weights/Chinese-Llama-2-LoRA-7B',
}

class LlamaLoraWeight:
    def __init__(
        self,
        config: LlamaConfig,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.q = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.k = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.o = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.gate = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.up = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.down = LoraWeight(
            config.num_hidden_layers,
            config.intermediate_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        if ts["q.A"] is not None:
            self.q.copy_from_tensor(ts["q.A"], ts["q.B"])
        if ts["k.A"] is not None:
            self.k.copy_from_tensor(ts["k.A"], ts["k.B"])
        if ts["v.A"] is not None:
            self.v.copy_from_tensor(ts["v.A"], ts["v.B"])
        if ts["o.A"] is not None:
            self.o.copy_from_tensor(ts["o.A"], ts["o.B"])
        if ts["gate.A"] is not None:
            self.gate.copy_from_tensor(ts["gate.A"], ts["gate.B"])
        if ts["up.A"] is not None:
            self.up.copy_from_tensor(ts["up.A"], ts["up.B"])
        if ts["down.A"] is not None:
            self.down.copy_from_tensor(ts["down.A"], ts["down.B"])

def weight_convert(weights,rank):
    qA,qB = [],[]
    kA,kB = [],[]
    vA,vB = [],[]
    oA,oB = [],[]
    gateA,gateB = [],[]
    upA,upB = [],[]
    downA,downB = [],[]
    for key in weights.keys():
        if 'q_proj' in key:
            if 'A' in key:
                qA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                qB.append(weights[key].unsqueeze(0))
        if 'k_proj' in key:
            if 'A' in key:
                kA.append(weights[key].unsqueeze(0))    
            if 'B' in key:
                kB.append(weights[key].unsqueeze(0))
        if 'v_proj' in key:
            if 'A' in key:
                vA.append(weights[key].unsqueeze(0))    
            if 'B' in key:
                vB.append(weights[key].unsqueeze(0))
        if 'o_proj' in key:
            if 'A' in key:
                oA.append(weights[key].unsqueeze(0)) 
            if 'B' in key:
                oB.append(weights[key].unsqueeze(0))
        if 'gate_proj' in key:
            if 'A' in key:
                gateA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                gateB.append(weights[key].unsqueeze(0))
        if 'up_proj' in key:
            if 'A' in key:
                upA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                upB.append(weights[key].unsqueeze(0))
        if 'down_proj' in key:
            if 'A' in key:
                downA.append(weights[key].unsqueeze(0))
            if 'B' in key:
                downB.append(weights[key].unsqueeze(0))
    weights = {
        'q.A':torch.cat(qA, dim=0) if qA else None,
        'q.B':torch.cat(qB, dim=0) if qB else None,
        'k.A':torch.cat(kA, dim=0) if kA else None,
        'k.B':torch.cat(kB, dim=0) if kB else None,
        'v.A':torch.cat(vA, dim=0) if vA else None,
        'v.B':torch.cat(vB, dim=0) if vB else None,
        'o.A':torch.cat(oA, dim=0) if oA else None,
        'o.B':torch.cat(oB, dim=0) if oB else None,
        'gate.A':torch.cat(gateA, dim=0) if gateA else None,
        'gate.B':torch.cat(gateB, dim=0) if gateB else None,
        'up.A':torch.cat(upA, dim=0) if upA else None,
        'up.B':torch.cat(upB, dim=0) if upB else None,
        'down.A':torch.cat(downA, dim=0) if downA else None,
        'down.B':torch.cat(downB, dim=0) if downB else None,
    }
    if rank == 8:
        for key in weights.keys():
            if weights[key] is not None:
                if 'A' in key:
                    complement = torch.zeros_like(weights[key])
                    weights[key] = torch.cat([weights[key], complement], dim=1)
                if 'B' in key:
                    complement = torch.zeros_like(weights[key])
                    weights[key] = torch.cat([weights[key], complement], dim=2)
    return weights