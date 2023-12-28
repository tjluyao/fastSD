import torch
import transformers
from punica import KvCache, KvPool

class TextGeneration:
    def __init__(
        self,
        input_ids: list[int],
        kvpool: KvPool,
        tokenizer,
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
        lora_id = None,
        additionl_init_length : int = 0,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id
        self.additionl_init_length = additionl_init_length

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.kvcache = KvCache(kvpool, self.prompt_len + additionl_init_length)
        self.tokenizer = tokenizer
        self.lora_id = lora_id
        self.prefix_offset = 0
        self.read_offset = 0

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> int:
        if len(self.output_ids) >= self.maxlen:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len

    def decode_tokens(self) -> str:
        # Adapted from: https://github.com/huggingface/text-generation-inference/blob/a5def7c222174e03d815f890093584f3e815c5ce/server/text_generation_server/models/model.py#L68
        prefix_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\uFFFD"):
            new_text = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.output_ids)
            return new_text
        else:
            return ""