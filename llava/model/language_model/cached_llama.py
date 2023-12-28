#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM,LlamaConfig
from .llavamodel import LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from punica.utils import BatchedKvCache, BatchLenInfo

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
    def forward(
        self,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            torch.cuda.nvtx.range_push("embed")
            hidden_states = self.embed_tokens(input_ids)
            torch.cuda.nvtx.range_pop()
        else:
            hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            hidden_states = decoder_layer(hidden_states, blen, prefill_kv, decode_kv)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states



class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #print(input_ids.shape)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        #print(inputs_embeds.shape if inputs_embeds is not None else input_ids.shape)
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states = self.model(blen=blen, 
                                   prefill_kv=prefill_kv, 
                                   decode_kv=decode_kv,
                                   input_ids = input_ids, 
                                   inputs_embeds = inputs_embeds)
        torch.cuda.nvtx.range_push("lm_head")
        logits = self.lm_head(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return logits, hidden_states


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

#AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
