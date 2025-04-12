import torch
from torch import nn
import math
import random

from transformers.models.llama.modeling_llama import eager_attention_forward, apply_rotary_pos_emb, repeat_kv, LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Tuple, Optional


# Helper function to inject noise
def inject_redsum_noise(x, stdev):
    if stdev <= 0:
        return x
    noise = torch.randn_like(x) * stdev
    return x + x * noise


# Modified eager attention implementation simulating photonic noise
def noisy_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Bootstrapped from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L180
    Here, we inject simple gaussian noise depending on the value scale of the parameter.

    Two spots to add noise:
        1. After QK^T
        2. After Softmax((Q·K^T)·V/sqrt(d_k))V
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    
    ##################################################################
    ###################### FIRST INJECTION POINT #####################
    # >>> Inject noise into QK^T
    attn_weights = inject_redsum_noise(attn_weights, module.noise_std)
    ##################################################################
    ##################################################################

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    ##################################################################
    ##################### SECOND INJECTION POINT #####################
    # >>> Inject noise into the final attention output
    attn_output = inject_redsum_noise(attn_output, module.noise_std)
    ##################################################################
    ##################################################################

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights




class PhotonicLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, noise_std: float):
        super().__init__(config, layer_idx)
        self.noise_std = noise_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = noisy_eager_attention_forward

        ###### NOTE: SDPA AND FLASHATTENTION NOT MODIFIED
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        attention_interface: Callable = noisy_eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def noisify_attentions(model, noise_std=0.0):
    for name, module in model.named_children():
        if module.__class__.__name__ == "LlamaAttention" or module.__class__.__name__ == "PhotonicLlamaAttention":
            wrapped = PhotonicLlamaAttention(
                config=module.config,
                layer_idx=module.layer_idx,
                noise_std=noise_std
            )
            wrapped.load_state_dict(module.state_dict(), strict=False)

            ### Fix device and dtype issues
            device_of_old_module = module.q_proj.weight.device
            dtype_of_old_module = module.q_proj.weight.dtype
            wrapped.to(device=device_of_old_module, dtype=dtype_of_old_module)

            ### Relace the old module
            setattr(model, name, wrapped)
        else:
            ### Recurse on child modules
            noisify_attentions(module, noise_std)
