"""DS MoE/LoRA operations, preserving the native dispatcher contracts.

Inference only. Inputs/outputs remain on the caller's device and current stream.
The native dispatcher preserves the DS input layouts and mutation contracts.
"""

from ._runtime import call


def bgmv_shrink(x, weight, indices, y, scale):
    return call("bgmv_shrink", x, weight, indices, y, scale)


def bgmv_expand(x, weight, indices, y, slice_offset, slice_size):
    return call("bgmv_expand", x, weight, indices, y, slice_offset, slice_size)


def sgmv_shrink(x, weight, lora_indices, seq_len, y, scale):
    return call("sgmv_shrink", x, weight, lora_indices, seq_len, y, scale)


def sgmv_expand(x, weight, lora_indices, seq_len, y, slice_offset, slice_size):
    return call("sgmv_expand", x, weight, lora_indices, seq_len, y, slice_offset, slice_size)


def build_combined_lora_idx(dest, topk_ids, lora_indices, adapter_enabled,
                            first_expert_idx, num_experts, top_k):
    return call("build_combined_lora_idx", dest, topk_ids, lora_indices,
                adapter_enabled, first_expert_idx, num_experts, top_k)


def add_lora(y, x, lora_a, lora_b, lora_indices, seq_len, token_lora_indices,
             output_slices, offset_start, scale, add_inputs, use_gmm, no_lora,
             use_add_lora):
    return call("add_lora", y, x, lora_a, lora_b, lora_indices, seq_len,
                token_lora_indices, output_slices, offset_start, scale,
                add_inputs, use_gmm, no_lora, use_add_lora)


def add_lora_shrink(y, x, lora_a, lora_indices, seq_len, token_lora_indices,
                    scale, use_gmm, no_lora, is_moe=False, group_list_type=1):
    return call("add_lora_shrink", y, x, lora_a, lora_indices, seq_len,
                token_lora_indices, scale, use_gmm, no_lora, is_moe, group_list_type)


def add_lora_expand(y, x, lora_b, lora_indices, seq_len, token_lora_indices,
                    output_slices, offset_start, add_inputs, use_gmm, no_lora,
                    is_moe=False, group_list_type=1):
    return call("add_lora_expand", y, x, lora_b, lora_indices, seq_len,
                token_lora_indices, output_slices, offset_start, add_inputs,
                use_gmm, no_lora, is_moe, group_list_type)


def add_lora_expand_delta(gate_up, x, lora_b, seq_len, group_list_type=1):
    return call("add_lora_expand_delta", gate_up, x, lora_b, seq_len, group_list_type)


def add_lora_swiglu_quant(gate_up, delta=None, swiglu_limit=0.0):
    return call("add_lora_swiglu_quant", gate_up, delta, swiglu_limit)
