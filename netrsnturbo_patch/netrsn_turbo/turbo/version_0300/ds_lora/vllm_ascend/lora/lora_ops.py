# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""vLLM-facing signatures for the built-in BGMV/SGMV public APIs.

These are bound to Punica instances, including shared-expert dense LoRA.
The caller's rank/hardware fallback policy stays in Punica.__init__.
As in Ascend's original wrappers, native expand always accumulates into y;
the compatibility add_inputs argument does not alter that native contract.
"""

from __future__ import annotations

from .ops_backend import get_lora_ops


def bgmv_shrink(
    inputs,
    lora_a_weights,
    output_tensor,
    lora_indices_tensor,
    scaling: float = 1.0,
):
    return get_lora_ops().bgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, output_tensor, scaling
    )


def bgmv_expand(
    inputs,
    lora_b_weights,
    output_tensor,
    lora_indices_tensor,
    add_inputs: bool = True,
):
    return get_lora_ops().bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, 0, output_tensor.size(1)
    )


def bgmv_expand_slice(
    inputs,
    lora_b_weights,
    output_tensor,
    lora_indices_tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return get_lora_ops().bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def sgmv_shrink(
    inputs,
    lora_a_weights,
    output_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    lora_indices_tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return get_lora_ops().sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs,
    lora_b_weights,
    output_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    lora_indices_tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    return get_lora_ops().sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, 0, output_tensor.size(1)
    )


def sgmv_expand_slice(
    inputs,
    lora_b_weights,
    output_tensor,
    b_seq_start_loc,
    seq_len_tensor,
    lora_indices_tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return get_lora_ops().sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
