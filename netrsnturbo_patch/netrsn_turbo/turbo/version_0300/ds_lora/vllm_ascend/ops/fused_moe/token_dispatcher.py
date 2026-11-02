# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DS LoRA changes from vllm_ascend/ops/fused_moe/token_dispatcher.py; installed by TurboManager.

Only changed/new functions are carried here. Importing this module does not
replace upstream symbols or execute registration decorators.
"""

from __future__ import annotations

import torch
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.lora.quant_moe import validate_quant_moe_lora_activation_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEAllToAllCombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)
from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all


def _can_pack_lora_slots(lora_context, dtype) -> bool:
    """Use rank-independent host configuration to avoid lossy slot casts."""
    wrapper = getattr(lora_context, "punica_wrapper", None)
    config = getattr(wrapper, "lora_config", None)
    capacity = getattr(config, "max_loras", None)
    # The same configured capacity applies on every EP rank, including
    # ranks with no local tokens. Never branch on a device-side slot max.
    exact_integer_limit = {
        torch.bfloat16: 2**8,
        torch.float16: 2**11,
        torch.float32: 2**24,
    }.get(dtype)
    return (
        isinstance(capacity, int)
        and capacity > 0
        and exact_integer_limit is not None
        and capacity - 1 <= exact_integer_limit
    )


def token_dispatch(
    self,
    token_dispatch_input: MoETokenDispatchInput,
):
    use_mxfp_quant = token_dispatch_input.quant.is_mxfp
    with_quant = token_dispatch_input.quant.dispatch_with_quant
    if has_lora(self.lora_context) and token_dispatch_input.quant.is_quant:
        validate_quant_moe_lora_activation_input(
            quant_type=token_dispatch_input.quant.quant_type,
            hidden_states=token_dispatch_input.hidden_states,
            dynamic_scale=token_dispatch_input.routing.pertoken_scale,
        )
        # LoRA A requires the original BF16/FP16 activations. The W8A8
        # LoRA backend performs dynamic quantization immediately before
        # each local expert GMM, after the AlltoAll exchange.
        with_quant = False
    dst_type = token_dispatch_input.quant.get_dst_type
    hidden_states = token_dispatch_input.hidden_states
    topk_weights = token_dispatch_input.topk_weights
    topk_ids = token_dispatch_input.topk_ids

    (
        permutated_local_input_tokens,
        reversed_local_input_permutation_mapping,
        tokens_per_expert,
        input_splits,
        output_splits,
        global_input_tokens_local_experts_indices,
        hidden_shape,
        hidden_shape_before_permute,
    ) = self._dispatch_preprocess(hidden_states, topk_ids)

    dynamic_scale_after_all2all = None
    if with_quant:
        permutated_local_input_tokens, dynamic_scale = DeviceOperator.npu_dynamic_quant(
            permutated_local_input_tokens, act_quant_type=dst_type, use_mxfp_quant=use_mxfp_quant
        )
        _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
            dynamic_scale, output_splits, input_splits, self.ep_group
        )
        permute2_ep_all_to_all_handle.wait()
        dynamic_scale.untyped_storage().resize_(0)

    # Concat the per-row LoRA slot onto the hidden-states all-to-all as
    # one extra bf16 column instead of exchanging them in a separate blocking
    # all_to_all per MoE layer. The column rides through the same
    # splits/collective and is stripped right after the exchange, so the
    # downstream postprocess sees identical inputs.
    lora_extra = None
    if self.lora_context is not None:
        permuted = getattr(self.lora_context, "permuted_lora_indices", None)
        if permuted is not None and _can_pack_lora_slots(self.lora_context, permutated_local_input_tokens.dtype):
            lora_extra = permuted.to(permutated_local_input_tokens.dtype).reshape(-1, 1)
            permutated_local_input_tokens = torch.cat(
                [permutated_local_input_tokens, lora_extra], dim=1
            )

    _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
        permutated_local_input_tokens, output_splits, input_splits, self.ep_group
    )
    permute1_ep_all_to_all_handle.wait()
    permutated_local_input_tokens.untyped_storage().resize_(0)

    if lora_extra is not None:
        # Recover the exchanged LoRA slots in global-row order; values
        # round-trip exactly through bf16 (small ints incl. -1).
        self.lora_context.exchanged_lora_indices = (
            global_input_tokens[:, -1:].to(torch.long).reshape(-1)
        )
        global_input_tokens = global_input_tokens[:, :-1]
    elif self.lora_context is not None:
        # Preserve the original exchange (including its no-op when no
        # permuted indices exist) whenever packing is not proven exact.
        from vllm_ascend.lora.fused_moe import all2all_lora_indices

        all2all_lora_indices(
            self.lora_context,
            output_splits=output_splits,
            input_splits=input_splits,
            ep_group=self.ep_group,
        )

    # Postprocess
    global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
        self._dispatch_postprocess(
            global_input_tokens,
            dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices,
            with_quant,
            dst_type,
        )
    )

    return MoETokenDispatchOutput(
        hidden_states=global_input_tokens,
        dynamic_scale=dynamic_scale_final,
        group_list=tokens_per_expert,
        group_list_type=1,
        combine_metadata=MoEAllToAllCombineMetadata(
            input_splits=input_splits,
            output_splits=output_splits,
            topk_weights=topk_weights,
            reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
            reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
            hidden_shape=hidden_shape,
            hidden_shape_before_permute=hidden_shape_before_permute,
        ),
    )
