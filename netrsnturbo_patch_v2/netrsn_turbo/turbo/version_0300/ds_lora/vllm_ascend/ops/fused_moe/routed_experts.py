#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""DS LoRA changes from vllm_ascend/ops/fused_moe/routed_experts.py; installed by TurboManager.

Only changed/new functions are carried here. Importing this module does not
replace upstream symbols or execute registration decorators.
"""

from __future__ import annotations

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput


def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
    layer = mlp_compute_input.layer
    assert layer is not None
    w1, _ = self.get_mlp_weights(layer)
    w1 = self._maybe_transpose(w1, mlp_compute_input.need_trans)
    hidden_states = mlp_compute_input.hidden_states
    w1_bias = layer.w13_bias if self.moe.has_bias else None
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=w1 if isinstance(w1, list) else [w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=mlp_compute_input.group_list_type,
        group_type=0,
        group_list=mlp_compute_input.group_list,
    )[0]

    # MoE LoRA: only attempt injection when an adapter wraps this layer and
    # the comm method provided routing metadata in lora_context.
    self._lora_routing = None
    lora_context = mlp_compute_input.lora_context
    if lora_context is not None:  # LoRA applied
        from vllm_ascend.lora.fused_moe import (
            _build_combined_lora_idx_allgather,
            _recover_moe_lora_routing_all2all,
            moe_lora_apply_w13,
        )

        if mlp_compute_input.expanded_row_idx is not None and mlp_compute_input.topk_ids is not None:
            # AllGather path: single-pass combined index (replaces the
            # recovery + build chain), stashed for the w13/w2 applies.
            lora_context.combined_lora_idx = _build_combined_lora_idx_allgather(
                lora_context,
                mlp_compute_input.expanded_row_idx,
                mlp_compute_input.topk_ids,
            )
            self._lora_routing = None
        elif getattr(lora_context, "exchanged_lora_indices", None) is not None:
            # AlltoAll path: tokens already sorted by expert after exchange.
            self._lora_routing = _recover_moe_lora_routing_all2all(
                lora_context,
                group_list=mlp_compute_input.group_list,
            )
        else:
            raise AssertionError(
                "MoE LoRA requires either expanded_row_idx+topk_ids "
                "(AllGather) or lora_context.exchanged_lora_indices "
                "(AlltoAll). Neither was provided."
            )

        delta = moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            lora_routing=self._lora_routing,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
        )
        if delta is not None:
            gate_up_out += delta
    return gate_up_out


def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
    layer = mlp_compute_input.layer
    assert layer is not None
    _, w2 = self.get_mlp_weights(layer)
    w2 = self._maybe_transpose(w2, mlp_compute_input.need_trans)
    w2_bias = layer.w2_bias if self.moe.has_bias else None
    silu_out = hidden_states
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=w2 if isinstance(w2, list) else [w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=mlp_compute_input.group_list_type,
        group_type=0,
        group_list=mlp_compute_input.group_list,
    )[0]

    # LoRA w2 delta: applied to the down-proj output, with the activation
    # output as the lora_a input. Gate on lora_context (not _lora_routing):
    # on the AllGather path _lora_routing is None by design and
    # moe_lora_apply_w2 consumes the combined_lora_idx stash instead --
    # gating on _lora_routing here silently skipped the w2 (down_proj)
    # delta for every unquantized TP/AllGather deployment.
    if self.lora_context is not None:
        from vllm_ascend.lora.fused_moe import moe_lora_apply_w2

        moe_lora_apply_w2(
            self.lora_context,
            down_out=hidden_states,
            silu_out=silu_out,
            lora_routing=self._lora_routing,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
        )
        self._lora_routing = None
    return hidden_states
