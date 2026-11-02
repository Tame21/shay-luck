#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""DS LoRA changes from vllm_ascend/lora/fused_moe.py; installed by TurboManager.

Only changed/new functions are carried here. Importing this module does not
replace upstream symbols or execute registration decorators.
"""

from __future__ import annotations

from .ops_backend import get_lora_ops

import torch
from vllm_ascend.lora.fused_moe import (
    reset_lora_indices,
)


def _gather_by_row_permutation(values: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """
    Sorting int32/int64 keys falls back to AiCpu on Ascend,
    so we cast it in fp32. Result is identical.
    """
    return values[torch.argsort(mapping.reshape(-1).to(torch.float32))]


def preprocess_lora_indices(
    lora_context,
    *,
    topk_ids: torch.Tensor,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Expand and permute LoRA token indices for the AlltoAll dispatch path.

    Reads ``lora_context.split_lora_indices``, repeats each entry by
    ``topk``, applies the token permutation to align with the dispatched
    hidden states, and stores the result in
    ``lora_context.permuted_lora_indices``.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``split_lora_indices`` has been populated.
    """
    split_indices = getattr(lora_context, "split_lora_indices", None)
    if split_indices is None:
        return
    expanded = split_indices.repeat_interleave(topk_ids.shape[1])
    lora_context.permuted_lora_indices = _gather_by_row_permutation(expanded, reversed_permutation_mapping)


def postprocess_lora_indices(
    lora_context,
    *,
    reversed_permutation_mapping: torch.Tensor,
) -> None:
    """Re-permute exchanged LoRA indices to align with the global token
    ordering after ``npu_moe_token_permute`` in the AlltoAll dispatch
    postprocess.

    Reads ``lora_context.exchanged_lora_indices``, applies the
    global permutation, and writes the result back.

    Caller must ensure ``lora_context`` is not ``None`` and
    ``exchanged_lora_indices`` has been populated.
    """
    exchanged = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged is None:
        return
    lora_context.exchanged_lora_indices = _gather_by_row_permutation(exchanged, reversed_permutation_mapping)


def _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids):
    """Recover per-permuted-row (expert_id, lora_slot) for the dispatched rows.

    npu_moe_init_routing semantics (verified empirically): ``expanded_row_idx``
    is indexed by the ORIGINAL flat (token, k) position and gives where that
    pair landed in the expert-sorted array -- not the reverse. So recovering
    "which (token, k) pair does sorted row i hold" needs the inverse permutation
    of ``expanded``, not a direct gather by it. ``argsort`` output shape ==
    input shape (value-independent), so this stays graph-capturable -- no
    ``.item()``/data-dependent host sync.

    Under expert parallelism the dispatcher restricts routing to the local
    expert range (``active_expert_range``). Per the npu_moe_init_routing_v2
    contract (row_idx_type=0): active pairs keep their compacted destination
    row in ``[0, available)`` while inactive pairs are filled with -1, and
    the tail rows of the static (num_tokens * top_k)-row tensors are
    undefined. Recovery therefore (a) pushes inactive pairs past the valid
    destination range before ``argsort`` so the first entries of the inverse
    still map dispatched row -> original pair, and (b) converts global expert
    ids to this rank's local ids, disabling LoRA on rows whose pair routed
    to another rank (including the undefined tail) via the -1 slot sentinel.
    """
    top_k = lora_context.top_k
    dest = expanded_row_idx.reshape(-1).to(torch.long)
    num_pairs = dest.numel()
    active = dest >= 0
    # Keys keep active destinations in [0, available) and push inactive
    # pairs beyond num_pairs (distinct keys -> deterministic argsort), so
    # inv maps dispatched rows to their original pairs and every undefined
    # tail row lands on an inactive pair that is masked out below.
    keys = torch.where(active, dest, num_pairs + torch.arange(num_pairs, device=dest.device))
    # again sorting issue with fallback to AiCPU
    inv = torch.argsort(keys.to(torch.float32))
    expert_per_row = topk_ids.reshape(-1)[inv].to(torch.long)

    # token_lora_indices is a 1D LongTensor sized to max_num_batched_tokens
    # (host-known constant). When prepare split the batch across TP ranks
    # (flash-comm off), the dispatched rows come from this rank's shard and
    # split_lora_indices carries the matching (padded) shard slots; with
    # flash-comm on (the SP/EP AllGather deployment) prepare keeps the full
    # batch and the global indices apply. Prefer the split when present.
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    orig_token = inv // top_k
    orig_token = orig_token.clamp_(max=lora_indices.numel() - 1)
    lora_per_row = lora_indices[orig_token]

    if getattr(lora_context, "use_ep", False):
        # Same contiguous-sharding formula as the AllGather dispatcher's
        # active_expert_range: first = ep_rank * num_local_experts.
        from vllm.distributed.parallel_state import get_ep_group

        num_local_experts = lora_context.local_num_experts
        first_expert_idx = get_ep_group().rank_in_group * num_local_experts
        local_expert = expert_per_row - first_expert_idx
        local_row = (local_expert >= 0) & (local_expert < num_local_experts)
        # Rows outside the local range (other ranks' pairs and the undefined
        # tail) must not index the EP-sharded LoRA stacks: clamp the expert
        # id for safety and disable the adapter slot so add_lora_fused_moe's
        # combined gather index becomes -1 (bgmv skips those rows entirely).
        expert_per_row = local_expert.clamp_(0, num_local_experts - 1)
        lora_per_row = torch.where(local_row, lora_per_row, torch.full_like(lora_per_row, -1))
    return expert_per_row, lora_per_row


def _build_combined_lora_idx_allgather(lora_context, expanded_row_idx, topk_ids):
    """Build the per-row combined gather index (AG path).

    aten builds the unique fp32 sort keys and argsorts them; the in-tree
    kernel (get_lora_ops().build_combined_lora_idx,
    csrc/kernels/combined_lora_idx.cpp) then folds the per-pair
    slot/expert/adapter lookup into a single gather pass.
    Bit-identical to _build_combined_lora_idx_allgather_torch, which is kept
    as the reference implementation for the equality tests.
    """
    # token->slot mapping: prefer the TP-split shard when present
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    first_expert_idx = 0
    if getattr(lora_context, "use_ep", False):
        # Same contiguous-sharding formula as the AllGather dispatcher's
        # active_expert_range: first = ep_rank * num_local_experts.
        from vllm.distributed.parallel_state import get_ep_group

        first_expert_idx = get_ep_group().rank_in_group * lora_context.local_num_experts
    num_experts = lora_context.w13_lora_a_stacked[0].shape[1]
    return get_lora_ops().build_combined_lora_idx(
        expanded_row_idx,
        topk_ids,
        lora_indices,
        lora_context.adapter_enabled,
        first_expert_idx,
        num_experts,
        lora_context.top_k,
    )


def _build_combined_lora_idx_allgather_torch(lora_context, expanded_row_idx, topk_ids):
    """Reference torch implementation of the combined gather index (AG path).

    Produces the combined value per (token, k) pair and applies ONE final
    gather by the argsort inverse, instead of recovering two per-row tensors
    and recombining them.

    Row semantics: rows whose pair routed to another rank (or the undefined
    tail) land on inactive pairs and receive the -1 sentinel.
    """
    top_k = lora_context.top_k
    dest = expanded_row_idx.reshape(-1).to(torch.long)
    num_pairs = dest.numel()
    active = dest >= 0
    # Identical key construction to _recover_moe_lora_routing_allgather, so
    # `inv` is bit-identical (see its fp32-sort rationale).
    keys = torch.where(active, dest, num_pairs + torch.arange(num_pairs, device=dest.device))
    inv = torch.argsort(keys.to(torch.float32))

    # Per-pair LoRA slot: pair p belongs to token p // top_k.
    lora_indices = getattr(lora_context, "split_lora_indices", None)
    if lora_indices is None:
        lora_indices = lora_context.punica_wrapper.token_lora_indices
    num_tokens = min(num_pairs // top_k, lora_indices.numel())
    # NOTE: expand+reshape, NOT repeat_interleave -- NPU's repeat_interleave
    # degrades catastrophically at prefill sizes (~14ms at 98k pairs vs
    # ~36us for this copy). Bitwise-equal output.
    slot_pairs = lora_indices[:num_tokens].unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    if slot_pairs.numel() < num_pairs:  # shapes are static per captured graph
        slot_pairs = torch.nn.functional.pad(slot_pairs, (0, num_pairs - slot_pairs.numel()), value=-1)

    # Per-pair local expert id (identity without EP).
    if getattr(lora_context, "use_ep", False):
        from vllm.distributed.parallel_state import get_ep_group

        num_local_experts = lora_context.local_num_experts
        first_expert_idx = get_ep_group().rank_in_group * num_local_experts
        expert_pairs = topk_ids.reshape(-1).to(torch.long) - first_expert_idx
    else:
        expert_pairs = topk_ids.reshape(-1).to(torch.long)

    num_experts = lora_context.w13_lora_a_stacked[0].shape[1]
    slot_safe = slot_pairs.clamp(min=0)
    enabled = active & (slot_pairs >= 0) & lora_context.adapter_enabled[slot_safe].bool()
    combined_pairs = torch.where(enabled, slot_safe * num_experts + expert_pairs, -1)
    return combined_pairs[inv].contiguous()


def _recover_moe_lora_routing_all2all(
    lora_context,
    group_list: torch.Tensor,
):
    """Recover per-row (expert_id, lora_id) for the AlltoAll dispatched tokens.

    In the AlltoAll + EP path, tokens have already been exchanged via
    all_to_all and sorted by local expert.  ``group_list`` tells us how
    many tokens belong to each local expert. The LoRA indices for those
    dispatched rows are stored on ``lora_context.exchanged_lora_indices``.

    Returns:
        expert_per_row: [num_dispatched_tokens] local expert id (0..E-1)
        lora_per_row:   [num_dispatched_tokens] lora adapter id (-1 = none)
    """
    exchanged_lora_indices = getattr(lora_context, "exchanged_lora_indices", None)
    if exchanged_lora_indices is None:
        raise AssertionError("AlltoAll MoE LoRA requires exchanged_lora_indices in lora_context.")

    # Build per-token expert IDs via cumsum+searchsorted: equivalent to the
    # previous repeat_interleave(arange(E), group_list) but the search runs
    # on the vector cores (repeat_interleave with tensor repeats takes a slower
    # path). Output length stays sum(group_list)-dependent as before
    group_ends = torch.cumsum(group_list.to(torch.int64), dim=0)
    num_rows = int(group_ends[-1])
    expert_per_row = torch.searchsorted(group_ends, torch.arange(num_rows, device=group_list.device), right=True)

    lora_per_row = exchanged_lora_indices.reshape(-1).to(torch.long)
    if expert_per_row.numel() != lora_per_row.numel():
        raise AssertionError(
            "AlltoAll MoE LoRA routing metadata is misaligned: "
            f"group_list describes {expert_per_row.numel()} rows, but "
            f"received {lora_per_row.numel()} LoRA indices."
        )

    return expert_per_row, lora_per_row


def moe_lora_apply_w13(
    lora_context,
    *,
    gate_up_out,
    hidden_states,
    lora_routing,
    group_list=None,
    group_list_type=1,
):
    """Apply the w13 LoRA delta, before activation.

    Called from ``unquant_apply_mlp`` right after the base gate_up GMM.

    Args:
        lora_routing: (expert_per_row, lora_per_row) pre-computed by the
            caller via _recover_moe_lora_routing (AllGather) or
            _recover_moe_lora_routing_all2all (AlltoAll).
        group_list: base per-expert row counts, reused by the gmm path.

    Returns:
        The delta when it was not folded into ``gate_up_out``, else None.
    """
    # AG callers (quant_moe / unquant_apply_mlp) stash the single-pass
    # combined index built by _build_combined_lora_idx_allgather directly;
    # A2A callers pass the recovered routing pair and we build here
    combined_idx = getattr(lora_context, "combined_lora_idx", None)
    if combined_idx is not None and combined_idx.numel() != gate_up_out.shape[0]:
        combined_idx = None  # stale stash: fall through to the routing path
    if combined_idx is None:
        if lora_routing is None:
            raise AssertionError("moe_lora_apply_w13 requires either a stashed combined index or a routing pair")
        expert_per_row, lora_per_row = lora_routing
        # EP rank may receive 0 dispatched tokens when all tokens route to
        # experts on other ranks. Skip LoRA to avoid passing empty tensors
        # to add_lora_fused_moe (which can trigger NPU kernel crashes).
        if expert_per_row.numel() == 0:
            return None
        # The w13 and w2 applies of this layer share one routing, hence one
        # combined gather index: build it once here, stash it for w2, and
        # pass it through explicitly.
        from vllm_ascend.lora.punica_npu import build_combined_lora_idx

        combined_idx = build_combined_lora_idx(
            lora_per_row,
            expert_per_row,
            lora_context.adapter_enabled,
            lora_context.w13_lora_a_stacked[0].shape[1],
        )
        lora_context.combined_lora_idx = combined_idx
        return lora_context.punica_wrapper.add_lora_fused_moe(
            y=gate_up_out,
            x=hidden_states,
            lora_a_stacked=lora_context.w13_lora_a_stacked,
            lora_b_stacked=lora_context.w13_lora_b_stacked,
            expert_ids=expert_per_row,
            adapter_enabled=lora_context.adapter_enabled,
            fully_sharded=lora_context.fully_sharded,
            token_lora_mapping=lora_per_row,
            group_list=group_list,
            combined_idx=combined_idx,
        )
    if combined_idx.numel() == 0:
        return None
    return lora_context.punica_wrapper.add_lora_fused_moe(
        y=gate_up_out,
        x=hidden_states,
        lora_a_stacked=lora_context.w13_lora_a_stacked,
        lora_b_stacked=lora_context.w13_lora_b_stacked,
        expert_ids=None,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
    )


def moe_lora_apply_w2(
    lora_context,
    *,
    down_out,
    silu_out,
    lora_routing,
    group_list=None,
    group_list_type=1,
):
    """Add the w2 LoRA delta into ``down_out`` (in place), after the down GMM.

    Reuses the per-row routing computed by ``moe_lora_apply_w13``; ``silu_out``
    is the activation output that fed the base down GMM.
    """
    # AG path: the stash holds the single-pass combined index (built by the
    # caller or by w13); A2A path: recover the routing pair and build here.
    combined_idx = getattr(lora_context, "combined_lora_idx", None)
    if combined_idx is not None and combined_idx.numel() != down_out.shape[0]:
        combined_idx = None  # stale stash: fall through to the routing path
    expert_per_row = token_lora_mapping = None
    if combined_idx is None:
        if lora_routing is None:
            raise AssertionError("moe_lora_apply_w2 requires either a stashed combined index or a routing pair")
        expert_per_row, token_lora_mapping = lora_routing
        # EP rank may receive 0 dispatched tokens; skip LoRA to avoid NPU
        # kernel crashes with empty tensors.
        if expert_per_row.numel() == 0:
            return None
    offset = 0
    if lora_context.fully_sharded:
        shard_size = lora_context.w2_lora_b_stacked[0].shape[-2]
        offset = shard_size * lora_context.tp_rank
    # Fully-sharded W2 needs a shrink all-reduce before its output-sharded
    # B projection. The MoE runner's final output reduction cannot replace it.
    delta = lora_context.punica_wrapper.add_lora_fused_moe(
        y=down_out,
        x=silu_out,
        lora_a_stacked=lora_context.w2_lora_a_stacked,
        lora_b_stacked=lora_context.w2_lora_b_stacked,
        expert_ids=expert_per_row,
        adapter_enabled=lora_context.adapter_enabled,
        fully_sharded=lora_context.fully_sharded,
        offset=offset,
        token_lora_mapping=token_lora_mapping,
        group_list=group_list,
        group_list_type=group_list_type,
        combined_idx=combined_idx,
    )
    if delta is not None:
        down_out += delta
    # Clear per-forward intermediate indices now that the LoRA delta
    # for this layer has been fully applied — they are not needed for
    # the remaining combine/finalize stages.
    reset_lora_indices(lora_context)
