# SPDX-License-Identifier: Apache-2.0
# Vendored from ops-repo/tests/npu/test_moe_lora.py on 2026-11-02.
# Retains the upstream CPU references and test algorithms; only backend wiring changed.
"""Opt-in direct-kernel regression tests; no vLLM imports.

Adapted from vllm-ascend misha tests/ut/lora/test_add_lora_fused_kernel.py,
test_add_lora_fused_moe.py, test_bgmv_small_rank.py and
test_combined_idx_dedup.py. CPU fp32 references are independent of the
migrated kernels. Tests do not select or probe any NPU during collection.

Run on an available device after compiling/installing the library:
  TURBO_OPS_RUN_NPU_TESTS=1 TURBO_OPS_TEST_DEVICE=npu:0 python -m pytest tests/npu -q
"""

import os

import pytest

if os.environ.get("TURBO_OPS_RUN_NPU_TESTS") != "1":
    pytest.skip("NPU tests require explicit TURBO_OPS_RUN_NPU_TESTS=1", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
from netrsn_turbo import turbo_ops  # noqa: F401

DEV = os.environ.get("TURBO_OPS_TEST_DEVICE", "npu:0")


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(DEV)
    turbo_ops.initialize()


def _flag(value):
    # Control flags are intentionally CPU tensors: C++ branches on their data.
    return torch.tensor(value, dtype=torch.bool)


def _case(rank, rows=8, widths=(128, 80), hidden=512, experts=1, dtype=torch.bfloat16, padding=16):
    generator = torch.Generator().manual_seed(rank + rows + hidden)
    slots = 3

    def rand(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    x = rand((rows, hidden), 0.3)
    a = [rand((slots, experts, rank, hidden), 0.05) for _ in widths]
    b = [rand((slots, experts, width, rank), 0.05) for width in widths]
    y = rand((rows, sum(widths) + padding), 0.2)
    mapping = torch.tensor([-1 if t % 5 == 4 else (t // 2) % slots for t in range(rows)])
    eids = torch.tensor([t % experts for t in range(rows)])
    combined = torch.where(mapping >= 0, mapping * experts + eids, -1)
    # MoE converts adapter/expert dimensions to ordinary linear adapter slots.
    a_flat = [value.reshape(slots * experts, 1, rank, hidden).to(DEV) for value in a]
    b_flat = [value.reshape(slots * experts, 1, width, rank).to(DEV) for value, width in zip(b, widths)]
    return x.to(DEV), y.to(DEV), a_flat, b_flat, combined.to(DEV), widths


def _reference(x, y, a, b, mapping, widths, scale=1.0, add_inputs=True, offset=0):
    xc, out = x.cpu().float(), y.cpu().float().clone()
    ac, bc = [t.cpu().float() for t in a], [t.cpu().float() for t in b]
    for row, slot in enumerate(mapping.cpu().tolist()):
        if slot < 0:
            if not add_inputs:
                out[row, offset : offset + sum(widths)] = 0
            continue
        col = offset
        for left, right, width in zip(ac, bc, widths):
            delta = (xc[row] @ left[slot, 0].T * scale) @ right[slot, 0].T
            if add_inputs:
                out[row, col : col + width] += delta
            else:
                out[row, col : col + width] = delta
            col += width
    return out


def _apply(x, y, a, b, mapping, widths, scale=1.0, add_inputs=True, fused=True, no_lora=False, offset=0):
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    result = torch.ops.turbo_ops.add_lora(
        y,
        x,
        a,
        b,
        empty,
        empty,
        mapping,
        list(widths),
        offset,
        scale,
        add_inputs,
        _flag(False),
        _flag(no_lora),
        _flag(fused),
    )
    assert result is None
    return y


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("experts", [1, 4])
@pytest.mark.parametrize("add_inputs", [True, False])
def test_fused_dense_and_moe(rank, experts, add_inputs):
    x, y0, a, b, idx, widths = _case(rank, experts=experts)
    before = [t.cpu().clone() for t in (x, idx, *a, *b)]
    expected = _reference(x, y0, a, b, idx, widths, 0.7313, add_inputs)
    y = _apply(x, y0.clone(), a, b, idx, widths, 0.7313, add_inputs)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)
    assert torch.equal(y[:, sum(widths) :].cpu(), y0[:, sum(widths) :].cpu())
    for tensor, original in zip((x, idx, *a, *b), before):
        assert torch.equal(tensor.cpu(), original), "only y may be mutated"
    inactive = idx.cpu() < 0
    if add_inputs:
        assert torch.equal(y.cpu()[inactive], y0.cpu()[inactive])
    else:
        assert torch.count_nonzero(y.cpu()[inactive, : sum(widths)]) == 0


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("fused", [True, False])
def test_decode_fused_and_bgmv_fallback(rank, fused):
    x, y0, a, b, idx, widths = _case(rank, rows=17, experts=4)
    expected = _reference(x, y0, a, b, idx, widths)
    y = _apply(x, y0.clone(), a, b, idx, widths, fused=fused)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)


@pytest.mark.parametrize(
    "rank,hidden,widths",
    [
        (8, 2048, (1040, 1552)),
        (16, 4096, (512, 528)),
        (32, 4112, (256, 272, 752, 32)),
        (64, 7168, (128, 144)),
    ],
)
def test_rank_and_tile_boundaries(rank, hidden, widths):
    x, y0, a, b, idx, widths = _case(rank, hidden=hidden, widths=widths)
    expected = _reference(x, y0, a, b, idx, widths)
    first = _apply(x, y0.clone(), a, b, idx, widths)
    second = _apply(x, y0.clone(), a, b, idx, widths)
    torch.testing.assert_close(first.cpu().float(), expected, atol=3e-2, rtol=0)
    assert torch.equal(first.cpu(), second.cpu()), "repeated launches must be deterministic"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("no_lora", [True, False])
def test_no_adapter_or_host_no_lora_leaves_output(dtype, no_lora):
    x, y0, a, b, idx, widths = _case(8, dtype=dtype)
    if not no_lora:
        idx.fill_(-1)
    y = _apply(x, y0.clone(), a, b, idx, widths, no_lora=no_lora)
    assert torch.equal(y.cpu(), y0.cpu())


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_split_shrink_expand_matches_reference(rank):
    x, y0, a, b, idx, widths = _case(rank, experts=4)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    scratch = [torch.zeros(x.shape[0], rank, device=DEV, dtype=torch.float32) for _ in a]
    assert (
        torch.ops.turbo_ops.add_lora_shrink(scratch, x, a, empty, empty, idx, 0.25, _flag(False), _flag(False)) is None
    )
    y = y0.clone()
    assert (
        torch.ops.turbo_ops.add_lora_expand(
            y, scratch, b, empty, empty, idx, list(widths), 0, True, _flag(False), _flag(False)
        )
        is None
    )
    expected = _reference(x, y0, a, b, idx, widths, scale=0.25)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)


@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32, 64])
def test_bgmv_shrink_small_rank(rank):
    x, _, a, _, idx, _ = _case(rank, rows=17, experts=4)
    weight = a[0].squeeze(1)
    out = torch.zeros(x.shape[0], rank, device=DEV, dtype=torch.float32)
    before = [t.cpu().clone() for t in (x, weight, idx)]
    assert torch.ops.turbo_ops.bgmv_shrink(x, weight, idx, out, 0.5) is None
    ref = torch.zeros(out.shape, dtype=torch.float32)
    for row, slot in enumerate(idx.cpu().tolist()):
        if slot >= 0:
            ref[row] = before[0][row].float() @ before[1][slot].float().T * 0.5
    torch.testing.assert_close(out.cpu(), ref, atol=3e-2, rtol=0)
    assert ref.abs().max() > 1e-3
    for tensor, original in zip((x, weight, idx), before):
        assert torch.equal(tensor.cpu(), original)


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_bgmv_expand_alias_and_padding(rank):
    x, y0, a, b, idx, widths = _case(rank, widths=(128,))
    z = torch.zeros(x.shape[0], rank, device=DEV, dtype=torch.float32)
    torch.ops.turbo_ops.bgmv_shrink(x, a[0], idx, z, 1.0)
    y = y0.clone()
    result = torch.ops.turbo_ops.bgmv_expand(z, b[0], idx, y, 0, widths[0])
    assert result.data_ptr() == y.data_ptr(), "expand returns its mutated y"
    expected = _reference(x, y0, a, b, idx, widths)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)
    assert torch.equal(y[:, widths[0] :].cpu(), y0[:, widths[0] :].cpu())


@pytest.mark.parametrize(
    "tokens,pad_lora,disable_slot,all_inactive",
    [
        (8, False, -1, False),
        (48, False, 1, False),
        (5, True, -1, False),
        (12, False, -1, True),
        (2048, False, -1, False),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_combined_idx_matches_reference(tokens, pad_lora, disable_slot, all_inactive, index_dtype):
    generator = torch.Generator().manual_seed(tokens * 7)
    top_k, num_experts, slots, first_expert = 6, 8, 3, 16
    topk = torch.randint(0, 32, (tokens, top_k), generator=generator, dtype=index_dtype)
    active = (
        torch.zeros(tokens, top_k, dtype=torch.bool)
        if all_inactive
        else (torch.rand(tokens, top_k, generator=generator) < 0.5)
    )
    dest = torch.full((tokens, top_k), -1, dtype=index_dtype)
    active_count = int(active.sum())
    # Exercise a nontrivial permutation of active rows, not already sorted input.
    dest[active] = torch.randperm(active_count, generator=generator).to(index_dtype)
    lora_len = tokens - 3 if pad_lora else tokens + 8
    mapping = torch.randint(-1, slots, (lora_len,), generator=generator)
    enabled = torch.ones(slots + 1, dtype=torch.int32)
    if disable_slot >= 0:
        enabled[disable_slot] = 0
    expected = torch.full((tokens * top_k,), -1, dtype=torch.long)
    for pair, destination in enumerate(dest.flatten().tolist()):
        token = pair // top_k
        if destination < 0 or token >= lora_len:
            continue
        slot = int(mapping[token])
        expert = int(topk.flatten()[pair]) - first_expert
        if slot >= 0 and enabled[slot] and 0 <= expert < num_experts:
            expected[destination] = slot * num_experts + expert
    inputs = [value.to(DEV) for value in (dest, topk, mapping, enabled)]
    result = torch.ops.turbo_ops.build_combined_lora_idx(*inputs, first_expert, num_experts, top_k)
    assert result.shape == (tokens * top_k,) and result.dtype == torch.long
    assert torch.equal(result.cpu(), expected)
    for actual, original in zip(inputs, (dest, topk, mapping, enabled)):
        assert torch.equal(actual.cpu(), original)


def test_combined_idx_empty():
    empty = torch.empty(0, 6, device=DEV, dtype=torch.int32)
    result = torch.ops.turbo_ops.build_combined_lora_idx(
        empty,
        empty,
        torch.empty(0, device=DEV, dtype=torch.long),
        torch.ones(4, device=DEV, dtype=torch.int32),
        0,
        8,
        6,
    )
    assert result.shape == (0,) and result.dtype == torch.long


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_sgmv_sequence_metadata(rank):
    x, y0, a, b, _, widths = _case(rank, rows=9, widths=(128,))
    seq = torch.tensor([2, 3, 4], dtype=torch.long, device=DEV)
    slots = torch.tensor([0, -1, 2], dtype=torch.long, device=DEV)
    token_slots = torch.tensor([0, 0, -1, -1, -1, 2, 2, 2, 2], device=DEV)
    z = torch.zeros(x.shape[0], rank, dtype=torch.float32, device=DEV)
    assert torch.ops.turbo_ops.sgmv_shrink(x, a[0], slots, seq, z, 0.25) is None
    y = y0.clone()
    result = torch.ops.turbo_ops.sgmv_expand(z, b[0], slots, seq, y, 0, widths[0])
    assert result.data_ptr() == y.data_ptr()
    expected = _reference(x, y0, a, b, token_slots, widths, scale=0.25)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)
    assert torch.equal(slots.cpu(), torch.tensor([0, -1, 2]))
    assert torch.equal(seq.cpu(), torch.tensor([2, 3, 4]))


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sgmv_shrink_small_rank_copyout(rank, dtype):
    # Each FP32 output row occupies less than one 32-byte DMA block. The
    # DS CopyOut uses byte-count DataCopyPad, preserving the adjacent rows.
    x, _, a, _, _, _ = _case(rank, rows=9, widths=(128,), dtype=dtype)
    seq = torch.tensor([2, 3, 4], dtype=torch.long, device=DEV)
    slots = torch.tensor([0, -1, 2], dtype=torch.long, device=DEV)
    storage = torch.zeros(x.shape[0] + 1, rank, dtype=torch.float32, device=DEV)
    storage[-1].fill_(123.0)
    out = storage[:-1]
    turbo_ops.sgmv_shrink(x, a[0], slots, seq, out, 0.25)
    xc, ac = x.cpu().float(), a[0].cpu().float()
    expected = torch.zeros(x.shape[0], rank)
    for row, slot in enumerate((0, 0, -1, -1, -1, 2, 2, 2, 2)):
        if slot >= 0:
            expected[row] = (xc[row] @ ac[slot, 0].T) * 0.25
    torch.testing.assert_close(out.cpu(), expected, atol=3e-2, rtol=0)
    assert torch.equal(storage[-1].cpu(), torch.full((rank,), 123.0))
    assert torch.count_nonzero(out[2:5]).item() == 0


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_moe_grouped_matmul_and_separate_delta(rank):
    # GMM uses expert-sorted rows and per-expert counts; adapter slots are
    # masked inside the stacked shrink result before the second projection.
    rows, slots, experts = 16, 3, 4
    x, y0, flat_a, flat_b, idx, widths = _case(rank, rows=rows, experts=experts, padding=0)
    order = torch.argsort(torch.arange(rows) % experts).to(DEV)
    x, y0, idx = x[order], y0[order], idx[order]
    a = [value.reshape(slots, experts, rank, x.shape[1]) for value in flat_a]
    b = [value.reshape(slots, experts, width, rank) for value, width in zip(flat_b, widths)]
    group_counts = torch.full((experts,), rows // experts, dtype=torch.long, device=DEV)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    scratch = [torch.zeros(rows, slots * rank, device=DEV, dtype=torch.float32) for _ in a]
    scratch.append(torch.zeros(len(a), rows, rank, device=DEV, dtype=torch.float32))
    assert (
        torch.ops.turbo_ops.add_lora_shrink(
            scratch, x, a, empty, group_counts, idx, 0.5, _flag(True), _flag(False), True, 1
        )
        is None
    )
    y = y0.clone()
    assert (
        torch.ops.turbo_ops.add_lora_expand(
            y, scratch, b, empty, group_counts, idx, list(widths), 0, True, _flag(True), _flag(False), True, 1
        )
        is None
    )
    expected = _reference(x, y0, flat_a, flat_b, idx, widths, scale=0.5)
    torch.testing.assert_close(y.cpu().float(), expected, atol=3e-2, rtol=0)
    original = y0.cpu().clone()
    delta = torch.ops.turbo_ops.add_lora_expand_delta(y0, scratch, b, group_counts, 1)
    assert delta.data_ptr() != y0.data_ptr()
    assert torch.equal(y0.cpu(), original)
    torch.testing.assert_close((y0 + delta).cpu().float(), expected, atol=3e-2, rtol=0)
