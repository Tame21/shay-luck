# SPDX-License-Identifier: Apache-2.0
# Vendored from ops-repo/tests/npu/test_moe_lora_graph.py on 2026-11-02.
# Retains the upstream CPU references and test algorithms; only backend wiring changed.
"""Opt-in NPU graph regression for migrated LoRA kernels.

Capture uses a nondefault stream. Each replay updates device tensors in place,
retains their addresses, and compares with independent CPU fp32 references.
CPU dispatch flags and shapes are static for the lifetime of each graph.
"""

import json
import os

import pytest

if os.environ.get("TURBO_OPS_RUN_NPU_TESTS") != "1":
    pytest.skip("NPU tests require explicit TURBO_OPS_RUN_NPU_TESTS=1", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
from netrsn_turbo import turbo_ops

DEV = os.environ.get("TURBO_OPS_TEST_DEVICE", "npu:0")
REPLAYS = 8


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(DEV)
    turbo_ops.initialize()


def _random(shape, seed, scale=0.1, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * scale).to(dtype)


def _capture(function):
    stream = torch.npu.Stream(device=DEV)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = function()
    return graph, stream, output


def _lora_reference(x, base, a, b, indices, widths, scale, add_inputs):
    result = base.float().clone()
    for row, slot in enumerate(indices.tolist()):
        if slot < 0:
            if not add_inputs:
                result[row, :sum(widths)] = 0
            continue
        offset = 0
        for left, right, width in zip(a, b, widths):
            delta = (x[row].float() @ left[slot, 0].float().T * scale) @ right[slot, 0].float().T
            if add_inputs:
                result[row, offset:offset + width] += delta
            else:
                result[row, offset:offset + width] = delta
            offset += width
    return result


def _quant_reference(gate, delta, limit):
    merged = gate.float() + (delta.float() if delta is not None else 0)
    g, u = merged.chunk(2, dim=1)
    if limit > 0:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    act = u * (g * torch.sigmoid(g))
    scale = act.abs().amax(dim=1) / 127
    payload = (act * (127 / act.abs().amax(dim=1))[:, None]).half().float().round()
    return act, scale, payload


def _routing_reference(dest, topk, mapping, enabled, first_expert, experts, top_k):
    out = torch.full((dest.numel(),), -1, dtype=torch.long)
    for pair, destination in enumerate(dest.flatten().tolist()):
        token = pair // top_k
        if destination < 0 or token >= mapping.numel():
            continue
        slot = int(mapping[token])
        expert = int(topk.flatten()[pair]) - first_expert
        if slot >= 0 and enabled[slot] and 0 <= expert < experts:
            out[destination] = slot * experts + expert
    return out


def _metric(kind, **values):
    print("GRAPH_METRIC " + json.dumps(dict(kind=kind, replays=REPLAYS, **values), sort_keys=True))


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("add_inputs", [True, False])
def test_add_lora_capture_replay_updated_inputs(rank, add_inputs):
    rows, hidden, slots, widths = 16, 512, 4, (128, 80)
    a_cpu = [_random((slots, 1, rank, hidden), 100 + n, 0.05) for n in range(2)]
    b_cpu = [_random((slots, 1, width, rank), 200 + n, 0.05) for n, width in enumerate(widths)]
    a, b = [v.to(DEV) for v in a_cpu], [v.to(DEV) for v in b_cpu]
    x = _random((rows, hidden), 301, 0.3).to(DEV)
    y = _random((rows, sum(widths) + 16), 302, 0.2).to(DEV)
    indices = torch.zeros(rows, dtype=torch.long, device=DEV)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    use_gmm, no_lora, fused = [torch.tensor(v) for v in (False, False, True)]

    def run():
        turbo_ops.add_lora(y, x, a, b, empty, empty, indices, list(widths), 0,
                          0.7313, add_inputs, use_gmm, no_lora, fused)
        return y

    addresses = [value.data_ptr() for value in (x, y, indices, *a, *b)]
    graph, stream, output = _capture(run)
    max_error = 0.0
    try:
        for iteration in range(REPLAYS):
            x_cpu = _random(x.shape, 400 + iteration, 0.3)
            base = _random(y.shape, 500 + iteration, 0.2)
            idx_cpu = torch.tensor([
                -1 if iteration == REPLAYS - 1 or (row + iteration) % 5 == 0
                else (row + iteration) % slots for row in range(rows)])
            expected = _lora_reference(x_cpu, base, a_cpu, b_cpu, idx_cpu, widths,
                                       0.7313, add_inputs)
            with torch.npu.stream(stream):
                x.copy_(x_cpu)
                y.copy_(base)
                indices.copy_(idx_cpu)
                graph.replay()
            stream.synchronize()
            actual = output.cpu()
            error = (actual.float() - expected).abs().max().item()
            max_error = max(max_error, error)
            torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=0)
            assert torch.equal(actual[:, sum(widths):], base[:, sum(widths):])
            assert addresses == [value.data_ptr() for value in (x, y, indices, *a, *b)]
        _metric("add_lora", rank=rank, rows=rows, hidden=hidden, widths=widths,
                add_inputs=add_inputs, max_abs_error=max_error)
    finally:
        graph.reset()


@pytest.mark.parametrize("width", [64, 960, 4096])
@pytest.mark.parametrize("with_delta", [False, True])
def test_swiglu_quant_capture_replay_updated_inputs(width, with_delta):
    rows, limit = 17, 0.75
    gate = _random((rows, width * 2), 601, 0.35).to(DEV)
    delta = _random(gate.shape, 602, 0.35).to(DEV) if with_delta else None
    graph, stream, output = _capture(lambda: turbo_ops.add_lora_swiglu_quant(gate, delta, limit))
    addresses = [value.data_ptr() for value in output]
    maxima = dict(act_max_abs_error=0.0, scale_max_abs_error=0.0,
                  payload_max_abs_difference=0.0, dequant_max_abs_error=0.0)
    try:
        for iteration in range(REPLAYS):
            gate_cpu = _random(gate.shape, 700 + iteration, 0.35)
            delta_cpu = _random(gate.shape, 800 + iteration, 0.35) if with_delta else None
            if iteration == REPLAYS - 1:
                gate_cpu[:3, :width] = 0
                if delta_cpu is not None:
                    delta_cpu[:3, :width] = 0
            ref_act, ref_scale, ref_q = _quant_reference(gate_cpu, delta_cpu, limit)
            with torch.npu.stream(stream):
                gate.copy_(gate_cpu)
                if delta is not None:
                    delta.copy_(delta_cpu)
                graph.replay()
            stream.synchronize()
            act, q, scale = [value.cpu() for value in output]
            act_error = (act.float() - ref_act.to(gate.dtype).float()).abs().max().item()
            scale_error = (scale - ref_scale).abs().max().item()
            valid = ref_scale > 0
            payload_difference = (q.float()[valid] - ref_q[valid]).abs()
            dequant_error = (q.float() * scale[:, None] - ref_act).abs()
            maxima["act_max_abs_error"] = max(maxima["act_max_abs_error"], act_error)
            maxima["scale_max_abs_error"] = max(maxima["scale_max_abs_error"], scale_error)
            maxima["payload_max_abs_difference"] = max(
                maxima["payload_max_abs_difference"], payload_difference.max().item())
            maxima["dequant_max_abs_error"] = max(maxima["dequant_max_abs_error"], dequant_error.max().item())
            assert act_error < 4e-2
            torch.testing.assert_close(scale, ref_scale, rtol=1e-3, atol=1e-7)
            assert payload_difference.max().item() <= 1
            assert (payload_difference > 0).float().mean().item() < 0.02
            assert torch.all(dequant_error <= 0.6 * ref_scale[:, None] + 1e-3)
            assert addresses == [value.data_ptr() for value in output]
        _metric("swiglu_quant", rows=rows, width=width, with_delta=with_delta, **maxima)
    finally:
        graph.reset()


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_combined_metadata_capture_replay_updated_inputs(index_dtype):
    tokens, top_k, experts, slots, first_expert = 13, 6, 8, 3, 16
    shape = (tokens, top_k)
    dest = torch.full(shape, -1, dtype=index_dtype, device=DEV)
    topk = torch.zeros(shape, dtype=index_dtype, device=DEV)
    mapping = torch.zeros(tokens - 2, dtype=torch.long, device=DEV)
    enabled = torch.ones(slots + 1, dtype=torch.int32, device=DEV)
    graph, stream, output = _capture(lambda: turbo_ops.build_combined_lora_idx(
        dest, topk, mapping, enabled, first_expert, experts, top_k))
    address = output.data_ptr()
    try:
        for iteration in range(REPLAYS):
            generator = torch.Generator().manual_seed(900 + iteration)
            active = torch.rand(shape, generator=generator) < 0.5
            if iteration == REPLAYS - 1:
                active.fill_(False)
            dest_cpu = torch.full(shape, -1, dtype=index_dtype)
            dest_cpu[active] = torch.randperm(int(active.sum()), generator=generator).to(index_dtype)
            topk_cpu = torch.randint(0, 32, shape, generator=generator, dtype=index_dtype)
            mapping_cpu = torch.randint(-1, slots, mapping.shape, generator=generator)
            enabled_cpu = torch.ones(slots + 1, dtype=torch.int32)
            enabled_cpu[iteration % slots] = 0
            expected = _routing_reference(dest_cpu, topk_cpu, mapping_cpu, enabled_cpu,
                                          first_expert, experts, top_k)
            with torch.npu.stream(stream):
                for actual, value in zip((dest, topk, mapping, enabled),
                                         (dest_cpu, topk_cpu, mapping_cpu, enabled_cpu)):
                    actual.copy_(value)
                graph.replay()
            stream.synchronize()
            assert torch.equal(output.cpu(), expected)
            assert output.data_ptr() == address
        _metric("combined_metadata", tokens=tokens, top_k=top_k,
                index_dtype=str(index_dtype), max_abs_error=0)
    finally:
        graph.reset()


@pytest.mark.parametrize("rank", [8, 64])
def test_metadata_lora_quant_chain_capture_replay(rank):
    tokens, top_k, slots, experts, hidden, widths = 8, 3, 3, 4, 512, (128, 128)
    rows = tokens * top_k
    dest = torch.full((tokens, top_k), -1, dtype=torch.int32, device=DEV)
    topk = torch.zeros_like(dest)
    mapping = torch.zeros(tokens, dtype=torch.long, device=DEV)
    enabled = torch.ones(slots + 1, dtype=torch.int32, device=DEV)
    a_cpu = [_random((slots * experts, 1, rank, hidden), 1100 + n, 0.05) for n in range(2)]
    b_cpu = [_random((slots * experts, 1, width, rank), 1200 + n, 0.05)
             for n, width in enumerate(widths)]
    a, b = [v.to(DEV) for v in a_cpu], [v.to(DEV) for v in b_cpu]
    x = _random((rows, hidden), 1300, 0.3).to(DEV)
    y = _random((rows, sum(widths)), 1301, 0.2).to(DEV)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    use_gmm, no_lora, fused = [torch.tensor(v) for v in (False, False, True)]

    def run(target=y):
        combined = turbo_ops.build_combined_lora_idx(dest, topk, mapping, enabled, 0, experts, top_k)
        turbo_ops.add_lora(target, x, a, b, empty, empty, combined, list(widths), 0,
                          0.5, True, use_gmm, no_lora, fused)
        act, quant, scale = turbo_ops.add_lora_swiglu_quant(target, None, 0.75)
        return combined, target, act, quant, scale

    graph, stream, outputs = _capture(run)
    max_error = 0.0
    try:
        for iteration in range(REPLAYS):
            generator = torch.Generator().manual_seed(1400 + iteration)
            active = torch.rand(dest.shape, generator=generator) < 0.75
            if iteration == REPLAYS - 1:
                active.fill_(False)
            dest_cpu = torch.full(dest.shape, -1, dtype=torch.int32)
            dest_cpu[active] = torch.randperm(int(active.sum()), generator=generator).int()
            topk_cpu = torch.randint(0, experts, topk.shape, generator=generator, dtype=torch.int32)
            mapping_cpu = torch.randint(-1, slots, mapping.shape, generator=generator)
            enabled_cpu = torch.ones(slots + 1, dtype=torch.int32)
            enabled_cpu[iteration % slots] = 0
            x_cpu = _random(x.shape, 1500 + iteration, 0.3)
            base = _random(y.shape, 1600 + iteration, 0.2)
            combined_ref = _routing_reference(dest_cpu, topk_cpu, mapping_cpu, enabled_cpu,
                                              0, experts, top_k)
            y_ref = _lora_reference(x_cpu, base, a_cpu, b_cpu, combined_ref, widths, 0.5, True)
            with torch.npu.stream(stream):
                for target, value in zip((dest, topk, mapping, enabled, x, y),
                                          (dest_cpu, topk_cpu, mapping_cpu, enabled_cpu, x_cpu, base)):
                    target.copy_(value)
                graph.replay()
            stream.synchronize()
            captured = [value.cpu() for value in outputs]
            assert torch.equal(captured[0], combined_ref)
            error = (captured[1].float() - y_ref).abs().max().item()
            max_error = max(max_error, error)
            torch.testing.assert_close(captured[1].float(), y_ref, atol=3e-2, rtol=0)
            # Compare the entire graph chain with a fresh ordinary launch on
            # the same nondefault stream, including integer quantized output.
            with torch.npu.stream(stream):
                eager = run(base.to(DEV))
            stream.synchronize()
            for replay_value, eager_value in zip(captured, eager):
                assert torch.equal(replay_value, eager_value.cpu()), "graph/eager chain differs"
        _metric("metadata_lora_quant_chain", rank=rank, rows=rows, hidden=hidden,
                widths=widths, max_abs_error=max_error, graph_eager_bitwise_equal=True)
    finally:
        graph.reset()
