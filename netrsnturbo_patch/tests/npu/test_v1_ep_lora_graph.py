# SPDX-License-Identifier: Apache-2.0
# Vendored from ops-repo/tests/npu/test_v1_ep_lora_graph.py on 2026-11-02.
# Retains the upstream CPU references and test algorithms; only backend wiring changed.
"""Real DSv4 v1 EP shapes on one NPU, without model or HCCL dependencies.

The local expert shard has 32 experts. Decode covers eight one-token requests
and eight requests with five speculative tokens. Prefill explicitly exercises
stacked MoE GMM and the standalone delta consumed by fused SwiGLU quantization.
Routing is regenerated and expert-sorted on every replay, including empty
expert groups, disabled adapters, and a final batch without LoRA.
"""

import os

import pytest

if os.environ.get("TURBO_OPS_RUN_NPU_TESTS") != "1":
    pytest.skip("NPU tests require explicit TURBO_OPS_RUN_NPU_TESTS=1", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
from netrsn_turbo import turbo_ops

from test_moe_lora_graph import (
    DEV, REPLAYS, _capture, _lora_reference, _metric, _quant_reference,
    _random, _routing_reference,
)

EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, RANK, SLOTS = 32, 6, 4096, 2048, 16, 3
FIRST_EXPERT = 64
WIDTHS = (INTERMEDIATE, INTERMEDIATE)
LORA_SCALE = 0.5


@pytest.fixture(scope="module")
def _weights():
    torch.npu.set_device(DEV)
    turbo_ops.initialize()
    shapes = (
        [(SLOTS, EXPERTS, RANK, HIDDEN)] * 2,
        [(SLOTS, EXPERTS, INTERMEDIATE, RANK)] * 2,
        [(SLOTS, EXPERTS, RANK, INTERMEDIATE)],
        [(SLOTS, EXPERTS, HIDDEN, RANK)],
    )
    cpu = [[_random(shape, 3000 + 10 * group + part, 0.025)
            for part, shape in enumerate(group_shapes)]
           for group, group_shapes in enumerate(shapes)]
    device = [[value.to(DEV) for value in group] for group in cpu]
    flat_cpu = [[value.flatten(0, 1).unsqueeze(1) for value in group] for group in cpu]
    flat_device = [[value.flatten(0, 1).unsqueeze(1) for value in group] for group in device]
    return device, flat_device, flat_cpu


def _route(tokens, iteration):
    generator = torch.Generator().manual_seed(3100 + iteration)
    # Every token chooses six distinct experts from this EP shard. This is
    # the maximum local routed-row count; some expert groups remain empty.
    expert_ids = torch.stack([
        (torch.randperm(EXPERTS - 4, generator=generator)[:TOP_K] + iteration) % EXPERTS
        for _ in range(tokens)
    ])
    order = torch.argsort(expert_ids.flatten(), stable=True)
    dest = torch.empty(tokens * TOP_K, dtype=torch.int32)
    dest[order] = torch.arange(tokens * TOP_K, dtype=torch.int32)
    mapping = (torch.arange(tokens) + iteration) % SLOTS
    mapping[(torch.arange(tokens) + iteration) % 5 == 0] = -1
    if iteration == REPLAYS - 1:
        mapping.fill_(-1)
    enabled = torch.ones(SLOTS + 1, dtype=torch.int32)
    enabled[iteration % SLOTS] = 0
    counts = torch.bincount(expert_ids.flatten(), minlength=EXPERTS)
    token_x = _random((tokens, HIDDEN), 3200 + iteration, 0.3)
    routed_x = token_x.repeat_interleave(TOP_K, dim=0)[order]
    return dest.view(tokens, TOP_K), (expert_ids + FIRST_EXPERT).int(), mapping, enabled, counts, routed_x


@pytest.mark.parametrize("path,tokens", [
    ("decode", 8), ("decode", 48), ("prefill_gmm", 64),
])
def test_v1_ep_real_shapes_updated_graph(path, tokens, _weights):
    device_weights, flat_weights, flat_cpu = _weights
    a13, b13, a2, b2 = device_weights
    fa13, fb13, fa2, fb2 = flat_weights
    ca13, cb13, ca2, cb2 = flat_cpu
    rows = tokens * TOP_K
    inputs = [value.to(DEV) for value in _route(tokens, 0)]
    dest, topk, mapping, enabled, counts, x = inputs
    gate = _random((rows, 2 * INTERMEDIATE), 3300, 0.2).to(DEV)
    down = _random((rows, HIDDEN), 3301, 0.2).to(DEV)
    empty = torch.empty(0, dtype=torch.long, device=DEV)
    decode, gmm, no_lora, fused = [torch.tensor(value) for value in (False, True, False, True)]

    def scratch(slices):
        return [torch.empty(rows, SLOTS * RANK, dtype=torch.float32, device=DEV)
                for _ in range(slices)] + [torch.empty(slices, rows, RANK, dtype=torch.float32, device=DEV)]

    shrink13, shrink2 = scratch(2), scratch(1)

    def run(gate_target=gate, down_target=down):
        combined = turbo_ops.build_combined_lora_idx(
            dest, topk, mapping, enabled, FIRST_EXPERT, EXPERTS, TOP_K)
        delta = None
        if path == "decode":
            turbo_ops.add_lora(gate_target, x, fa13, fb13, empty, empty, combined,
                              list(WIDTHS), 0, LORA_SCALE, True, decode, no_lora, fused)
        else:
            turbo_ops.add_lora_shrink(shrink13, x, a13, empty, counts, combined,
                                     LORA_SCALE, gmm, no_lora, True, 1)
            delta = turbo_ops.add_lora_expand_delta(gate_target, shrink13, b13, counts, 1)
        act, quant, scale = turbo_ops.add_lora_swiglu_quant(gate_target, delta, 0.75)
        if path == "decode":
            turbo_ops.add_lora(down_target, act, fa2, fb2, empty, empty, combined,
                              [HIDDEN], 0, LORA_SCALE, True, decode, no_lora, fused)
        else:
            turbo_ops.add_lora_shrink(shrink2, act, a2, empty, counts, combined,
                                     LORA_SCALE, gmm, no_lora, True, 1)
            turbo_ops.add_lora_expand(down_target, shrink2, b2, empty, counts, combined,
                                     [HIDDEN], 0, True, gmm, no_lora, True, 1)
        return combined, gate_target, delta, act, quant, scale, down_target

    graph, stream, outputs = _capture(run)
    pointers = [value.data_ptr() for value in (*inputs, gate, down)]
    maxima = dict(gate_max_abs_error=0.0, down_max_abs_error=0.0,
                  act_max_abs_error=0.0, scale_max_abs_error=0.0,
                  payload_max_abs_difference=0.0, dequant_max_abs_error=0.0)
    try:
        for iteration in range(REPLAYS):
            values = _route(tokens, iteration)
            dest_cpu, topk_cpu, mapping_cpu, enabled_cpu, _, x_cpu = values
            base_gate = _random(gate.shape, 3400 + iteration, 0.2)
            base_down = _random(down.shape, 3500 + iteration, 0.2)
            idx_ref = _routing_reference(dest_cpu, topk_cpu, mapping_cpu, enabled_cpu,
                                         FIRST_EXPERT, EXPERTS, TOP_K)
            gate_ref = _lora_reference(x_cpu, base_gate, ca13, cb13, idx_ref,
                                       WIDTHS, LORA_SCALE, True)
            with torch.npu.stream(stream):
                for target, value in zip(inputs, values):
                    target.copy_(value)
                gate.copy_(base_gate)
                down.copy_(base_down)
                graph.replay()
            stream.synchronize()
            captured = [value.cpu() if value is not None else None for value in outputs]
            idx, actual_gate, delta, act, quant, scale, actual_down = captured
            assert torch.equal(idx, idx_ref)
            merged_gate = actual_gate.float() + (delta.float() if delta is not None else 0)
            gate_error = (merged_gate - gate_ref).abs().max().item()
            maxima["gate_max_abs_error"] = max(maxima["gate_max_abs_error"], gate_error)
            torch.testing.assert_close(merged_gate, gate_ref, atol=3e-2, rtol=0)
            if delta is not None:
                assert torch.equal(actual_gate, base_gate), "standalone delta must not mutate gate"

            act_ref, scale_ref, quant_ref = _quant_reference(actual_gate, delta, 0.75)
            act_error = (act.float() - act_ref.to(act.dtype).float()).abs().max().item()
            scale_error = (scale - scale_ref).abs().max().item()
            payload_difference = (quant.float() - quant_ref).abs()
            dequant_error = (quant.float() * scale[:, None] - act_ref).abs()
            maxima["act_max_abs_error"] = max(maxima["act_max_abs_error"], act_error)
            maxima["scale_max_abs_error"] = max(maxima["scale_max_abs_error"], scale_error)
            maxima["payload_max_abs_difference"] = max(
                maxima["payload_max_abs_difference"], payload_difference.max().item())
            maxima["dequant_max_abs_error"] = max(maxima["dequant_max_abs_error"], dequant_error.max().item())
            assert act_error < 4e-2
            torch.testing.assert_close(scale, scale_ref, rtol=1e-3, atol=1e-7)
            assert payload_difference.max().item() <= 1
            assert (payload_difference > 0).float().mean().item() < 0.02
            assert torch.all(dequant_error <= 0.6 * scale_ref[:, None] + 1e-3)
            down_ref = _lora_reference(act, base_down, ca2, cb2, idx_ref,
                                       (HIDDEN,), LORA_SCALE, True)
            down_error = (actual_down.float() - down_ref).abs().max().item()
            maxima["down_max_abs_error"] = max(maxima["down_max_abs_error"], down_error)
            torch.testing.assert_close(actual_down.float(), down_ref, atol=3e-2, rtol=0)
            inactive = idx_ref < 0
            assert torch.equal(actual_down[inactive], base_down[inactive])
            assert torch.equal(merged_gate[inactive], base_gate[inactive].float())
            assert pointers == [value.data_ptr() for value in (*inputs, gate, down)]

            with torch.npu.stream(stream):
                eager = run(base_gate.to(DEV), base_down.to(DEV))
            stream.synchronize()
            for replay_value, eager_value in zip(captured, eager):
                if replay_value is not None:
                    assert torch.equal(replay_value, eager_value.cpu()), "EP graph/eager chain differs"
        _metric("v1_ep_real_shape", path=path, tokens=tokens, rows=rows,
                experts=EXPERTS, top_k=TOP_K, hidden=HIDDEN, intermediate=INTERMEDIATE,
                rank=RANK, slots=SLOTS, graph_eager_bitwise_equal=True, **maxima)
    finally:
        graph.reset()
