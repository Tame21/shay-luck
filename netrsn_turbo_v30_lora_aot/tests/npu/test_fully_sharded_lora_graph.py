# SPDX-License-Identifier: Apache-2.0
# Vendored from ops-repo/tests/npu/test_fully_sharded_lora_graph.py on 2026-11-02.
# Retains the upstream CPU references and test algorithms; only backend wiring changed.
"""TP8 LoRA buffer contract on one NPU: local rank 2 -> gathered rank 16.

The eight local projections run on the selected device. torch.cat models the
all_gather layout, including [T,L,R_local] for GMM. This verifies migrated
operators and graph dependencies, not distributed HCCL communication.
"""

import os

import pytest

if os.environ.get("TURBO_OPS_RUN_NPU_TESTS") != "1":
    pytest.skip("NPU tests require explicit TURBO_OPS_RUN_NPU_TESTS=1", allow_module_level=True)

import torch
import torch_npu  # noqa: F401
from netrsn_turbo import turbo_ops

from .test_moe_lora_graph import DEV, REPLAYS, _capture, _metric, _random


@pytest.fixture(scope="module", autouse=True)
def _select_device():
    torch.npu.set_device(DEV)
    turbo_ops.initialize()


@pytest.mark.parametrize("use_gmm", [False, True], ids=["decode_bgmv", "prefill_gmm"])
def test_local_rank2_gathered_rank16_expand_eager_and_graph(use_gmm):
    rows, hidden, slots, experts = 16, 512, 2, 4
    local_rank, full_rank, tp_size, widths = 2, 16, 8, (256, 256)
    a_cpu = [_random((slots, experts, full_rank, hidden), 1700 + s, 0.05) for s in range(2)]
    b_cpu = [_random((slots, experts, width, full_rank), 1800 + s, 0.05) for s, width in enumerate(widths)]
    a_shards = [
        [value[:, :, shard * local_rank : (shard + 1) * local_rank].contiguous().to(DEV) for value in a_cpu]
        for shard in range(tp_size)
    ]
    b = [value.to(DEV) for value in b_cpu]
    scratch = [
        [torch.empty(rows, slots * local_rank, device=DEV, dtype=torch.float32) for _ in range(2)]
        + [torch.empty(2, rows, local_rank, device=DEV, dtype=torch.float32)]
        for _ in range(tp_size)
    ]
    unused_gathered = torch.empty(2, rows, full_rank, device=DEV, dtype=torch.float32)
    x = _random((rows, hidden), 1900, 0.3).to(DEV)
    y = _random((rows, sum(widths)), 1901, 0.2).to(DEV)
    combined = torch.zeros(rows, dtype=torch.long, device=DEV)
    counts = torch.full((experts,), rows // experts, dtype=torch.long, device=DEV)
    flag, no_lora = torch.tensor(use_gmm), torch.tensor(False)

    def run(target=y):
        for local, weights in zip(scratch, a_shards):
            turbo_ops.add_lora_shrink(local, x, weights, combined, counts, combined, 1.0, flag, no_lora, True, 1)
        if use_gmm:
            gathered = [
                torch.cat([part[s].view(rows, slots, local_rank) for part in scratch], dim=-1).reshape(
                    rows, slots * full_rank
                )
                for s in range(2)
            ]
            gathered.append(unused_gathered)
        else:
            gathered = scratch[0][:2] + [torch.cat([part[-1] for part in scratch], dim=-1)]
        turbo_ops.add_lora_expand(
            target, gathered, b, combined, counts, combined, list(widths), 0, True, flag, no_lora, True, 1
        )
        # Production GMM may return the standalone delta for later folding.
        delta = turbo_ops.add_lora_expand_delta(target, gathered, b, counts, 1) if use_gmm else None
        return target, gathered, delta

    graph, stream, outputs = _capture(run)
    max_expand_error, max_shrink_error = 0.0, 0.0
    try:
        for iteration in range(REPLAYS):
            x_cpu = _random(x.shape, 2000 + iteration, 0.3)
            base = _random(y.shape, 2100 + iteration, 0.2)
            expert_ids = torch.arange(rows) // (rows // experts)
            lora_slots = torch.tensor(
                [
                    -1 if iteration == REPLAYS - 1 or (row + iteration) % 5 == 0 else (row + iteration) % slots
                    for row in range(rows)
                ]
            )
            idx_cpu = torch.where(lora_slots >= 0, lora_slots * experts + expert_ids, -1)
            shrink_ref = torch.zeros(2, rows, full_rank)
            expected = base.float().clone()
            for row, slot in enumerate(lora_slots.tolist()):
                if slot < 0:
                    continue
                expert = int(expert_ids[row])
                for s, width in enumerate(widths):
                    z = x_cpu[row].float() @ a_cpu[s][slot, expert].float().T
                    shrink_ref[s, row] = z
                    expected[row, s * width : (s + 1) * width] += z @ b_cpu[s][slot, expert].float().T
            with torch.npu.stream(stream):
                x.copy_(x_cpu)
                y.copy_(base)
                combined.copy_(idx_cpu)
                graph.replay()
            stream.synchronize()
            actual_y = outputs[0].cpu()
            expand_error = (actual_y.float() - expected).abs().max().item()
            max_expand_error = max(max_expand_error, expand_error)
            torch.testing.assert_close(actual_y.float(), expected, atol=3e-2, rtol=0)
            if use_gmm:
                gathered_cpu = [value.cpu().view(rows, slots, full_rank) for value in outputs[1][:2]]
                for s in range(2):
                    expected_stacked = torch.zeros(rows, slots, full_rank)
                    for row, slot in enumerate(lora_slots.tolist()):
                        if slot >= 0:
                            expected_stacked[row, slot] = shrink_ref[s, row]
                    error = (gathered_cpu[s] - expected_stacked).abs().max().item()
                    max_shrink_error = max(max_shrink_error, error)
                    torch.testing.assert_close(gathered_cpu[s], expected_stacked, atol=3e-2, rtol=0)
                torch.testing.assert_close((base + outputs[2].cpu()).float(), expected, atol=3e-2, rtol=0)
            else:
                actual_shrink = outputs[1][-1].cpu()
                max_shrink_error = max(max_shrink_error, (actual_shrink - shrink_ref).abs().max().item())
                torch.testing.assert_close(actual_shrink, shrink_ref, atol=3e-2, rtol=0)
            with torch.npu.stream(stream):
                eager, _, eager_delta = run(base.to(DEV))
            stream.synchronize()
            assert torch.equal(actual_y, eager.cpu()), "rank2/gather/rank16 graph differs from eager"
            if use_gmm:
                assert torch.equal(outputs[2].cpu(), eager_delta.cpu())
        _metric(
            "fully_sharded_buffer_contract",
            use_gmm=use_gmm,
            rows=rows,
            hidden=hidden,
            local_rank=local_rank,
            full_rank=full_rank,
            tp_size=tp_size,
            widths=widths,
            max_shrink_abs_error=max_shrink_error,
            max_expand_abs_error=max_expand_error,
            graph_eager_bitwise_equal=True,
            collective="single-device concat layout only",
        )
    finally:
        graph.reset()
