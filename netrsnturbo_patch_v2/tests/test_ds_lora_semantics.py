# SPDX-License-Identifier: Apache-2.0
"""Execute the changed Python control flow without torch, vLLM, or an NPU.

The small tensor/operator doubles below model the particular scalar LoRA
example and metadata exchanges under test; they do not validate NPU kernels.
"""

from __future__ import annotations

import ast
import contextlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PAYLOAD = Path(__file__).resolve().parents[1] / "netrsn_turbo/turbo/version_0300/ds_lora"


def load_functions(filename, names, **dependencies):
    tree = ast.parse((PAYLOAD / filename).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing requested functions in {filename}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {"log_once": lambda *args: None, **dependencies}
    exec(compile(ast.fix_missing_locations(module), str(PAYLOAD / filename), "exec"), namespace)
    return namespace


class Tensor:
    """A flat, shared-storage tensor double for tiny deterministic examples."""

    def __init__(self, values, shape, dtype="float32", *, share=False):
        self.values = values if share else list(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "npu:0"
        assert len(self.values) == math.prod(self.shape)

    def numel(self):
        return len(self.values)

    def item(self):
        raise AssertionError("Device scalar synchronization is not allowed in these paths")

    def view(self, *shape):
        if -1 in shape:
            size = math.prod(dimension for dimension in shape if dimension != -1)
            shape = tuple(self.numel() // size if dimension == -1 else dimension for dimension in shape)
        return Tensor(self.values, shape, self.dtype, share=True)

    reshape = view

    def to(self, dtype):
        return Tensor(self.values, self.shape, dtype)

    def __getitem__(self, indices):
        rows, columns = indices
        rows = list(range(self.shape[0]))[rows]
        columns = list(range(self.shape[1]))[columns]
        return Tensor(
            [self.values[row * self.shape[1] + column] for row in rows for column in columns],
            (len(rows), len(columns)),
            self.dtype,
        )

    def __iadd__(self, other):
        self.values[:] = [left + right for left, right in zip(self.values, other.values)]
        return self

    def untyped_storage(self):
        return types.SimpleNamespace(resize_=lambda size: None)


def fake_torch(**extra):
    def empty(shape, dtype, device):
        return Tensor([0] * math.prod(shape), shape, dtype)

    def cat(tensors, dim):
        assert dim == 1
        rows = tensors[0].shape[0]
        values = []
        for row in range(rows):
            for tensor in tensors:
                width = tensor.shape[1]
                values.extend(tensor.values[row * width : (row + 1) * width])
        return Tensor(values, (rows, sum(t.shape[1] for t in tensors)), tensors[0].dtype)

    return types.SimpleNamespace(
        Tensor=Tensor, bfloat16="bfloat16", float16="float16", float32="float32",
        int8="int8", long="long", empty=empty, cat=cat, **extra,
    )


class HostFlag:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FullyShardedW2Tests(unittest.TestCase):
    def test_reduction_preserves_cross_rank_contributions(self):
        # A=[1, 1], x=[2, 3], B=[7, 11]^T gives [35, 55]. Expanding
        # each rank's own shrink into its own B slice gives [14, 33].
        reference = [(2 + 3) * coefficient for coefficient in (7, 11)]
        self.assertNotEqual(reference, [2 * 7, 3 * 11])
        for use_gmm in (False, True):
            for phase in ("full", "expand"):
                with self.subTest(use_gmm=use_gmm, phase=phase):
                    rank_outputs = []
                    reductions = []
                    for rank, (partial, coefficient) in enumerate(((2, 7), (3, 11))):
                        def reduce(buffer):
                            reductions.append((rank, buffer.shape))
                            self.assertEqual(buffer.values, [partial])
                            return Tensor([5], buffer.shape)

                        def shrink(buffers, *args):
                            buffers[0].values[:] = [partial]
                            buffers[1].values[:] = [partial]

                        def expand(y, buffers, weights, *args):
                            offset = args[4]
                            reduced = buffers[0 if use_gmm else 1]
                            y.values[offset] += reduced.values[0] * weights[0].values[0]

                        def unexpected_gather(*args, **kwargs):
                            self.fail("W2 must reduce input contributions, not gather LoRA ranks")

                        backend = types.SimpleNamespace(add_lora_shrink=shrink, add_lora_expand=expand)
                        torch = fake_torch()
                        apply = load_functions(
                            "vllm_ascend/lora/punica_npu.py", {"add_lora_fused_moe"}, torch=torch,
                            get_lora_ops=lambda: backend,
                            ENABLE_ADD_LORA_KERNEL=True,
                            get_tensor_model_parallel_world_size=lambda: 2,
                            tensor_model_parallel_all_reduce=reduce,
                            tensor_model_parallel_all_gather=unexpected_gather,
                        )["add_lora_fused_moe"]
                        wrapper = types.SimpleNamespace(
                            _use_moe_gmm_cpu=HostFlag(use_gmm), _no_lora_cpu=HostFlag(False),
                        )
                        output = Tensor([0, 0], (1, 2))
                        apply(
                            wrapper, output, Tensor([partial], (1, 1)),
                            (Tensor([1], (1, 1, 1, 1)),), (Tensor([coefficient], (1, 1, 1, 1)),),
                            adapter_enabled=Tensor([1], (1,)), fully_sharded=True,
                            # External callers using the old option must also remain correct.
                            partial_expand=True, offset=rank,
                            group_list=Tensor([1], (1,)), combined_idx=Tensor([0], (1,)),
                            phase=phase,
                            shrink_buffers=[Tensor([partial], (1, 1)), Tensor([partial], (1, 1, 1))],
                        )
                        rank_outputs.append(output.values)
                    self.assertEqual([sum(values) for values in zip(*rank_outputs)], reference)
                    self.assertEqual(len(reductions), 2)

    def test_regular_w2_caller_keeps_collective_and_output_offset(self):
        calls = []
        context = types.SimpleNamespace(
            combined_lora_idx=Tensor([0], (1,)), fully_sharded=True, tp_rank=1, tp_size=2, use_ep=False,
            w2_lora_a_stacked=(Tensor([1], (1, 1, 1, 1)),),
            w2_lora_b_stacked=(Tensor([11], (1, 1, 1, 1)),), adapter_enabled=Tensor([1], (1,)),
            punica_wrapper=types.SimpleNamespace(add_lora_fused_moe=lambda **kwargs: calls.append(kwargs)),
        )
        reset = []
        apply = load_functions("vllm_ascend/lora/fused_moe.py", {"moe_lora_apply_w2"}, reset_lora_indices=reset.append)[
            "moe_lora_apply_w2"
        ]
        apply(context, down_out=Tensor([0, 0], (1, 2)), silu_out=Tensor([3], (1, 1)), lora_routing=None)
        self.assertFalse(calls[0].get("partial_expand", False))
        self.assertTrue(calls[0]["fully_sharded"])
        self.assertEqual(calls[0]["offset"], 1)
        self.assertEqual(reset, [context])

    def test_overlap_w2_collective_runs_in_main_stream_expand_phase(self):
        active_stream = ["main"]
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                events.append((self.name, "wait"))

            def record_event(self):
                return Event()

        class Event:
            def record(self, stream):
                events.append((stream.name, "record"))

        @contextlib.contextmanager
        def stream_scope(stream):
            active_stream[0] = stream.name
            try:
                yield
            finally:
                active_stream[0] = "main"

        calls = []
        a13 = Tensor([1], (1, 1, 1, 1))
        a2 = Tensor([2], (1, 1, 1, 1))

        def apply(**kwargs):
            calls.append((active_stream[0], kwargs))
            return [] if kwargs["phase"] == "shrink" else None

        context = types.SimpleNamespace(
            punica_wrapper=types.SimpleNamespace(add_lora_fused_moe=apply),
            w13_lora_a_stacked=(a13,), w13_lora_b_stacked=(a13,),
            w2_lora_a_stacked=(a2,), w2_lora_b_stacked=(a2,),
            adapter_enabled=Tensor([1], (1,)), fully_sharded=True, tp_size=2, tp_rank=1, use_ep=False,
        )
        mlp_input = types.SimpleNamespace(
            lora_context=context, group_list=Tensor([1], (1,)), group_list_type=1,
            expanded_row_idx=None, topk_ids=None, activation=None, swiglu_limit=None,
            swiglu_alpha=None, swiglu_beta=None, topk_scales=None,
        )
        main = Stream("main")
        torch = fake_torch(npu=types.SimpleNamespace(
            current_stream=lambda: main, stream=stream_scope, Event=Event,
        ))
        module = types.ModuleType("vllm_ascend.lora.fused_moe")
        module._build_combined_lora_idx_allgather = lambda *args: Tensor([0], (1,))
        run = load_functions(
            "vllm_ascend/lora/quant_moe.py", {"_apply_dynamic_int8_moe_lora_overlap_phase"}, torch=torch,
            _lora_aux_stream=lambda: Stream("aux"),
            DeviceOperator=types.SimpleNamespace(
                npu_dynamic_quant=lambda **kwargs: (kwargs["hidden_states"], Tensor([1], (1,))),
                npu_grouped_matmul_gmm2=lambda **kwargs: Tensor([0, 0], (1, 2)),
            ),
            torch_npu=types.SimpleNamespace(npu_grouped_matmul=lambda **kwargs: [Tensor([0, 0], (1, 2))]),
            _apply_moe_activation=lambda *args: Tensor([3], (1, 1)),
        )["_apply_dynamic_int8_moe_lora_overlap_phase"]
        with patch.dict(sys.modules, {module.__name__: module}):
            run(mlp_compute_input=mlp_input, hidden_states=Tensor([2], (1, 1)), w1=None, w2=None,
                w1_scale=Tensor([1], (1,)), w2_scale=Tensor([1], (1,)))
        self.assertEqual([(stream, call["phase"]) for stream, call in calls], [
            ("aux", "shrink"), ("main", "expand"), ("aux", "shrink"), ("main", "expand"),
        ])
        for _, call in calls[2:]:
            self.assertIs(call["lora_a_stacked"][0], a2)
            self.assertTrue(call["fully_sharded"])
            self.assertFalse(call.get("partial_expand", False))
            self.assertEqual(call["offset"], 1)
        self.assertIn(("main", "wait"), events)


class AllToAllSlotTests(unittest.TestCase):
    def _dispatch(self, capacity, slots, dtype="bfloat16", *, with_context=True):
        rows = 1 if slots is None else len(slots)
        hidden = Tensor([10, 20] * rows, (rows, 2), dtype)
        context = types.SimpleNamespace(punica_wrapper=types.SimpleNamespace(
            lora_config=types.SimpleNamespace(max_loras=capacity), no_lora=False,
        )) if with_context else None
        if slots is not None and context is not None:
            context.permuted_lora_indices = Tensor(slots, (rows,), "long")
        exchanges, fallback_calls = [], []

        def exchange(tensor, output_splits, input_splits, group):
            exchanges.append((tensor.shape, output_splits, input_splits, group))
            return None, Tensor(tensor.values, tensor.shape, tensor.dtype), types.SimpleNamespace(wait=lambda: None)

        def fallback(lora_context, **kwargs):
            fallback_calls.append(kwargs)
            if getattr(lora_context, "permuted_lora_indices", None) is not None:
                lora_context.exchanged_lora_indices = lora_context.permuted_lora_indices

        module = types.ModuleType("vllm_ascend.lora.fused_moe")
        module.all2all_lora_indices = fallback
        namespace = load_functions(
            "vllm_ascend/ops/fused_moe/token_dispatcher.py", {"_can_pack_lora_slots", "token_dispatch"}, torch=fake_torch(),
            has_lora=lambda value: value is not None, async_all_to_all=exchange,
            MoETokenDispatchOutput=types.SimpleNamespace, MoEAllToAllCombineMetadata=types.SimpleNamespace,
        )
        dispatcher = types.SimpleNamespace(
            lora_context=context, ep_group="ep",
            _dispatch_preprocess=lambda *args: (hidden, None, [rows], [rows], [rows], None, hidden.shape, hidden.shape),
            _dispatch_postprocess=lambda values, scales, *args: (values, scales, None),
        )
        request = types.SimpleNamespace(
            quant=types.SimpleNamespace(is_mxfp=False, dispatch_with_quant=False, is_quant=False, get_dst_type=dtype),
            hidden_states=hidden, topk_weights=None, topk_ids=None,
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            result = namespace["token_dispatch"](dispatcher, request)
        self.assertEqual(result.hidden_states.values, hidden.values)
        return context, exchanges, fallback_calls

    def test_exact_capacity_boundary_packs_without_extra_collective(self):
        for dtype, last_exact_slot in (("bfloat16", 256), ("float16", 2048), ("float32", 2**24)):
            with self.subTest(dtype=dtype):
                context, exchanges, fallback = self._dispatch(last_exact_slot + 1, [-1, last_exact_slot], dtype)
                self.assertEqual(exchanges[0][0], (2, 3))
                self.assertEqual(context.exchanged_lora_indices.values, [-1, last_exact_slot])
                self.assertEqual(fallback, [])

    def test_large_capacity_uses_exact_integer_exchange_even_for_small_active_slot(self):
        # Different ranks see different local slots but must make the same
        # collective decision from their common configured capacity.
        for slots in ([0], [257], []):
            with self.subTest(slots=slots):
                context, exchanges, fallback = self._dispatch(258, slots)
                self.assertEqual(exchanges[0][0], (len(slots), 2))
                self.assertEqual(context.exchanged_lora_indices.values, slots)
                self.assertEqual(fallback, [{"output_splits": [len(slots)], "input_splits": [len(slots)], "ep_group": "ep"}])

    def test_empty_rank_uses_same_packed_width_when_capacity_is_safe(self):
        _, exchanges, fallback = self._dispatch(4, [])
        self.assertEqual(exchanges[0][0], (0, 3))
        self.assertEqual(fallback, [])

    def test_missing_metadata_preserves_original_fallback_call(self):
        context, exchanges, fallback = self._dispatch(4, None)
        self.assertEqual(exchanges[0][0], (1, 2))
        self.assertEqual(len(fallback), 1)
        self.assertFalse(hasattr(context, "exchanged_lora_indices"))

    def test_unknown_capacity_or_dtype_uses_original_exchange(self):
        for capacity, dtype in ((None, "bfloat16"), (4, "int8")):
            with self.subTest(capacity=capacity, dtype=dtype):
                _, exchanges, fallback = self._dispatch(capacity, [0], dtype)
                self.assertEqual(exchanges[0][0], (1, 2))
                self.assertEqual(len(fallback), 1)

    def test_no_lora_context_does_not_exchange_indices(self):
        _, exchanges, fallback = self._dispatch(None, None, with_context=False)
        self.assertEqual(exchanges[0][0], (1, 2))
        self.assertEqual(fallback, [])


if __name__ == "__main__":
    unittest.main()
