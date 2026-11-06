# SPDX-License-Identifier: Apache-2.0
"""DS v2 dispatch regressions; execute real Python functions without torch."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from ._entry_scope import ds_patch_specs

DELIVERY = Path(__file__).resolve().parents[1]
WORKSPACE = DELIVERY.parent
PAYLOAD = DELIVERY / "netrsn_turbo/turbo/version_v0300"
PUNICA = PAYLOAD / "vllm_ascend/lora/punica_npu.py"
BASE_PUNICA = WORKSPACE / "vllm-ascend/vllm_ascend/lora/punica_npu.py"
DS_PUNICA = WORKSPACE / "vllm-ascend-ds-v2/vllm_ascend/lora/punica_npu.py"
BASE_HELPERS = ("_lora_linear_kernel", "_get_shrink_buffer", "_lora_linear_matmul")


def load_function(source, name, **dependencies):
    """Keep the function's control flow intact and replace only dependencies."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"log_once": lambda *args: None, **dependencies}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


class SingleSlotDispatchTests(unittest.TestCase):
    def setUp(self):
        self.registry_op = mock.Mock(name="upstream_lora_linear")
        self.path_log = mock.Mock()
        self.add_linear = load_function(
            PUNICA,
            "add_lora_linear",
            lora_linear=self.registry_op,
            ENABLE_ADD_LORA_KERNEL=True,
            log_once=self.path_log,
        )
        self.wrapper = types.SimpleNamespace(
            _single_lora_slot=True,
            no_lora=False,
            _wrapper_id=47,
            add_lora=mock.Mock(),
            add_shrink=mock.Mock(),
            add_expand=mock.Mock(),
        )
        self.y, self.x = object(), object()
        self.a, self.b = (object(), object()), (object(), object())
        self.slices = (16, 32)
        self.scale = 0.25

    def invoke(self, **kwargs):
        self.add_linear(self.wrapper, self.y, self.x, self.a, self.b, self.scale, self.slices, **kwargs)

    def test_active_single_slot_dispatches_to_existing_wrapper_registry(self):
        self.invoke()
        self.registry_op.assert_called_once_with(
            47, self.y, self.x, list(self.a), list(self.b), self.scale, list(self.slices), None, None, True
        )
        self.wrapper.add_lora.assert_not_called()
        self.wrapper.add_shrink.assert_not_called()
        self.wrapper.add_expand.assert_not_called()
        self.assertEqual(self.path_log.call_args.args[0], "linear.single_slot_registry")

    def test_single_slot_preserves_packed_weights_and_add_inputs(self):
        packed_a, packed_b = object(), object()
        self.invoke(packed_lora_a=packed_a, packed_lora_b=packed_b, add_inputs=False)
        self.registry_op.assert_called_once_with(
            47, self.y, self.x, list(self.a), list(self.b), self.scale, list(self.slices), packed_a, packed_b, False
        )
        self.wrapper.add_lora.assert_not_called()

    def test_no_lora_and_multiple_slots_keep_generic_fused_dispatch(self):
        for single_slot, no_lora in ((True, True), (False, False), (False, True)):
            with self.subTest(single_slot=single_slot, no_lora=no_lora):
                self.wrapper._single_lora_slot = single_slot
                self.wrapper.no_lora = no_lora
                self.wrapper.add_lora.reset_mock()
                marker = object()
                self.invoke(add_inputs=False, packed_lora_a=marker)
                self.wrapper.add_lora.assert_called_once_with(
                    self.y,
                    self.x,
                    self.a,
                    self.b,
                    self.scale,
                    self.slices,
                    add_inputs=False,
                    packed_lora_a=marker,
                )
                self.registry_op.assert_not_called()
                self.wrapper.add_shrink.assert_not_called()
                self.wrapper.add_expand.assert_not_called()
                self.assertEqual(self.path_log.call_args.args[0], "linear.turbo_ops_add_lora")

    def test_explicit_buffer_bypasses_both_fused_dispatches(self):
        buffer, marker = (object(), object()), object()
        self.invoke(buffer=buffer, add_inputs=False, marker=marker)
        self.registry_op.assert_not_called()
        self.wrapper.add_lora.assert_not_called()
        self.wrapper.add_shrink.assert_called_once_with(buffer, self.x, self.a, self.scale, marker=marker)
        # The pre-existing two-stage path accumulates its expand result.
        self.wrapper.add_expand.assert_called_once_with(
            self.y, buffer, self.b, self.slices, add_inputs=True, marker=marker
        )
        self.assertEqual(self.path_log.call_args.args[0], "linear.shrink_expand")

    def test_single_slot_fastpath_does_not_depend_on_generic_kernel_flag(self):
        self.add_linear.__globals__["ENABLE_ADD_LORA_KERNEL"] = False
        self.invoke()
        self.registry_op.assert_called_once()
        self.wrapper.add_lora.assert_not_called()
        self.wrapper.add_shrink.assert_not_called()


class BaseHelperPreservationTests(unittest.TestCase):
    def test_payload_uses_upstream_registry_and_keeps_base_helpers(self):
        tree = ast.parse(PUNICA.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.lora.lora_ops" and node.level == 0
            for alias in node.names
        }
        self.assertTrue({"_LORA_WRAPPERS", "_LORA_WRAPPER_IDS", "lora_linear"}.issubset(imports))
        payload_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        specs = ds_patch_specs()
        registered = {symbol for module, _, existing, additions in specs for symbol in existing + additions}
        for name in BASE_HELPERS:
            with self.subTest(helper=name):
                self.assertNotIn(name, payload_functions)
                self.assertNotIn("PunicaWrapperNPU." + name, registered)

    @unittest.skipUnless(BASE_PUNICA.is_file() and DS_PUNICA.is_file(), "source checkouts not present")
    def test_v2_reintroduced_helpers_are_ast_identical_to_base(self):
        def methods(source):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PunicaWrapperNPU")
            return {node.name: ast.dump(node) for node in wrapper.body if isinstance(node, ast.FunctionDef)}

        baseline, ds_v2 = methods(BASE_PUNICA), methods(DS_PUNICA)
        for name in BASE_HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(ds_v2[name], baseline[name])


class QuantModuleLookupTests(unittest.TestCase):
    def test_quant_helpers_follow_runtime_module_attribute_replacements(self):
        """A cached payload function must observe later upstream monkeypatches."""
        upstream = types.ModuleType("vllm_ascend.lora.quant_moe")
        ascend = types.ModuleType("vllm_ascend")
        ascend.__path__ = []
        lora = types.ModuleType("vllm_ascend.lora")
        lora.__path__ = []
        ascend.lora, lora.quant_moe = lora, upstream
        modules = {module.__name__: module for module in (ascend, lora, upstream)}
        hidden = types.SimpleNamespace(dtype="bf16", shape=(2, 8))
        scale = types.SimpleNamespace(dtype="float32", to=mock.Mock(return_value=object()))
        weights = types.SimpleNamespace(
            w1=object(),
            w2=object(),
            w1_scale=scale,
            w2_scale=scale,
            w1_scale_bias=None,
            w2_scale_bias=None,
            w1_offset=None,
            w2_offset=None,
        )
        inputs = types.SimpleNamespace(
            lora_context=object(),
            dynamic_eplb=False,
            hidden_states=hidden,
            dynamic_scale=None,
            weights=weights,
            group_list=object(),
            group_list_type=1,
            activation=object(),
            swiglu_limit=1.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            topk_scales=None,
        )
        gate, activated, output, event = object(), object(), object(), object()
        device_ops = types.SimpleNamespace(
            npu_dynamic_quant=mock.Mock(return_value=(object(), object())),
            npu_grouped_matmul_gmm2=mock.Mock(return_value=output),
        )
        apply = load_function(
            PAYLOAD / "vllm_ascend/lora/quant_moe.py",
            "_apply_dynamic_int8_moe_lora",
            _EXTRA_CTX=types.SimpleNamespace(moe_comm_type="alltoall"),
            MoECommType=types.SimpleNamespace(ALLGATHER="allgather", ALLTOALL="alltoall"),
            torch=types.SimpleNamespace(
                int8="int8",
                bfloat16="bf16",
                npu=types.SimpleNamespace(current_stream=lambda: types.SimpleNamespace(record_event=lambda: event)),
            ),
            torch_npu=types.SimpleNamespace(npu_grouped_matmul=mock.Mock(return_value=[gate])),
            DeviceOperator=device_ops,
            _lora_overlap_eligible=lambda *args: None,
            _fused_w13_epilogue_eligible=lambda *args: False,
            _apply_moe_activation=mock.Mock(return_value=activated),
        )
        with mock.patch.dict(sys.modules, modules):
            for generation in range(2):
                with self.subTest(generation=generation):
                    routing = object()
                    upstream._recover_moe_lora_routing_all2all = recover = mock.Mock(return_value=routing)
                    upstream.moe_lora_apply_w13 = w13 = mock.Mock(return_value=None)
                    upstream.moe_lora_apply_w2 = w2 = mock.Mock()
                    self.assertEqual(apply(inputs), (output, event))
                    recover.assert_called_once_with(inputs.lora_context, group_list=inputs.group_list)
                    w13.assert_called_once_with(
                        inputs.lora_context,
                        gate_up_out=gate,
                        hidden_states=hidden,
                        lora_routing=routing,
                        group_list=inputs.group_list,
                        group_list_type=1,
                    )
                    w2.assert_called_once_with(
                        inputs.lora_context,
                        down_out=output,
                        silu_out=activated,
                        lora_routing=routing,
                        group_list=inputs.group_list,
                        group_list_type=1,
                    )


if __name__ == "__main__":
    unittest.main()
