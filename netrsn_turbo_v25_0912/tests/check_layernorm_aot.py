# SPDX-License-Identifier: Apache-2.0

"""CPU regression for the Turbo patch, without installing the NPU-only stack.

Run this script with PyTorch 2.10 from this workspace. Only the NPU kernel,
unrelated vLLM imports and TurboManager are replaced by test doubles.
"""

import importlib
import importlib.machinery
import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from torch._dynamo.aot_compile_types import SerializableCallable

ROOT = Path(__file__).resolve().parents[2]
TURBO = ROOT / "netrsn_turbo_v25_0912"
PATCH_MODULE = "netrsn_turbo.turbo.version_v0250.vllm_ascend.ops.layernorm"
MANAGER_MODULE = "netrsn_turbo.turbo_manager.version_v0250.turbo_layernorm_aot"
TARGET = "vllm_ascend.ops.layernorm.AscendRMSNormGated.forward_oot"


def stub(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class TurboManagerForTest:
    registrations = []
    apply_count = 0

    @classmethod
    def register_patch(cls, target, replacement):
        cls.registrations.append((target, replacement))

    @classmethod
    def apply_patches(cls):
        cls.apply_count += 1
        for target, replacement in cls.registrations:
            parts = target.split(".")
            for index in range(len(parts) - 1, 0, -1):
                module = sys.modules.get(".".join(parts[:index]))
                if module is not None:
                    break
            else:
                raise AssertionError(f"Unknown patch target: {target}")
            for name in parts[index:-1]:
                module = getattr(module, name)
            setattr(module, parts[-1], replacement)


def load_runtime():
    stub("vllm", __path__=[])
    stub("vllm.config", get_current_vllm_config=lambda: None)
    stub("vllm.model_executor", __path__=[])
    stub("vllm.model_executor.layers", __path__=[])
    stub(
        "vllm.model_executor.layers.layernorm",
        RMSNorm=torch.nn.Module,
        GemmaRMSNorm=torch.nn.Module,
        RMSNormGated=torch.nn.Module,
    )
    ascend = ROOT / "vllm-ascend" / "vllm_ascend"
    stub("vllm_ascend", __path__=[str(ascend)])
    stub("vllm_ascend.ops", __path__=[str(ascend / "ops")])
    stub("vllm_ascend.device", __path__=[])
    stub("vllm_ascend.device.device_op", DeviceOperator=object)
    stub("vllm_ascend.utils", enable_custom_op=lambda: False)
    stub("vllm_ascend.ops.triton", __path__=[])
    stub(
        "vllm_ascend.ops.triton.layernorm_gated",
        layer_norm_fwd_npu=reference_kernel,
    )
    stub("netrsn_turbo", __path__=[str(TURBO)])
    stub(
        "netrsn_turbo.turbo_manager.turbo_utils",
        TurboManager=TurboManagerForTest,
    )
    layernorm = importlib.import_module("vllm_ascend.ops.layernorm")
    originals = (
        layernorm.AscendRMSNorm.forward_oot,
        layernorm.AscendGemmaRMSNorm.forward_oot,
        layernorm.LayerNormFn,
    )
    manager = importlib.import_module(MANAGER_MODULE)
    manager.apply_layernorm_aot_patch()
    return layernorm, manager, originals


def reference_kernel(
    x,
    weight,
    bias,
    eps,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    assert is_rms_norm
    values = x.float()
    gate = torch.nn.functional.silu(z.float()) if z is not None else None
    if gate is not None and not norm_before_gate:
        values = values * gate
    groups = values.reshape(*values.shape[:-1], -1, group_size or x.shape[-1])
    values = (
        groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)
    ).reshape_as(values)
    values = values * weight.float()
    if bias is not None:
        values = values + bias.float()
    if gate is not None and norm_before_gate:
        values = values * gate
    return values.to(x.dtype).contiguous(), None, torch.empty(0)


class GatedNormForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8), requires_grad=False)
        self.bias = None
        self.eps = 1e-5
        self.group_size = None
        self.norm_before_gate = False

    def forward(self, x, z=None):
        return layernorm.AscendRMSNormGated.forward_oot(self, x, z)


def check_graph(graph):
    targets = [
        str(node.target) for node in graph.graph.nodes if node.op == "call_function"
    ]
    assert "vllm_ascend.rms_norm_gated.default" in targets, targets
    assert not any("triton_kernel_wrapper" in target for target in targets)


class TestLayernormAOT(unittest.TestCase):
    def test_installation_is_idempotent_and_scoped(self):
        forward = layernorm.AscendRMSNormGated.forward_oot
        apply_count = TurboManagerForTest.apply_count
        for _ in range(2):
            manager.apply_layernorm_aot_patch()
        self.assertEqual(TurboManagerForTest.apply_count, apply_count)
        self.assertEqual(TurboManagerForTest.registrations, [(TARGET, forward)])
        self.assertEqual(
            originals,
            (
                layernorm.AscendRMSNorm.forward_oot,
                layernorm.AscendGemmaRMSNorm.forward_oot,
                layernorm.LayerNormFn,
            ),
        )

    def test_preserves_kernel_arguments_and_output(self):
        cases = product(
            (torch.float32, torch.bfloat16),
            (None, 4),
            (False, True),
            (False, True),
            (False, True),
        )
        with (
            torch.inference_mode(),
            patch.object(
                layernorm, "layer_norm_fwd_npu", side_effect=reference_kernel
            ) as kernel,
        ):
            for dtype, group_size, before, has_gate, has_bias in cases:
                with self.subTest(
                    dtype=dtype,
                    group=group_size,
                    before=before,
                    gate=has_gate,
                    bias=has_bias,
                ):
                    model = GatedNormForTest()
                    model.group_size = group_size
                    model.norm_before_gate = before
                    model.weight = torch.nn.Parameter(
                        torch.randn(16, dtype=dtype)[::2], requires_grad=False
                    )
                    model.bias = torch.randn(16, dtype=dtype)[::2] if has_bias else None
                    x = torch.randn(2, 3, 16, dtype=dtype)[..., ::2]
                    z = (
                        torch.randn(2, 3, 16, dtype=dtype)[..., ::2]
                        if has_gate
                        else None
                    )
                    tensors = [x, model.weight, model.bias, z]
                    snapshots = [t.clone() for t in tensors if t is not None]
                    result = model(x, z) if has_gate else model(x)
                    expected = reference_kernel(
                        x,
                        model.weight,
                        model.bias,
                        model.eps,
                        z,
                        group_size,
                        before,
                        True,
                    )[0]
                    torch.testing.assert_close(result, expected)
                    for tensor, snapshot in zip(
                        (t for t in tensors if t is not None), snapshots
                    ):
                        torch.testing.assert_close(tensor, snapshot)
                    self.assertTrue(result.is_contiguous())
                    call = kernel.call_args
                    self.assertEqual(call.args[0].shape, (6, 8))
                    self.assertEqual(call.args[0].stride(-1), 1)
                    self.assertEqual(call.args[3], model.eps)
                    self.assertEqual(call.kwargs["group_size"], group_size)
                    self.assertEqual(call.kwargs["norm_before_gate"], before)
                    self.assertTrue(call.kwargs["is_rms_norm"])

    def test_fake_metadata_and_schema(self):
        for has_gate in (False, True):
            x = torch.randn(2, 3, 16)[..., ::2]
            z = torch.randn_like(x) if has_gate else None
            torch.library.opcheck(
                torch.ops.vllm_ascend.rms_norm_gated.default,
                (x, torch.ones(8), None, z, 1e-5, None, False),
                test_utils=("test_schema", "test_faketensor"),
            )

    def test_compiled_graph_keeps_gated_norm_opaque(self):
        graphs = []

        def backend(graph, inputs):
            check_graph(graph)
            graphs.append(graph)
            return graph.forward

        torch._dynamo.reset()
        try:
            model = GatedNormForTest()
            with torch.inference_mode():
                compiled = torch.compile(
                    model, backend=backend, fullgraph=True, dynamic=True
                )
                for tokens in (2, 7, 1):
                    x, z = torch.randn(tokens, 8), torch.randn(tokens, 8)
                    torch.testing.assert_close(compiled(x, z), model(x, z))
            self.assertGreaterEqual(len(graphs), 2)
        finally:
            torch._dynamo.reset()

    def test_reuses_registered_operator(self):
        operator = torch.ops.vllm_ascend.rms_norm_gated.default
        with patch.object(
            torch.library,
            "custom_op",
            side_effect=AssertionError("Must not register the operator twice"),
        ):
            module = importlib.reload(importlib.import_module(PATCH_MODULE))
        self.assertIs(module._rms_norm_gated, operator)
        with torch.inference_mode():
            model = GatedNormForTest()
            x, z = torch.randn(2, 8), torch.randn(2, 8)
            expected = reference_kernel(
                x, model.weight, None, model.eps, z, None, False, True
            )[0]
            torch.testing.assert_close(model(x, z), expected)


class SerializableFX(SerializableCallable):
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, *args):
        return self.graph(*args)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        for node in fn.graph.graph.nodes:
            node.meta.clear()
        return pickle.dumps(fn.graph)

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        return cls(pickle.loads(data))


def aot_backend(graph, inputs):
    check_graph(graph)
    return SerializableFX(graph)


def gated_forward(x, z, weight):
    return torch.ops.vllm_ascend.rms_norm_gated.default(
        x, weight, None, z, 1e-5, None, False
    )


def roundtrip(mode, directory):
    torch.manual_seed(0)
    with torch.inference_mode():
        for variant, tokens in (("lora", 4), ("base", 8), ("base_one", 1)):
            x = torch.randn(tokens, 8)
            z, weight = torch.randn_like(x), torch.randn(8)
            path = directory / ("model." + variant)
            if mode == "save":
                torch._dynamo.reset()
                if tokens > 1:
                    torch._dynamo.mark_dynamic(x, 0)
                    torch._dynamo.mark_dynamic(z, 0)
                with torch._dynamo.config.patch(enable_aot_compile=True):
                    fn = torch.compile(
                        gated_forward, backend=aot_backend, fullgraph=True
                    )
                    artifact = fn.aot_compile(((x, z, weight), {}))
                artifact.save_compiled_function(str(path))
            else:
                with path.open("rb") as cache:
                    artifact = torch.compiler.load_compiled_function(
                        cache, f_globals=gated_forward.__globals__
                    )
                if tokens > 1:
                    x, z = torch.randn(tokens + 3, 8), torch.randn(tokens + 3, 8)
                artifact.disable_guard_check()
            torch.testing.assert_close(
                artifact(x, z, weight), gated_forward(x, z, weight)
            )
            print(f"{mode}: {variant} OK", flush=True)


if __name__ == "__main__":
    layernorm, manager, originals = load_runtime()
    if len(sys.argv) > 1:
        roundtrip(sys.argv[1], Path(sys.argv[2]))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLayernormAOT)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        with tempfile.TemporaryDirectory(prefix="turbo-layernorm-aot-") as cache:
            for mode in ("save", "load"):
                subprocess.run(
                    [sys.executable, "-B", __file__, mode, cache],
                    check=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
