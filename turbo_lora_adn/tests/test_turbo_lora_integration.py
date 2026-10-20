# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; private Netrsn/CANN packages are replaced by spies.

Run this file separately for each adapter folder. These checks exercise real
torch arithmetic, but do not establish NPU accuracy, ACLGraph, or performance.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OWNER = "adn" if ROOT.name == "turbo_lora_adn" else "shared_gemm"
PACKAGE = f"netrsn_turbo.turbo.version_0230.lora_{OWNER}_310p"
ENTRY = f"netrsn_turbo.turbo_manager.turbo_lora_{OWNER}_0230_310p"
PUNICA = "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"


def _project(x, a, b, slots, scale, dtype):
    result = torch.zeros(x.shape[0], b.shape[-2], dtype=dtype)
    for row, slot in enumerate(slots):
        if slot >= 0:
            value = x[row].to(dtype)
            if a is not None:
                value = value @ a[slot, 0].to(dtype).T
            result[row] = value @ b[slot, 0].to(dtype).T * scale
    return result


@pytest.fixture
def make_env(monkeypatch):
    def create(*, version="0.23.0", hardware=True, extension=True):
        for name in tuple(sys.modules):
            if name.startswith(("netrsn_turbo", "vllm", "adn_custom_ops", "_turbo_test_alias")):
                monkeypatch.delitem(sys.modules, name, raising=False)

        def module(name, path=None):
            value = ModuleType(name)
            value.__path__ = [] if path is None else [str(path)]
            monkeypatch.setitem(sys.modules, name, value)
            if "." in name:
                parent, _, child = name.rpartition(".")
                setattr(sys.modules[parent], child, value)
            return value

        calls, registrations, custom_ops, native_calls = [], [], [], []
        vllm = module("vllm")
        vllm.__version__ = version
        module("vllm.logger").logger = logging.getLogger("turbo_test")
        module("vllm.lora")
        module("vllm.lora.layers")
        column = module("vllm.lora.layers.column_parallel_linear")
        base = module("vllm.lora.layers.base_linear")

        def original_mcp(x, bias, layer):
            return "original", x, bias, layer

        class Base:
            def _apply_sync(self, x, bias):
                return "sync", x, bias, self

        base.BaseLinearLayerWithLoRA = Base
        column._mcp_apply = original_mcp
        module("vllm.utils")
        torch_utils = module("vllm.utils.torch_utils")

        def register_custom_op(op_name, op_func, **kwargs):
            custom_ops.append(SimpleNamespace(name=op_name, function=op_func, **kwargs))
            monkeypatch.setattr(torch.ops.vllm, op_name, op_func, raising=False)

        torch_utils.direct_register_custom_op = register_custom_op
        module("vllm_ascend")
        module("vllm_ascend.lora")
        punica = module("vllm_ascend.lora.punica_npu")
        utils = module("vllm_ascend.utils")
        utils.is_310p = lambda: hardware

        def adapt_patch(*args, **kwargs):
            native_calls.append((args, kwargs))
            if env.native_override is not None:
                env.native_override()
            return "native-result"

        utils.adapt_patch = adapt_patch

        class Punica:
            def __init__(self, tokens=8, groups=2, device="cpu", **kwargs):
                self._token_lora_indices = torch.full((tokens,), 77, dtype=torch.long)
                self._sampler_indices = torch.full((tokens,), 77, dtype=torch.long)
                self._seq_lengths = torch.full((groups,), 77, dtype=torch.long)
                self._lora_indices_per_batch = torch.full((groups,), 77, dtype=torch.long)
                self.indices_len = [0, 0, 0, 0]
                self.batch_size = 0
                self.is_prefill = True
                self.no_lora = True

            def _get_token_lora_indices(self, x):
                return self._token_lora_indices[: x.shape[0]]

            def _update_base_metadata(self, mapping):
                self._token_lora_indices[: len(mapping.tokens)] = torch.tensor(mapping.tokens)
                self._sampler_indices[: len(mapping.sampler)] = torch.tensor(mapping.sampler)
                self.indices_len[:2] = [len(mapping.tokens), len(mapping.sampler)]
                return "base-result"

            def _update_prefill_metadata(self, tokens):
                order, counts = torch.unique_consecutive(tokens, return_counts=True)
                self.batch_size = order.numel()
                self._seq_lengths[: self.batch_size] = counts
                self._lora_indices_per_batch[: self.batch_size] = order
                return "prefill-result"

            def _shrink_prefill(self, y, x, w, scale):
                raise AssertionError("unpatched prefill")

            _shrink_decode = _shrink_prefill

            def _expand_prefill(self, y, x, w, add_inputs):
                raise AssertionError("unpatched prefill")

            _expand_decode = _expand_prefill

            def _expand_slice_prefill(self, y, x, w, offset, width, add_inputs):
                raise AssertionError("unpatched prefill")

            _expand_slice_decode = _expand_slice_prefill

            def add_lora_linear(self, y, x, a, b, scale, widths, *, buffer=None, **kwargs):
                if buffer is None:
                    buffer = tuple(torch.zeros(x.shape[0], weight.shape[-1]) for weight in b)
                offset = 0
                for shrink, wa, wb, width in zip(buffer, a, b, widths):
                    self._shrink_prefill(shrink, x, wa, scale)
                    self._expand_slice_prefill(y, shrink, wb, offset, width, True)
                    offset += width

            def add_lora_logits(self, *args, **kwargs):
                raise AssertionError("unpatched logits")

        punica.PunicaWrapperNPU = Punica

        def adn_op(
            y,
            *,
            x,
            weight_a,
            weight_b,
            indices=None,
            lora_order=None,
            lora_batch_num=None,
            layer_idx=0,
            scale=1.0,
            y_offset=0,
            y_slice_size=1,
        ):
            calls.append(SimpleNamespace(**locals()))
            assert layer_idx == 0 and y_slice_size == 1
            if indices is None:
                assert lora_order.dtype == lora_batch_num.dtype == torch.int32
                assert lora_order.numel() < min(y.shape[0], x.shape[0])
                slots = []
                for slot, count in zip(lora_order.tolist(), lora_batch_num.tolist()):
                    slots.extend([slot] * count)
                slots.extend([-1] * (x.shape[0] - len(slots)))
            else:
                assert indices.dtype == torch.int32
                slots = indices.tolist()
            delta = _project(x, weight_a, weight_b, slots, scale, y.dtype)
            y[:, y_offset : y_offset + delta.shape[1]].add_(delta)

        if extension:
            module("adn_custom_ops").adn_batch_gather_matmul_ = adn_op
        else:
            monkeypatch.setitem(sys.modules, "adn_custom_ops", None)

        module("netrsn_turbo", ROOT / "netrsn_turbo")
        module("netrsn_turbo.turbo", ROOT / "netrsn_turbo/turbo")
        module("netrsn_turbo.turbo.version_0230", ROOT / "netrsn_turbo/turbo/version_0230")
        module("netrsn_turbo.turbo_manager", ROOT / "netrsn_turbo/turbo_manager")
        turbo_utils = module("netrsn_turbo.turbo_manager.turbo_utils")

        def resolve(path):
            parts = path.split(".")
            for end in range(len(parts), 0, -1):
                value = sys.modules.get(".".join(parts[:end]))
                if value is not None:
                    for name in parts[end:-1]:
                        value = getattr(value, name)
                    return value, parts[-1]
            raise AssertionError(path)

        class Manager:
            patches_info: ClassVar[dict[str, object]] = {}
            apply_count = 0

            @classmethod
            def register_patch(cls, path, replacement):
                registrations.append((path, replacement))
                cls.patches_info[path] = replacement

            @classmethod
            def apply_patches(cls):
                cls.apply_count += 1
                for path, replacement in cls.patches_info.items():
                    target, attr = resolve(path)
                    original = getattr(target, attr)
                    setattr(target, attr, replacement)
                    # Simulate Patch's documented sys.modules alias propagation.
                    for loaded in tuple(sys.modules.values()):
                        if not isinstance(loaded, ModuleType):
                            continue
                        for name, value in tuple(vars(loaded).items()):
                            if value is original:
                                setattr(loaded, name, replacement)

        turbo_utils.TurboManager = Manager
        alias = module("_turbo_test_alias")
        alias.mcp = original_mcp
        alias.adapt = adapt_patch
        env = SimpleNamespace(
            cls=Punica,
            column=column,
            utils=utils,
            manager=Manager,
            alias=alias,
            registrations=registrations,
            custom_ops=custom_ops,
            calls=calls,
            native_calls=native_calls,
            native_override=None,
            original_mcp=original_mcp,
            original_adapt=adapt_patch,
        )
        env.load = lambda: importlib.import_module(PACKAGE + ".bootstrap")
        env.entry = lambda: importlib.import_module(ENTRY)
        env.resolve = resolve
        return env

    return create


@pytest.mark.parametrize("version,hardware", [("0.22.0", True), ("0.24.0", True), ("0.23.0", False)])
def test_gate_before_payload_import_or_registration(make_env, version, hardware):
    env = make_env(version=version, hardware=hardware, extension=False)
    assert env.load().apply_lora_patch() is False
    assert not env.registrations and not env.custom_ops
    assert PACKAGE + ".patches" not in sys.modules
    assert env.column._mcp_apply is env.original_mcp


def test_entry_installs_without_adapter_package_environment(make_env, monkeypatch):
    monkeypatch.delenv("ADAPTATION_PKG_ID", raising=False)
    env = make_env(version="0.23.0+vendor")
    env.entry()
    state = env.cls._netrsn_turbo_lora_310p_state
    assert state.owner == OWNER
    assert env.alias.mcp is env.column._mcp_apply
    assert env.alias.adapt is env.utils.adapt_patch
    paths = {path for path, _ in env.registrations}
    assert paths == {path for path, _ in state.replacements}
    assert f"{PUNICA}._shrink_prefill" in paths
    assert "vllm_ascend.utils.adapt_patch" in paths
    assert "vllm.lora.layers.column_parallel_linear._mcp_apply" in paths
    assert len(env.custom_ops) == (1 if OWNER == "adn" else 0)


def test_repeated_apply_does_not_stack_wrappers_or_register_again(make_env):
    env = make_env()
    bootstrap = env.load()
    assert bootstrap.apply_lora_patch()
    originals = tuple(env.registrations)
    for _ in range(3):
        assert bootstrap.apply_lora_patch()
    assert tuple(env.registrations) == originals
    assert env.manager.apply_count == 1
    for path, replacement in originals:
        parent, attr = env.resolve(path)
        assert getattr(parent, attr) is replacement


def test_native_adapt_runs_then_restores_stable_replacements(make_env):
    env = make_env()
    env.entry()
    replacements = tuple(env.registrations)

    def overwrite():
        env.cls._shrink_prefill = lambda *args: None
        env.column._mcp_apply = env.original_mcp

    env.native_override = overwrite
    for _ in range(2):
        assert env.alias.adapt(is_global_patch=False) == "native-result"
        for path, replacement in replacements:
            parent, attr = env.resolve(path)
            assert getattr(parent, attr) is replacement
    assert len(env.native_calls) == 2
    assert tuple(env.registrations) == replacements


def test_fresh_process_module_state_installs_again(make_env):
    first = make_env()
    first.entry()
    state = first.cls._netrsn_turbo_lora_310p_state
    fresh = make_env()
    fresh.entry()
    assert fresh.cls is not first.cls
    assert fresh.cls._netrsn_turbo_lora_310p_state is not state
    assert fresh.registrations


def test_conflicting_owner_fails_before_payload_or_custom_operator(make_env):
    env = make_env(extension=False)
    env.cls._netrsn_turbo_lora_310p_state = SimpleNamespace(owner="adn" if OWNER != "adn" else "shared_gemm")
    with pytest.raises(RuntimeError, match="already selected"):
        env.load().apply_lora_patch()
    assert not env.registrations and not env.custom_ops
    assert PACKAGE + ".patches" not in sys.modules


@pytest.mark.parametrize("flag", ["_ascend_lora_prefill_310_patch", "_ascend_lora_adn_310_patch"])
def test_previous_native_patch_is_rejected(make_env, flag):
    env = make_env()
    setattr(env.cls, flag, True)
    with pytest.raises(RuntimeError, match="previous handwritten"):
        env.load().apply_lora_patch()
    assert not env.registrations and not env.custom_ops


@pytest.mark.parametrize(
    "tp,sharded,expected",
    [(1, False, "original"), (4, False, "sync"), (4, True, "original")],
)
def test_tp_ordinary_lora_skips_gather_but_sharded_preserved(make_env, tp, sharded, expected):
    env = make_env()
    env.entry()
    layer = SimpleNamespace(tp_size=tp, lora_config=SimpleNamespace(fully_sharded_loras=sharded))
    assert env.column._mcp_apply("x", "bias", layer)[0] == expected


def _wrapper(env, indices, groups=2):
    wrapper = env.cls(len(indices), groups)
    wrapper._update_base_metadata(SimpleNamespace(tokens=indices, sampler=indices))
    wrapper._update_prefill_metadata(torch.tensor(indices, dtype=torch.long))
    return wrapper


@pytest.mark.parametrize("slots", [1, 2])
def test_linear_real_torch_math_base_rows_scale_and_slices(make_env, slots):
    env = make_env()
    env.entry()
    indices = [0, 0, -1, -1] if slots == 1 else [0, 0, 1, 1]
    wrapper = _wrapper(env, indices)
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 9
    a = (torch.arange(slots * 8).reshape(slots, 1, 2, 4).float() / 13,)
    b = (torch.arange(slots * 6).reshape(slots, 1, 3, 2).float() / 11,)
    y = torch.full((4, 5), 7.0)
    expected = y.clone()
    expected[:, :3] += _project(x, a[0], b[0], indices, 0.25, y.dtype)
    wrapper.add_lora_linear(y, x, a, b, 0.25, (3,))
    torch.testing.assert_close(y, expected)
    if OWNER == "adn":
        assert env.calls[0].y.data_ptr() == y.data_ptr()
        assert env.calls[0].y_offset == 0


def test_dynamic_weight_updates_are_read_without_rebuilding_patch(make_env):
    env = make_env()
    env.entry()
    wrapper = _wrapper(env, [0, 0, 0, 0], groups=1)
    x = torch.ones(4, 3)
    a, b = torch.ones(1, 1, 2, 3), torch.ones(1, 1, 5, 2)

    def run():
        y = torch.zeros(4, 5)
        wrapper.add_lora_linear(y, x, (a,), (b,), 1.0, (5,))
        return y

    first = run()
    b.mul_(3)
    torch.testing.assert_close(run(), first * 3)
    a.zero_()
    torch.testing.assert_close(run(), torch.zeros_like(first))


def test_metadata_clears_stale_token_padding(make_env):
    env = make_env()
    env.entry()
    wrapper = env.cls(8, 4)
    assert torch.all(wrapper._token_lora_indices == -1)
    wrapper._update_base_metadata(SimpleNamespace(tokens=[0] * 8, sampler=[0] * 8))
    assert wrapper._update_base_metadata(SimpleNamespace(tokens=[0, -1], sampler=[0])) == "base-result"
    assert wrapper._token_lora_indices.tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
    if OWNER == "adn":
        assert wrapper._sampler_indices.tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
        wrapper._update_prefill_metadata(torch.tensor([0, 1, -1]))
        assert wrapper._update_prefill_metadata(torch.tensor([0, 0])) == "prefill-result"
        assert wrapper._seq_lengths.tolist() == [2, 0, 0, 0]
        assert wrapper._lora_indices_per_batch.tolist() == [0, -1, -1, -1]


@pytest.mark.skipif(OWNER != "shared_gemm", reason="GEMM-specific precision and mask")
def test_gemm_fp32_shrink_masks_base_and_nan_padding(make_env):
    env = make_env()
    env.entry()
    wrapper = _wrapper(env, [0, 0, -1, -1])
    x = torch.ones(4, 3, dtype=torch.float16)
    x[2:] = float("nan")
    a = torch.ones(1, 1, 2, 3, dtype=torch.float16)
    y = torch.full((4, 4), 11.0, dtype=torch.float32)
    wrapper._shrink_prefill(y, x, a, 0.5)
    torch.testing.assert_close(y[:, :2], torch.tensor([[1.5, 1.5], [1.5, 1.5], [0, 0], [0, 0]]))
    assert y.dtype == torch.float32
    assert torch.all(y[:, 2:] == 11)


@pytest.mark.skipif(OWNER != "adn", reason="ADN shape ABI")
@pytest.mark.parametrize("rows,grouped", [(1, False), (2, False), (3, True), (8, True)])
def test_adn_runtime_shape_route_and_direct_output(make_env, rows, grouped):
    env = make_env()
    env.entry()
    wrapper = _wrapper(env, [0] * rows, groups=2)
    y = torch.ones(rows, 4)
    x = torch.ones(rows, 3)
    a, b = torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 2)
    wrapper.add_lora_linear(y, x, (a, a), (b, b), 0.5, (2, 2))
    assert [call.y_offset for call in env.calls] == [0, 2]
    assert all(call.y.data_ptr() == y.data_ptr() for call in env.calls)
    assert all((call.indices is None) == grouped for call in env.calls)
    torch.testing.assert_close(y, torch.full_like(y, 4.0))
    registration = env.custom_ops[0]
    assert registration.mutates_args == ["y"]
    assert registration.dispatch_key == "PrivateUse1"


@pytest.mark.skipif(OWNER != "adn", reason="ADN separate split/logits ABI")
def test_adn_split_overwrite_and_sampler_indices(make_env):
    env = make_env()
    env.entry()
    wrapper = _wrapper(env, [0, 0, -1, -1])
    wrapper._sampler_indices[:] = torch.tensor([-1, 0, -1, 0])
    x, weight = torch.ones(4, 3), torch.ones(1, 1, 2, 3)
    shrink = torch.full((4, 2), 99.0)
    wrapper._shrink_decode(shrink, x, weight, 0.5)
    torch.testing.assert_close(shrink, torch.tensor([[1.5, 1.5], [1.5, 1.5], [0, 0], [0, 0]]))
    y = torch.full((4, 2), 7.0)
    a, b = torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 2)
    wrapper.add_lora_logits(y, x, a, b, 0.5, add_inputs=False)
    torch.testing.assert_close(y, torch.tensor([[0.0, 0.0], [3.0, 3.0], [0.0, 0.0], [3.0, 3.0]]))
    assert env.calls[-1].indices.tolist() == [-1, 0, -1, 0]


def test_projection_registrations_are_direct_module_functions(make_env):
    env = make_env()
    env.entry()
    payload = importlib.import_module(PACKAGE + ".patches")
    registered = dict(env.registrations)
    for name in (
        "add_lora_linear",
        "add_lora_logits",
        "_shrink_prefill",
        "_shrink_decode",
        "_expand_prefill",
        "_expand_decode",
        "_expand_slice_prefill",
        "_expand_slice_decode",
    ):
        replacement = registered[f"{PUNICA}.{name}"]
        assert replacement is getattr(payload, name)
        assert replacement is getattr(env.cls, name)
        assert not hasattr(replacement, "__wrapped__")
    assert env.custom_ops[0].function is payload._adn_dispatch


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_direct_shrink_overwrites_rank_only_and_selects_explicit_phase(make_env, phase):
    env = make_env()
    env.entry()
    slots = [0, 0, -1, -1]
    wrapper = _wrapper(env, slots)
    # A directly selected split method owns its phase. It must not pick up a
    # stale wrapper flag or accidentally share the other phase's closure.
    wrapper.is_prefill = phase != "prefill"
    x = torch.arange(12).reshape(2, 2, 3).float() / 7
    weight = torch.arange(6).reshape(1, 1, 2, 3).float() / 11
    y = torch.full((2, 2, 4), 19.0)
    expected = y.clone().view(4, 4)
    expected[:, :2] = _project(x.reshape(4, 3), None, weight, slots, 0.375, y.dtype)

    getattr(wrapper, f"_shrink_{phase}")(y, x, weight, 0.375)

    torch.testing.assert_close(y.view(4, 4), expected)
    assert len(env.calls) == 1
    call = env.calls[0]
    assert call.weight_a is None and call.weight_b is weight
    assert call.y.data_ptr() == y.data_ptr()
    assert call.y_offset == 0 and call.scale == 0.375
    assert (call.indices is None) == (phase == "prefill")
    if phase == "decode":
        assert call.indices.tolist() == slots


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("sliced", [False, True], ids=["expand", "expand_slice"])
@pytest.mark.parametrize("add_inputs", [False, True], ids=["overwrite", "accumulate"])
def test_direct_expand_preserves_offsets_and_explicit_phase(make_env, phase, sliced, add_inputs):
    env = make_env()
    env.entry()
    slots = [0, 0, -1, -1]
    wrapper = _wrapper(env, slots)
    wrapper.is_prefill = phase != "prefill"
    x = torch.arange(12).reshape(2, 2, 3).float() / 7
    weight = torch.arange(9).reshape(1, 1, 3, 3).float() / 11
    y = torch.full((2, 2, 7), 13.0)
    offset = 2 if sliced else 0
    delta = _project(x.reshape(4, 3), None, weight, slots, 1.0, y.dtype)
    expected = y.clone().view(4, 7)
    if add_inputs:
        expected[:, offset : offset + 3] += delta
    else:
        expected[:, offset : offset + 3] = delta

    if sliced:
        getattr(wrapper, f"_expand_slice_{phase}")(y, x, weight, offset, 3, add_inputs)
    else:
        getattr(wrapper, f"_expand_{phase}")(y, x, weight, add_inputs)

    torch.testing.assert_close(y.view(4, 7), expected)
    assert len(env.calls) == 1
    call = env.calls[0]
    assert call.weight_a is None and call.weight_b is weight
    assert call.y.data_ptr() == y.data_ptr()
    assert call.y_offset == offset and call.scale == 1.0
    assert (call.indices is None) == (phase == "prefill")
    if phase == "decode":
        assert call.indices.tolist() == slots
