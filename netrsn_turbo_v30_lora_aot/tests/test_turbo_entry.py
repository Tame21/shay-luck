# SPDX-License-Identifier: Apache-2.0
"""Exercise the entry and payload with the documented TurboManager contract."""

import importlib
import logging
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from ._turbo_contract import apply_patches

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "netrsn_turbo.turbo_manager.version_v0300.turbo_lora_aot"
COMBINED_ENTRY = "netrsn_turbo.turbo_manager.version_v0300.turbo_ds_lora_aot"


@pytest.fixture
def make_env(monkeypatch):
    def create(*, version="0.30.0", hardware=True):
        for name in tuple(sys.modules):
            if name.startswith(("netrsn_turbo", "vllm", "_aot_entry_alias")):
                monkeypatch.delitem(sys.modules, name, raising=False)

        def module(name, **attrs):
            obj = ModuleType(name)
            obj.__path__ = []
            vars(obj).update(attrs)
            monkeypatch.setitem(sys.modules, name, obj)
            if "." in name:
                parent, _, child = name.rpartition(".")
                if parent not in sys.modules:
                    module(parent)
                monkeypatch.setattr(sys.modules[parent], child, obj, raising=False)
            return obj

        for name in ("netrsn_turbo", "netrsn_turbo.turbo", "netrsn_turbo.turbo_manager"):
            module(name).__path__ = [str(ROOT / name.replace(".", "/"))]

        calls = []
        config = SimpleNamespace(model_config=None, lora_config=None)

        class Wrapper:
            def __init__(self, compile_prefix="", is_encoder=False):
                calls.append(("init", compile_prefix, is_encoder))

        class Runner:
            def _dummy_run(self, num_tokens, is_profile=False, num_active_loras=0):
                return num_tokens

            @contextmanager
            def maybe_dummy_run_with_lora(self, *args, num_active_loras=0):
                yield

            def capture_model(self):
                return 17

        def load(model, path):
            calls.append(("load", path))
            return path

        def adapt_patch(is_global_patch=False):
            calls.append(("adapt", is_global_patch))
            return "native-result"

        module("vllm", __version__=version)
        module("vllm.envs", VLLM_USE_AOT_COMPILE=True, VLLM_USE_BYTECODE_HOOK=True)
        module("vllm.compilation.wrapper", TorchCompileWithNoGuardsWrapper=Wrapper)
        decorators = module("vllm.compilation.decorators", _try_load_aot_compiled_fn=load)
        module("vllm.compilation.backends", model_tag="backbone")
        module(
            "vllm.config",
            CompilationMode=SimpleNamespace(VLLM_COMPILE=3),
            CUDAGraphMode=SimpleNamespace(NONE=0),
            get_current_vllm_config=lambda: config,
            set_current_vllm_config=lambda _: nullcontext(),
        )
        module("vllm.forward_context", get_forward_context=lambda: None, is_forward_context_available=lambda: False)
        module("vllm.logger", init_logger=logging.getLogger)
        module("vllm_ascend.worker.model_runner_v1", NPUModelRunner=Runner)
        utils = module("vllm_ascend.utils", adapt_patch=adapt_patch)
        module(
            "vllm_ascend.device.hardware_profile",
            HardwareCapability=SimpleNamespace(STANDARD_WORKER_PATCHES="standard"),
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: hardware),
        )
        module("torch", npu=SimpleNamespace(synchronize=lambda: None))
        alias = module("_aot_entry_alias", init=Wrapper.__init__, load=load, adapt=adapt_patch)

        class Manager:
            patches_info: ClassVar[dict[str, object]] = {}
            registrations: ClassVar[list] = []
            apply_count = 0
            fail_register = False
            fail_apply = False

            @classmethod
            def register_patch(cls, path, replacement):
                if cls.fail_register:
                    raise RuntimeError("registration failed")
                cls.registrations.append((path, replacement))
                cls.patches_info[path] = replacement

            @classmethod
            def apply_patches(cls):
                cls.apply_count += 1
                if cls.fail_apply:
                    raise RuntimeError("application failed")
                apply_patches(monkeypatch, cls.patches_info.items())

        module("netrsn_turbo.turbo_manager.turbo_utils", TurboManager=Manager)
        env = SimpleNamespace(
            wrapper=Wrapper,
            runner=Runner,
            decorators=decorators,
            utils=utils,
            alias=alias,
            manager=Manager,
            calls=calls,
            load_entry=lambda: importlib.import_module(ENTRY),
        )
        return env

    return create


@pytest.mark.parametrize("version,hardware", [("0.25.1", True), ("0.30.1", True), ("0.31.0", True), ("0.30.0", False)])
def test_runtime_selection_is_delegated_to_outer_router(make_env, monkeypatch, version, hardware):
    env = make_env(version=version, hardware=hardware)
    monkeypatch.delitem(sys.modules, "vllm_ascend.device.hardware_profile")
    env.load_entry()
    assert len(env.manager.registrations) == 5
    env.wrapper(compile_prefix="outer-selected")
    assert env.calls[-1] == ("init", "outer-selected", False)


@pytest.mark.parametrize("version", ["0.30.0", "0.30.0+local", "0.30.0rc1"])
def test_entry_registers_all_targets_without_adapter_package_env(make_env, monkeypatch, version):
    monkeypatch.delenv("ADAPTATION_PKG_ID", raising=False)
    env = make_env(version=version)
    env.load_entry()
    assert len(env.manager.registrations) == 5
    assert env.manager.apply_count == 1
    assert env.alias.init is env.wrapper.__init__
    assert env.alias.load is env.decorators._try_load_aot_compiled_fn
    assert env.alias.adapt is env.utils.adapt_patch
    env.wrapper(compile_prefix="test")
    assert env.decorators._try_load_aot_compiled_fn(SimpleNamespace(), "cache") == "cache"
    assert env.utils.adapt_patch(is_global_patch=True) == "native-result"
    assert env.calls == [("init", "test", False), ("load", "cache"), ("adapt", True)]
    assert env.manager.apply_count == 1


@pytest.mark.parametrize("failure", ["registration", "application"])
def test_manager_failure_propagates(make_env, failure):
    env = make_env()
    env.manager.fail_register = failure == "registration"
    env.manager.fail_apply = failure == "application"
    with pytest.raises(RuntimeError, match="registration failed|application failed"):
        env.load_entry()


@pytest.mark.parametrize("adapter_package", [None, "adapter-package"])
def test_combined_entry_delegates_once_without_loading_standalone_entries(make_env, monkeypatch, adapter_package):
    env = make_env()
    if adapter_package is None:
        monkeypatch.delenv("ADAPTATION_PKG_ID", raising=False)
    else:
        monkeypatch.setenv("ADAPTATION_PKG_ID", adapter_package)
    calls = []
    ds = ModuleType("netrsn_turbo.turbo.version_v0300.installer")
    ds.apply_lora_patch = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, ds.__name__, ds)

    importlib.import_module(COMBINED_ENTRY)

    assert calls == [{"with_aot": True}]
    assert env.manager.registrations == []
    assert env.manager.apply_count == 0
    assert ENTRY not in sys.modules
    assert "netrsn_turbo.turbo_manager.version_0300.turbo_ds_lora" not in sys.modules


def test_combined_entry_propagates_installation_failure(make_env, monkeypatch):
    make_env()
    ds = ModuleType("netrsn_turbo.turbo.version_v0300.installer")

    def fail(**kwargs):
        raise RuntimeError("combined installation failed")

    ds.apply_lora_patch = fail
    monkeypatch.setitem(sys.modules, ds.__name__, ds)
    with pytest.raises(RuntimeError, match="combined installation failed"):
        importlib.import_module(COMBINED_ENTRY)
