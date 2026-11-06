# SPDX-License-Identifier: Apache-2.0
"""Exercise the real DS source rewrite together with the AOT worker wrappers."""

import importlib.util
import linecache
import runpy
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ._turbo_contract import apply_patches

ROOT = Path(__file__).resolve().parents[1]
DS_WORKER = (
    ROOT.parent / "netrsnturbo_patch_v2/netrsn_turbo/turbo/version_0300/ds_lora/vllm_ascend/worker/model_runner_v1.py"
)
pytestmark = pytest.mark.skipif(not DS_WORKER.is_file(), reason="DS LoRA v2 delivery is required for composition tests")
SOURCE = """\
import torch

class Runner:
    @torch.inference_mode()
    def _dummy_run(self, num_tokens, is_profile=False, num_active_loras=0,
                   is_graph_capturing=False):
        self.calls.append(("descriptor", num_tokens, is_profile,
                           num_active_loras, is_graph_capturing))
        self.calls.append(("inference", self.inference_depth))
        with self.maybe_dummy_run_with_lora(
            self.lora_config, num_tokens,
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ):
            if self.fail_dummy:
                raise RuntimeError("dummy failed")
            self.calls.append(("forward", self.active_count))
        return num_tokens
"""


@pytest.fixture
def composed_worker(monkeypatch):
    calls = []

    def inference_mode():
        def decorate(function):
            @wraps(function)
            def inference_wrapper(self, *args, **kwargs):
                self.inference_depth += 1
                try:
                    return function(self, *args, **kwargs)
                finally:
                    self.inference_depth -= 1

            return inference_wrapper

        return decorate

    torch = ModuleType("torch")
    torch.inference_mode = inference_mode
    torch.npu = SimpleNamespace(synchronize=lambda: calls.append(("sync",)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    filename = "<aot-ds-worker-composition>"
    monkeypatch.setitem(linecache.cache, filename, (len(SOURCE), None, SOURCE.splitlines(keepends=True), filename))
    fixture_module = ModuleType("_aot_ds_worker_fixture")
    monkeypatch.setitem(sys.modules, fixture_module.__name__, fixture_module)
    exec(compile(SOURCE, filename, "exec", dont_inherit=True), vars(fixture_module))
    runner_cls = fixture_module.Runner

    def initialize(self, *, enabled=True, max_tokens=8):
        self.calls = calls
        self.lora_config = SimpleNamespace(max_loras=64)
        self.vllm_config = SimpleNamespace(enabled=enabled)
        self.max_num_tokens = max_tokens
        self.fail_dummy = False
        self.inference_depth = 0
        self.active_count = None

    @contextmanager
    def lora_context(self, lora_config, num_scheduled_tokens, num_active_loras=0):
        previous = self.active_count
        self.active_count = num_active_loras
        calls.append(("mapping", num_active_loras))
        try:
            yield
        finally:
            self.active_count = previous

    def capture(self):
        calls.append(("capture",))
        return 17

    runner_cls.__init__ = initialize
    runner_cls.maybe_dummy_run_with_lora = lora_context
    runner_cls.capture_model = capture

    wrapper_name = "netrsn_turbo.turbo.version_v0300.vllm.compilation.wrapper"
    wrapper_module = ModuleType(wrapper_name)
    wrapper_module.enabled = lambda config: config.enabled
    monkeypatch.setitem(sys.modules, wrapper_name, wrapper_module)
    worker_name = "netrsn_turbo.turbo.version_v0300.vllm_ascend.worker.model_runner_v1"
    worker_path = ROOT / "netrsn_turbo/turbo/version_v0300/vllm_ascend/worker/model_runner_v1.py"
    spec = importlib.util.spec_from_file_location(worker_name, worker_path)
    helper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, worker_name, helper)
    spec.loader.exec_module(helper)

    ds_rewrite = runpy.run_path(str(DS_WORKER))["_dummy_run"]
    ds_dummy_run = ds_rewrite(runner_cls._dummy_run)
    replacements = {
        "_dummy_run": helper.wrap_dummy_run(ds_dummy_run),
        "maybe_dummy_run_with_lora": helper.wrap_maybe_dummy_run_with_lora(lora_context),
        "capture_model": helper.wrap_capture_model(capture),
    }
    source_module = ModuleType("vllm_ascend.worker.model_runner_v1")
    source_module.NPUModelRunner = runner_cls
    monkeypatch.setitem(sys.modules, source_module.__name__, source_module)
    aliases = ModuleType("_aot_ds_worker_aliases")
    aliases.dummy = runner_cls._dummy_run
    aliases.context = lora_context
    aliases.capture = capture
    monkeypatch.setitem(sys.modules, aliases.__name__, aliases)
    apply_patches(
        monkeypatch,
        [(f"{source_module.__name__}.NPUModelRunner.{name}", value) for name, value in replacements.items()],
    )
    return SimpleNamespace(runner=runner_cls, calls=calls, helper=helper, aliases=aliases)


@pytest.mark.parametrize("explicit_count", [None, 0, 2])
def test_initial_profile_includes_lora_in_descriptor_and_mapping(composed_worker, explicit_count):
    env = composed_worker
    runner = env.runner()
    kwargs = {} if explicit_count is None else {"num_active_loras": explicit_count}
    assert runner._dummy_run(8, is_profile=True, **kwargs) == 8
    assert ("descriptor", 8, True, 64, False) in env.calls
    assert ("mapping", 64) in env.calls
    assert ("forward", 64) in env.calls
    assert ("inference", 1) in env.calls
    assert runner.inference_depth == 0
    assert runner.active_count is None
    assert not hasattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR)


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("capturing", [False, True])
def test_regular_warmup_and_capture_agree_on_lora_count(composed_worker, count, capturing):
    env = composed_worker
    env.runner()._dummy_run(8, num_active_loras=count, is_graph_capturing=capturing)
    assert ("descriptor", 8, False, count, capturing) in env.calls
    assert ("mapping", count) in env.calls
    assert ("forward", count) in env.calls


def test_default_nonprofile_call_uses_zero_loras(composed_worker):
    env = composed_worker
    env.runner()._dummy_run(8)
    assert ("descriptor", 8, False, 0, False) in env.calls
    assert ("mapping", 0) in env.calls


def test_positional_profile_and_lora_count_are_bound_correctly(composed_worker):
    env = composed_worker
    runner = env.runner()
    runner._dummy_run(8, False, 3, True)
    assert ("descriptor", 8, False, 3, True) in env.calls
    assert ("mapping", 3) in env.calls
    env.calls.clear()
    runner._dummy_run(8, True, 0)
    assert ("descriptor", 8, True, 64, False) in env.calls
    assert ("mapping", 64) in env.calls


@pytest.mark.parametrize("max_tokens, sizes", [(8, [8, 1]), (1, [1])])
def test_capture_prewarms_only_zero_lora_base_variants(composed_worker, max_tokens, sizes):
    env = composed_worker
    runner = env.runner(max_tokens=max_tokens)
    assert runner.capture_model() == 17
    expected = []
    for size in sizes:
        expected.extend([("descriptor", size, True, 0, False), ("inference", 1), ("mapping", 0), ("forward", 0)])
    assert env.calls == [*expected, ("sync",), ("capture",)]
    assert not hasattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR)


@pytest.mark.parametrize("profile, count", [(True, 0), (False, 2)])
def test_disabled_aot_keeps_ds_rewritten_behavior(composed_worker, profile, count):
    env = composed_worker
    runner = env.runner(enabled=False)
    runner._dummy_run(8, is_profile=profile, num_active_loras=count)
    assert ("descriptor", 8, profile, count, False) in env.calls
    assert ("mapping", count) in env.calls
    env.calls.clear()
    assert runner.capture_model() == 17
    assert env.calls == [("capture",)]


@pytest.mark.parametrize("previous", [None, 0, 3])
def test_failed_dummy_restores_count_override_and_inference_state(composed_worker, previous):
    env = composed_worker
    runner = env.runner()
    runner.fail_dummy = True
    if previous is not None:
        setattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR, previous)
    with pytest.raises(RuntimeError, match="dummy failed"):
        runner._dummy_run(8, is_profile=True)
    count = 64 if previous is None else previous
    assert ("descriptor", 8, True, count, False) in env.calls
    assert ("mapping", count) in env.calls
    assert runner.inference_depth == 0
    assert runner.active_count is None
    if previous is None:
        assert not hasattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR)
    else:
        assert getattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR) == previous


def test_failed_base_prewarm_prevents_capture_and_restores_override(composed_worker):
    env = composed_worker
    runner = env.runner()
    runner.fail_dummy = True
    with pytest.raises(RuntimeError, match="dummy failed"):
        runner.capture_model()
    assert ("mapping", 0) in env.calls
    assert ("capture",) not in env.calls
    assert ("sync",) not in env.calls
    assert not hasattr(runner, env.helper._DUMMY_LORA_COUNT_ATTR)


def test_manager_alias_propagation_keeps_both_patches_callable(composed_worker):
    env = composed_worker
    assert env.aliases.dummy is env.runner._dummy_run
    assert env.aliases.context is env.runner.maybe_dummy_run_with_lora
    assert env.aliases.capture is env.runner.capture_model
    runner = env.runner()
    assert env.aliases.dummy(runner, 8, is_profile=True) == 8
    assert ("mapping", 64) in env.calls
    env.calls.clear()
    assert env.aliases.dummy(runner, 8, num_active_loras=2) == 8
    assert ("mapping", 2) in env.calls
