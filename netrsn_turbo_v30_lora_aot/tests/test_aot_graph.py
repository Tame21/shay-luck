# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from ._turbo_contract import apply_patches, load_payloads


@pytest.fixture(params=["qwen3_5_text", "deepseek_v4"])
def graph_hooks(request, monkeypatch):
    calls = []

    class RunnerBase:
        @contextmanager
        def maybe_dummy_run_with_lora(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora=True,
            num_active_loras=0,
            mapping_type=None,
        ):
            calls.append(("mapping", num_active_loras))
            yield

    class Runner(RunnerBase):
        def __init__(self, enabled=True, max_tokens=8):
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type=request.param),
                    hf_config=SimpleNamespace(
                        model_type="qwen3_5" if request.param == "qwen3_5_text" else request.param
                    ),
                    model="local/model",
                    enforce_eager=not enabled,
                ),
                lora_config=SimpleNamespace(fully_sharded_loras=False, max_loras=4),
                use_v2_model_runner=False,
                compilation_config=SimpleNamespace(
                    mode=3, backend="inductor", cudagraph_mode=1, cudagraph_specialize_lora=True
                ),
            )
            self.lora_config = self.vllm_config.lora_config
            self.max_num_tokens = max_tokens
            self.fail_dummy = False

        def _dummy_run(self, num_tokens, is_profile=False, num_active_loras=0, is_graph_capturing=False):
            calls.append(("dummy", num_tokens, is_profile, num_active_loras))
            # Match the runner workaround that otherwise overrides descriptor state.
            with self.maybe_dummy_run_with_lora(None, None, None, num_active_loras=4):
                if self.fail_dummy:
                    raise RuntimeError("dummy failed")
            return num_tokens

        def capture_model(self):
            calls.append(("capture",))
            return 17

    runner_module = ModuleType("vllm_ascend.worker.model_runner_v1")
    runner_module.NPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    fake_torch = ModuleType("torch")
    fake_torch.npu = SimpleNamespace(synchronize=lambda: calls.append(("sync",)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    envs = stub("vllm.envs", VLLM_USE_AOT_COMPILE=True)
    decorators = stub("vllm.compilation.decorators", _try_load_aot_compiled_fn=lambda *_: None)
    backends = stub("vllm.compilation.backends", model_tag="backbone")
    compilation = stub("vllm.compilation", decorators=decorators, backends=backends)
    stub("vllm", envs=envs, compilation=compilation)

    class Wrapper:
        def __init__(self, compile_prefix="", is_encoder=False):
            pass

    stub("vllm.compilation.wrapper", TorchCompileWithNoGuardsWrapper=Wrapper)
    stub(
        "vllm.config",
        CompilationMode=SimpleNamespace(VLLM_COMPILE=3),
        CUDAGraphMode=SimpleNamespace(NONE=0),
        get_current_vllm_config=lambda: None,
        set_current_vllm_config=lambda _: nullcontext(),
    )
    stub("vllm.forward_context", get_forward_context=lambda: None, is_forward_context_available=lambda: False)
    stub("vllm.logger", init_logger=logging.getLogger)

    methods = (Runner._dummy_run, Runner.maybe_dummy_run_with_lora, Runner.capture_model)
    aliases = stub("_lora_aot_graph_aliases", dummy=methods[0], context=methods[1], capture=methods[2])
    helper = load_payloads(monkeypatch).runner
    assert methods == (Runner._dummy_run, Runner.maybe_dummy_run_with_lora, Runner.capture_model)
    prefix = "vllm_ascend.worker.model_runner_v1.NPUModelRunner"
    replacements = (
        (f"{prefix}._dummy_run", helper.wrap_dummy_run(Runner._dummy_run)),
        (
            f"{prefix}.maybe_dummy_run_with_lora",
            helper.wrap_maybe_dummy_run_with_lora(Runner.maybe_dummy_run_with_lora),
        ),
        (f"{prefix}.capture_model", helper.wrap_capture_model(Runner.capture_model)),
    )
    assert methods == (Runner._dummy_run, Runner.maybe_dummy_run_with_lora, Runner.capture_model)
    apply_patches(monkeypatch, replacements)
    helper.test_reapply = lambda: apply_patches(monkeypatch, replacements)
    helper.test_aliases = aliases
    return Runner, calls, helper


@pytest.mark.parametrize("kwargs", [{"is_profile": True}, {"is_profile": True, "num_active_loras": 0}])
def test_initial_profiling_keeps_dummy_adapters(graph_hooks, kwargs):
    runner_cls, calls, _ = graph_hooks
    runner = runner_cls()
    assert runner._dummy_run(8, **kwargs) == 8
    assert ("dummy", 8, True, 4) in calls
    assert ("mapping", 4) in calls


def test_non_profile_default_count_matches_base_descriptor(graph_hooks):
    runner_cls, calls, _ = graph_hooks
    assert runner_cls()._dummy_run(8) == 8
    assert calls == [("dummy", 8, False, 0), ("mapping", 0)]


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("capturing", [False, True])
def test_warmup_and_capture_use_the_descriptor_lora_count(graph_hooks, count, capturing):
    runner_cls, calls, helper = graph_hooks
    runner = runner_cls()
    runner._dummy_run(8, num_active_loras=count, is_graph_capturing=capturing)
    assert ("mapping", count) in calls
    assert not hasattr(runner, helper._DUMMY_LORA_COUNT_ATTR)


def test_positional_lora_count_is_preserved(graph_hooks):
    runner_cls, calls, _ = graph_hooks
    runner_cls()._dummy_run(8, False, 3)
    assert ("mapping", 3) in calls


@pytest.mark.parametrize("previous", [None, 1])
def test_failed_dummy_run_restores_the_previous_override(graph_hooks, previous):
    runner_cls, calls, helper = graph_hooks
    runner = runner_cls()
    runner.fail_dummy = True
    if previous is not None:
        setattr(runner, helper._DUMMY_LORA_COUNT_ATTR, previous)
    with pytest.raises(RuntimeError, match="dummy failed"):
        runner._dummy_run(8, num_active_loras=2)
    assert ("mapping", 2 if previous is None else previous) in calls
    if previous is None:
        assert not hasattr(runner, helper._DUMMY_LORA_COUNT_ATTR)
    else:
        assert getattr(runner, helper._DUMMY_LORA_COUNT_ATTR) == previous


@pytest.mark.parametrize("max_tokens, expected_sizes", [(8, [8, 1]), (1, [1])])
def test_base_variants_are_ready_before_capture(graph_hooks, max_tokens, expected_sizes):
    runner_cls, calls, helper = graph_hooks
    runner = runner_cls(max_tokens=max_tokens)
    assert runner.capture_model() == 17
    expected = []
    for size in expected_sizes:
        expected.extend([("dummy", size, True, 0), ("mapping", 0)])
    assert calls == [*expected, ("sync",), ("capture",)]
    assert not hasattr(runner, helper._DUMMY_LORA_COUNT_ATTR)


def test_failed_base_warmup_does_not_enter_capture_or_leak_state(graph_hooks):
    runner_cls, calls, helper = graph_hooks
    runner = runner_cls()
    runner.fail_dummy = True
    with pytest.raises(RuntimeError, match="dummy failed"):
        runner.capture_model()
    assert ("capture",) not in calls
    assert ("sync",) not in calls
    assert not hasattr(runner, helper._DUMMY_LORA_COUNT_ATTR)


def test_disabled_configuration_preserves_original_runner_behavior(graph_hooks):
    runner_cls, calls, helper = graph_hooks
    runner = runner_cls(enabled=False)
    runner._dummy_run(8, num_active_loras=0, is_graph_capturing=True)
    assert calls[-1] == ("mapping", 4)
    calls.clear()
    assert runner.capture_model() == 17
    assert calls == [("capture",)]
    assert not hasattr(runner, helper._DUMMY_LORA_COUNT_ATTR)


def test_installing_twice_does_not_wrap_the_runner_again(graph_hooks):
    runner_cls, _, helper = graph_hooks
    methods = (runner_cls._dummy_run, runner_cls.maybe_dummy_run_with_lora, runner_cls.capture_model)
    helper.test_reapply()
    assert methods == (runner_cls._dummy_run, runner_cls.maybe_dummy_run_with_lora, runner_cls.capture_model)


def test_graph_aliases_use_wrappers_without_losing_original_calls(graph_hooks):
    runner_cls, calls, helper = graph_hooks
    aliases = helper.test_aliases
    assert aliases.dummy is runner_cls._dummy_run
    assert aliases.context is runner_cls.maybe_dummy_run_with_lora
    assert aliases.capture is runner_cls.capture_model
    assert aliases.capture(runner_cls()) == 17
    assert calls == [
        ("dummy", 8, True, 0),
        ("mapping", 0),
        ("dummy", 1, True, 0),
        ("mapping", 0),
        ("sync",),
        ("capture",),
    ]
