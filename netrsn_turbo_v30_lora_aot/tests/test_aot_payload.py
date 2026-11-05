# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from ._turbo_contract import apply_patches, load_payloads


@pytest.fixture(params=["qwen3_5_text", "deepseek_v4"])
def harness(request, monkeypatch, tmp_path):
    """Exercise the real dispatcher with observable CPU-only compiler boundaries."""
    events = []
    cache = {}
    control = SimpleNamespace(fail_compile=False, context=None)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=request.param),
            hf_config=SimpleNamespace(model_type="qwen3_5" if request.param == "qwen3_5_text" else request.param),
            model="local/model",
            enforce_eager=False,
        ),
        lora_config=SimpleNamespace(fully_sharded_loras=False),
        use_v2_model_runner=False,
        compilation_config=SimpleNamespace(
            mode=3, backend="inductor", cudagraph_mode=1, cudagraph_specialize_lora=True
        ),
    )

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    envs = stub(
        "vllm.envs",
        VLLM_USE_AOT_COMPILE=True,
        VLLM_USE_BYTECODE_HOOK=True,
        VLLM_DISABLE_COMPILE_CACHE=False,
        VLLM_FORCE_AOT_LOAD=False,
    )

    def load(model, path):
        events.append(("load", path))
        if envs.VLLM_FORCE_AOT_LOAD and path not in cache:
            raise FileNotFoundError(path)
        return cache.get(path)

    def init(model, compile_prefix="", is_encoder=False):
        events.append(("init", compile_prefix))
        model.vllm_config = config
        model._compile_prefix = compile_prefix
        model._is_encoder = is_encoder
        model.compiled = False
        model.first_compile = True
        model._compiled_callable = object()

    class Wrapper:
        __init__ = init

    class Runner:
        def _dummy_run(self, num_tokens, is_profile=False, num_active_loras=0):
            return num_tokens

        @contextmanager
        def maybe_dummy_run_with_lora(self, *args, num_active_loras=0, **kwargs):
            yield

        def capture_model(self):
            return None

    decorators = stub("vllm.compilation.decorators", _try_load_aot_compiled_fn=load)
    backends = stub("vllm.compilation.backends", model_tag="backbone")
    compilation = stub("vllm.compilation", decorators=decorators, backends=backends)
    stub("vllm", envs=envs, compilation=compilation)
    stub("vllm.compilation.wrapper", TorchCompileWithNoGuardsWrapper=Wrapper)
    stub(
        "vllm.config",
        CompilationMode=SimpleNamespace(VLLM_COMPILE=3),
        CUDAGraphMode=SimpleNamespace(NONE=0),
        get_current_vllm_config=lambda: config,
        set_current_vllm_config=lambda _: nullcontext(),
    )
    stub(
        "vllm.forward_context",
        get_forward_context=lambda: control.context,
        is_forward_context_available=lambda: control.context is not None,
    )
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("torch", npu=SimpleNamespace(synchronize=lambda: None))
    stub("vllm_ascend.worker.model_runner_v1", NPUModelRunner=Runner)
    aliases = stub("_lora_aot_aliases", init=init, load=load)
    payloads = load_payloads(monkeypatch)
    module = payloads.wrapper
    assert Wrapper.__init__ is init
    assert decorators._try_load_aot_compiled_fn is load
    replacements = (
        ("vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__", module.wrap_init(init)),
        (
            "vllm.compilation.decorators._try_load_aot_compiled_fn",
            payloads.decorators.wrap_try_load_aot_compiled_fn(load),
        ),
    )
    assert Wrapper.__init__ is init
    assert decorators._try_load_aot_compiled_fn is load
    apply_patches(monkeypatch, replacements)

    def _mark_dynamic_inputs(model, *args, **kwargs):
        events.append(("dynamic", model._ascend_aot_active_variant))

    # Retain the upstream decorator's closure and load/compile/save protocol;
    # the production dispatcher must call this path only for missing variants.
    def decorator_call(model, *args, **kwargs):
        if getattr(model, "aot_compiled_fn", None) is not None:
            return model.aot_compiled_fn(model, *args, **kwargs)
        variant = model._ascend_aot_active_variant
        events.append(("cold", variant))
        path = str(tmp_path / "model")
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            artifact = decorators._try_load_aot_compiled_fn(model, path)
            if artifact is not None:
                model.aot_compiled_fn = artifact
                model.was_aot_compile_fn_loaded_from_disk = True
                return artifact(model, *args, **kwargs)
        _mark_dynamic_inputs(model, *args, **kwargs)
        model._aot_compilation_path = path
        model._aot_cache_dir = str(tmp_path)
        events.append(("compile", variant))
        if control.fail_compile:
            raise RuntimeError("backend failed")

        def artifact(model, *args, **kwargs):
            events.append(("run", variant))
            return variant

        model.aot_compiled_fn = artifact
        model.save_aot_compiled_function()
        return artifact(model, *args, **kwargs)

    class Model:
        __call__ = decorator_call

        def __init__(self):
            self.punica = SimpleNamespace(no_lora=False)
            self.aot_compiled_fn = None
            self.was_aot_compile_fn_loaded_from_disk = False
            self._aot_compilation_path = None
            self._aot_cache_dir = None
            Wrapper.__init__(self)

        def forward(self, input_ids=None, inputs_embeds=None, intermediate_tensors=None):
            return input_ids

        def modules(self):
            return [SimpleNamespace(punica_wrapper=self.punica)]

        def save_aot_compiled_function(self):
            if envs.VLLM_DISABLE_COMPILE_CACHE or self.was_aot_compile_fn_loaded_from_disk:
                return
            events.append(("save", self._aot_compilation_path))
            cache[self._aot_compilation_path] = self.aot_compiled_fn

    return SimpleNamespace(
        module=module,
        model=Model(),
        new_model=Model,
        envs=envs,
        config=config,
        events=events,
        cache=cache,
        control=control,
        wrapper_cls=Wrapper,
        decorators=decorators,
        aliases=aliases,
        path=str(tmp_path / "model"),
    )


def _tokens(count):
    return SimpleNamespace(shape=(count,))


def test_alias_propagation_keeps_original_init_and_load_callable(harness):
    assert harness.aliases.init is harness.wrapper_cls.__init__
    assert harness.aliases.load is harness.decorators._try_load_aot_compiled_fn
    assert harness.model(_tokens(2)) == "lora"
    assert [value for event, value in harness.events if event == "init"] == [""]
    assert ("load", f"{harness.path}.lora") in harness.events
    harness.events.clear()
    model = harness.new_model()
    assert model(_tokens(2)) == "lora"
    assert model.was_aot_compile_fn_loaded_from_disk
    assert not any(event == "compile" for event, _ in harness.events)


@pytest.mark.parametrize(
    "mapping,expected",
    [((0,), "base_one"), ((0, 0), "base"), ((7,), "lora"), ((7, 9), "lora"), ((0, 7), "lora")],
)
def test_dispatch_uses_current_batch_including_mixed_base_and_adapters(harness, mapping, expected):
    harness.model.punica.no_lora = not any(adapter > 0 for adapter in mapping)
    assert harness.model(_tokens(len(mapping))) == expected


@pytest.mark.parametrize("input_name", ["inputs_embeds", "intermediate_tensors"])
def test_single_token_base_uses_embedding_or_pipeline_input_shape(harness, input_name):
    harness.model.punica.no_lora = True
    value = _tokens(1)
    if input_name == "intermediate_tensors":
        value = SimpleNamespace(tensors={"hidden_states": value})
    assert harness.model(**{input_name: value}) == "base_one"


def test_descriptor_fallback_and_missing_punica_remain_conservative(harness):
    harness.model.punica.no_lora = True
    harness.control.context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1), capturing=False)
    assert harness.model() == "base_one"
    model = harness.new_model()
    model.modules = list
    assert model(_tokens(1)) == "lora"


def test_each_variant_uses_original_cold_path_once_then_reuses_its_artifact(harness):
    for _ in range(2):
        for variant, count in (("lora", 2), ("base", 2), ("base_one", 1)):
            harness.model.punica.no_lora = variant != "lora"
            assert harness.model(_tokens(count)) == variant
    assert [value for event, value in harness.events if event == "cold"] == ["lora", "base", "base_one"]
    assert [value for event, value in harness.events if event == "compile"] == ["lora", "base", "base_one"]
    assert set(harness.cache) == {f"{harness.path}.{name}" for name in ("lora", "base", "base_one")}
    assert harness.model.aot_compiled_fn is harness.model._ascend_lora_aot_dispatcher


def test_partial_cache_loads_existing_variants_and_only_compiles_the_missing_one(harness):
    harness.model(_tokens(2))
    harness.model.punica.no_lora = True
    harness.model(_tokens(2))
    existing = harness.cache.copy()
    harness.events.clear()
    model = harness.new_model()
    assert model(_tokens(2)) == "lora"
    model.punica.no_lora = True
    assert model(_tokens(2)) == "base"
    assert model(_tokens(1)) == "base_one"
    model.save_aot_compiled_function()
    assert [value for event, value in harness.events if event == "compile"] == ["base_one"]
    assert all(harness.cache[path] is artifact for path, artifact in existing.items())
    assert set(harness.cache) == {*existing, f"{harness.path}.base_one"}
    assert model.was_aot_compile_fn_loaded_from_disk
    assert all(state.path == harness.path for state in model.aot_compiled_fn.variants.values())


@pytest.mark.parametrize("failure", ["compile", "forced_load"])
def test_cold_failure_restores_model_and_retry_uses_a_fresh_compiler(harness, failure):
    model = harness.model
    model.punica.no_lora = True
    before = vars(model).copy()
    harness.control.fail_compile = failure == "compile"
    harness.envs.VLLM_FORCE_AOT_LOAD = failure == "forced_load"
    with pytest.raises(RuntimeError if failure == "compile" else FileNotFoundError):
        model(_tokens(2))
    assert vars(model) == before
    state = model.aot_compiled_fn.variants["base"]
    assert state.artifact is None
    assert state.callable is None
    harness.control.fail_compile = False
    harness.envs.VLLM_FORCE_AOT_LOAD = False
    assert model(_tokens(2)) == "base"
    assert sum(event == "init" and prefix == "backbone.base" for event, prefix in harness.events) == 2


def test_cache_disabled_retains_upstream_no_load_no_save_behavior(harness):
    harness.cache[f"{harness.path}.base"] = lambda *_: pytest.fail("disabled cache was loaded")
    harness.envs.VLLM_DISABLE_COMPILE_CACHE = True
    harness.model.punica.no_lora = True
    assert harness.model(_tokens(2)) == "base"
    assert harness.model(_tokens(2)) == "base"
    assert not any(event in ("load", "save") for event, _ in harness.events)
    assert [value for event, value in harness.events if event == "compile"] == ["base"]


def test_only_cloned_base_one_decorator_skips_dynamic_marking(harness):
    original = type(harness.model).__call__
    cells = dict(zip(original.__code__.co_freevars, original.__closure__))
    original_marker = cells["_mark_dynamic_inputs"].cell_contents
    for variant, count in (("base", 2), ("base_one", 1), ("lora", 1)):
        harness.model.punica.no_lora = variant != "lora"
        assert harness.model(_tokens(count)) == variant
    assert [value for event, value in harness.events if event == "dynamic"] == ["base", "lora"]
    assert cells["_mark_dynamic_inputs"].cell_contents is original_marker
    plain = SimpleNamespace(
        _ascend_aot_active_variant="base_one",
        save_aot_compiled_function=lambda: None,
    )
    harness.envs.VLLM_DISABLE_COMPILE_CACHE = True
    assert original(plain, _tokens(1)) == "base_one"
    assert ("dynamic", "base_one") in harness.events


def test_capture_requires_warmup_but_allows_hot_variant_replay(harness):
    harness.model.punica.no_lora = True
    harness.control.context = SimpleNamespace(capturing=True)
    with pytest.raises(RuntimeError, match="warmed up"):
        harness.model(_tokens(1))
    harness.control.context = None
    assert harness.model(_tokens(1)) == "base_one"
    harness.control.context = SimpleNamespace(capturing=True)
    assert harness.model(_tokens(1)) == "base_one"


def test_reset_discards_variants_and_disabling_restores_original_save_method(harness):
    model = harness.model
    assert model(_tokens(2)) == "lora"
    previous = model.aot_compiled_fn
    harness.wrapper_cls.__init__(model)
    assert model.aot_compiled_fn is not previous
    assert all(state.artifact is None for state in model.aot_compiled_fn.variants.values())
    harness.config.model_config.enforce_eager = True
    harness.wrapper_cls.__init__(model)
    assert model.aot_compiled_fn is None
    assert not hasattr(model, "_ascend_lora_aot_dispatcher")
    assert model.save_aot_compiled_function.__func__ is type(model).save_aot_compiled_function
    assert harness.envs.VLLM_USE_BYTECODE_HOOK is True


@pytest.mark.parametrize(
    "path,value",
    [
        ("config.model_config", None),
        ("config.lora_config", None),
        ("config.lora_config.fully_sharded_loras", True),
        ("envs.VLLM_USE_AOT_COMPILE", False),
        ("config.model_config.enforce_eager", True),
        ("config.use_v2_model_runner", True),
        ("config.compilation_config.mode", 0),
        ("config.compilation_config.backend", "eager"),
        ("config.compilation_config.cudagraph_mode", 0),
        ("config.compilation_config.cudagraph_specialize_lora", False),
        ("config.model_config.hf_text_config.model_type", "llama"),
        ("config.model_config.hf_text_config.model_type", "deepseek_v3"),
        ("config.model_config.hf_text_config.model_type", "deepseek_v41"),
    ],
)
def test_unsupported_configuration_keeps_original_wrapper(harness, path, value):
    target = harness
    *parents, attribute = path.split(".")
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, attribute, value)
    assert not harness.module.enabled(harness.config)
    model = harness.new_model()
    assert model.aot_compiled_fn is None
    assert not hasattr(model, "_ascend_lora_aot_dispatcher")
    assert model.save_aot_compiled_function.__func__ is type(model).save_aot_compiled_function


@pytest.mark.parametrize(
    "text_type,outer_type,model_name",
    [
        ("qwen3_5_text", "qwen3_5", "Qwen/Qwen3.5-4B"),
        ("deepseek_v4", "deepseek_v4", "local/DeepSeek-V4"),
        ("mistral", "mistral3", "local/mistral3"),
        ("mistral", "mistral", "mistralai/Magistral-Small-2509"),
    ],
)
def test_supported_model_families_install_variant_dispatcher(harness, text_type, outer_type, model_name):
    model_config = harness.config.model_config
    model_config.hf_text_config.model_type = text_type
    model_config.hf_config.model_type = outer_type
    model_config.model = model_name
    assert harness.module.enabled(harness.config)
    assert isinstance(harness.new_model().aot_compiled_fn, harness.module._AOTVariantDispatcher)
