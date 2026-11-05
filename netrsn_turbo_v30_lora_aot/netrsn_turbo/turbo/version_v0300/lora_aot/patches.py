# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build v0.30 LoRA AOT replacements for installation by TurboManager."""

import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends, decorators
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

_VARIANTS = ("lora", "base", "base_one")
_DUMMY_LORA_COUNT_ATTR = "_ascend_aot_dummy_lora_count"


def enabled(config) -> bool:
    model = config.model_config
    lora = config.lora_config
    if model is None or lora is None:
        return False
    text_type = getattr(model.hf_text_config, "model_type", None)
    supported_model = text_type in ("qwen3_5_text", "deepseek_v4") or (
        text_type == "mistral"
        and (getattr(model.hf_config, "model_type", None) == "mistral3" or "magistral" in str(model.model).lower())
    )
    compilation = config.compilation_config
    return bool(
        supported_model
        and vllm_envs.VLLM_USE_AOT_COMPILE
        and not lora.fully_sharded_loras
        and not config.use_v2_model_runner
        and not model.enforce_eager
        and compilation.mode == CompilationMode.VLLM_COMPILE
        and compilation.backend != "eager"
        and compilation.cudagraph_mode != CUDAGraphMode.NONE
        and compilation.cudagraph_specialize_lora
    )


@contextmanager
def _dummy_lora_count(runner: Any, count: int) -> Iterator[None]:
    missing = object()
    previous = getattr(runner, _DUMMY_LORA_COUNT_ATTR, missing)
    setattr(runner, _DUMMY_LORA_COUNT_ATTR, count)
    try:
        yield
    finally:
        if previous is missing:
            delattr(runner, _DUMMY_LORA_COUNT_ATTR)
        else:
            setattr(runner, _DUMMY_LORA_COUNT_ATTR, previous)


def build_graph_hooks(runner_cls) -> tuple[tuple[str, Callable], ...]:
    """Capture existing runner methods without mutating their class."""
    original_dummy_run = runner_cls._dummy_run
    original_dummy_lora_context = runner_cls.maybe_dummy_run_with_lora
    original_capture_model = runner_cls.capture_model
    dummy_signature = inspect.signature(original_dummy_run)
    context_signature = inspect.signature(original_dummy_lora_context)

    @wraps(original_dummy_run)
    def dummy_run(self, *args, **kwargs):
        if not enabled(self.vllm_config):
            return original_dummy_run(self, *args, **kwargs)

        count = getattr(self, _DUMMY_LORA_COUNT_ATTR, None)
        if count is None:
            bound = dummy_signature.bind(self, *args, **kwargs)
            # Ordinary profiling must still activate dummy adapters so the
            # initial compilation and memory estimate include the LoRA path.
            if not bound.arguments.get("is_profile", False) and "num_active_loras" in bound.arguments:
                count = bound.arguments["num_active_loras"]
        if count is None:
            return original_dummy_run(self, *args, **kwargs)

        with _dummy_lora_count(self, count):
            return original_dummy_run(self, *args, **kwargs)

    @contextmanager
    @wraps(original_dummy_lora_context)
    def maybe_dummy_run_with_lora(self, *args, **kwargs):
        count = getattr(self, _DUMMY_LORA_COUNT_ATTR, None) if enabled(self.vllm_config) else None
        if count is not None:
            bound = context_signature.bind(self, *args, **kwargs)
            bound.arguments["num_active_loras"] = count
            context = original_dummy_lora_context(*bound.args, **bound.kwargs)
        else:
            context = original_dummy_lora_context(self, *args, **kwargs)
        with context:
            yield

    @wraps(original_capture_model)
    def capture_model(self, *args, **kwargs):
        if enabled(self.vllm_config):
            # Compile and execute both base variants before graph-pool capture
            # and workspace locking. The LoRA variant is prepared by profiling.
            with _dummy_lora_count(self, 0):
                if self.max_num_tokens > 1:
                    self._dummy_run(
                        num_tokens=self.max_num_tokens,
                        is_profile=True,
                        num_active_loras=0,
                    )
                self._dummy_run(num_tokens=1, is_profile=True, num_active_loras=0)
                torch.npu.synchronize()
        return original_capture_model(self, *args, **kwargs)

    prefix = "vllm_ascend.worker.model_runner_v1.NPUModelRunner"
    return (
        (f"{prefix}._dummy_run", dummy_run),
        (f"{prefix}.maybe_dummy_run_with_lora", maybe_dummy_run_with_lora),
        (f"{prefix}.capture_model", capture_model),
    )


@contextmanager
def _temporary_attributes(obj, **values):
    missing = object()
    previous = {name: vars(obj).get(name, missing) for name in values}
    try:
        for name, value in values.items():
            setattr(obj, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                vars(obj).pop(name, None)
            else:
                setattr(obj, name, value)


def _clone_forward(model, variant):
    original = model.forward.__func__
    names = {"co_name": f"{original.__code__.co_name}_ascend_{variant}"}
    if hasattr(original.__code__, "co_qualname"):
        names["co_qualname"] = f"{original.__code__.co_qualname}_ascend_{variant}"
    code = original.__code__.replace(**names)
    cloned = FunctionType(
        code,
        original.__globals__,
        code.co_name,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    cloned.__annotations__ = original.__annotations__
    return MethodType(cloned, model)


def _variant_decorator_call(original):
    """Keep v0.30's compile/load contexts; only skip dynamic marking for base_one."""
    closure = dict(zip(original.__code__.co_freevars, original.__closure__ or ()))
    if "_mark_dynamic_inputs" not in closure:
        raise RuntimeError("LoRA AOT variants require the v0.30 support_torch_compile decorator")
    mark_dynamic_inputs = closure["_mark_dynamic_inputs"].cell_contents

    def mark_variant_inputs(model, *args, **kwargs):
        if model._ascend_aot_active_variant != "base_one":
            mark_dynamic_inputs(model, *args, **kwargs)

    # Build a new cell instead of changing the decorator shared by other models.
    cell = (lambda: mark_variant_inputs).__closure__[0]
    cloned = FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        tuple(cell if name == "_mark_dynamic_inputs" else closure[name] for name in original.__code__.co_freevars),
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    return cloned


@dataclass
class _Variant:
    callable: Any = None
    artifact: Any = None
    path: str | None = None
    loaded: bool = False


class _AOTVariantDispatcher:
    def __init__(self, model, original_wrapper_init):
        self.original_wrapper_init = original_wrapper_init
        self.variants = {name: _Variant() for name in _VARIANTS}
        self.variants["lora"].callable = model._compiled_callable
        self.decorator_call = _variant_decorator_call(type(model).__call__)
        self.original_save = type(model).save_aot_compiled_function
        self.punica_wrappers = None
        self.input_positions = {
            name: index
            for index, (name, parameter) in enumerate(inspect.signature(model.forward).parameters.items())
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        }

    def select_variant(self, model, args, kwargs):
        if self.punica_wrappers is None:
            self.punica_wrappers = tuple(
                {
                    id(wrapper): wrapper
                    for module in model.modules()
                    if (wrapper := getattr(module, "punica_wrapper", None)) is not None
                }.values()
            )
        if not self.punica_wrappers or any(not wrapper.no_lora for wrapper in self.punica_wrappers):
            return "lora"

        for name in ("input_ids", "inputs_embeds", "intermediate_tensors"):
            position = self.input_positions.get(name, len(args))
            value = kwargs.get(name, args[position] if position < len(args) else None)
            if value is None:
                continue
            if name == "intermediate_tensors":
                value = next(iter(value.tensors.values()))
            return "base_one" if value.shape[0] == 1 else "base"
        if is_forward_context_available():
            descriptor = get_forward_context().batch_descriptor
            if descriptor is not None and descriptor.num_tokens == 1:
                return "base_one"
        return "base"

    def _make_callable(self, model, variant):
        forward = _clone_forward(model, variant)
        proxy = SimpleNamespace(forward=forward)
        prefix = f"{model._compile_prefix or 'backbone'}.{variant}"
        with (
            set_current_vllm_config(model.vllm_config),
            patch.object(vllm_envs, "VLLM_USE_BYTECODE_HOOK", False),
        ):
            self.original_wrapper_init(proxy, compile_prefix=prefix, is_encoder=model._is_encoder)
        return proxy._compiled_callable

    def __call__(self, model, *args, **kwargs):
        variant = self.select_variant(model, args, kwargs)
        state = self.variants[variant]
        if state.artifact is not None:
            return state.artifact(model, *args, **kwargs)
        if is_forward_context_available() and getattr(get_forward_context(), "capturing", False):
            raise RuntimeError(f"LoRA AOT variant {variant} must be warmed up before ACL graph capture")
        if state.callable is None:
            state.callable = self._make_callable(model, variant)

        # The original decorator handles source tracking, dynamic shape options,
        # partition wrappers, cache validation and the first profiling execution.
        # Hot calls above never swap model state.
        with _temporary_attributes(
            model,
            _compiled_callable=state.callable,
            aot_compiled_fn=None,
            compiled=False,
            first_compile=True,
            was_aot_compile_fn_loaded_from_disk=False,
            _aot_compilation_path=None,
            _aot_cache_dir=None,
            _ascend_aot_active_variant=variant,
        ):
            try:
                output = self.decorator_call(model, *args, **kwargs)
                state.artifact = model.aot_compiled_fn
                state.path = model._aot_compilation_path
                state.loaded = model.was_aot_compile_fn_loaded_from_disk
            except Exception:
                # A backend instance can only compile once. Do not reuse a
                # partially initialized backend on a later retry.
                state.callable = None
                raise
        model.compiled = True
        model.was_aot_compile_fn_loaded_from_disk = any(item.loaded for item in self.variants.values())
        logger.info(
            "LoRA AOT variant %s is ready (%s)",
            variant,
            "loaded" if state.loaded else "compiled",
        )
        return output

    def save(self, model):
        variant = getattr(model, "_ascend_aot_active_variant", None)
        if variant is not None:
            with _temporary_attributes(model, _aot_compilation_path=f"{model._aot_compilation_path}.{variant}"):
                return self.original_save(model)
        for name, state in self.variants.items():
            if state.artifact is None or state.path is None:
                continue
            with _temporary_attributes(
                model,
                aot_compiled_fn=state.artifact,
                _aot_compilation_path=f"{state.path}.{name}",
                _aot_cache_dir=os.path.dirname(state.path),
                was_aot_compile_fn_loaded_from_disk=state.loaded,
            ):
                self.original_save(model)


def _save_aot_compiled_function(model):
    model._ascend_lora_aot_dispatcher.save(model)


def build_patches() -> tuple[tuple[str, Callable], ...]:
    """Build one replacement set for the runtime selected by the outer router.

    Originals stay in closures or dispatcher attributes so TurboManager's
    module-level alias propagation cannot replace them with their wrappers.
    The caller owns registration, application and repeated-load handling.
    """
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_wrapper_init = TorchCompileWithNoGuardsWrapper.__init__
    original_load = decorators._try_load_aot_compiled_fn

    @wraps(original_load)
    def try_load_aot_compiled_fn(model, path):
        variant = getattr(model, "_ascend_aot_active_variant", None)
        if variant is None:
            return original_load(model, path)
        # The upstream load fast path does not initialize the model's save paths.
        model._aot_compilation_path = path
        model._aot_cache_dir = os.path.dirname(path)
        return original_load(model, f"{path}.{variant}")

    @wraps(original_wrapper_init)
    def wrapper_init(self, compile_prefix="", is_encoder=False):
        config = get_current_vllm_config()
        active = enabled(config) and backends.model_tag == "backbone" and not is_encoder
        if isinstance(getattr(self, "aot_compiled_fn", None), _AOTVariantDispatcher):
            self.aot_compiled_fn = None
        # reset_compile_wrapper must also discard all variant-local state.
        if "_ascend_lora_aot_dispatcher" in vars(self):
            del self._ascend_lora_aot_dispatcher
            vars(self).pop("save_aot_compiled_function", None)
        context = patch.object(vllm_envs, "VLLM_USE_BYTECODE_HOOK", False) if active else nullcontext()
        with context:
            original_wrapper_init(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
        if not active:
            return
        self._ascend_lora_aot_dispatcher = _AOTVariantDispatcher(self, original_wrapper_init)
        self.aot_compiled_fn = self._ascend_lora_aot_dispatcher
        self.save_aot_compiled_function = MethodType(_save_aot_compiled_function, self)

    return (
        (
            "vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__",
            wrapper_init,
        ),
        (
            "vllm.compilation.decorators._try_load_aot_compiled_fn",
            try_load_aot_compiled_fn,
        ),
        *build_graph_hooks(NPUModelRunner),
    )
