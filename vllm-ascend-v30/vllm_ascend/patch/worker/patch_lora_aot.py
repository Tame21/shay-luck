# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate base and LoRA AOT artifacts without replacing vLLM's compiler."""

import inspect
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import vllm.envs as vllm_envs
from vllm.compilation import backends, decorators
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config, set_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_ascend.patch.worker._lora_aot_graph import install_graph_hooks

logger = init_logger(__name__)

_VARIANTS = ("lora", "base", "base_one")
_ORIGINAL_WRAPPER_INIT = TorchCompileWithNoGuardsWrapper.__init__
_ORIGINAL_LOAD = decorators._try_load_aot_compiled_fn


def enabled(config) -> bool:
    model = config.model_config
    lora = config.lora_config
    if model is None or lora is None:
        return False
    text_type = getattr(model.hf_text_config, "model_type", None)
    supported_model = text_type == "qwen3_5_text" or (
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
    cloned = FunctionType(code, original.__globals__, code.co_name, original.__defaults__, original.__closure__)
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
    def __init__(self, model):
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
        with set_current_vllm_config(model.vllm_config), patch.object(vllm_envs, "VLLM_USE_BYTECODE_HOOK", False):
            _ORIGINAL_WRAPPER_INIT(proxy, compile_prefix=prefix, is_encoder=model._is_encoder)
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
        logger.info("LoRA AOT variant %s is ready (%s)", variant, "loaded" if state.loaded else "compiled")
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


def _try_load_aot_compiled_fn(model, path):
    variant = getattr(model, "_ascend_aot_active_variant", None)
    if variant is None:
        return _ORIGINAL_LOAD(model, path)
    # The upstream load fast path does not initialize the model's save paths.
    model._aot_compilation_path = path
    model._aot_cache_dir = os.path.dirname(path)
    return _ORIGINAL_LOAD(model, f"{path}.{variant}")


@wraps(_ORIGINAL_WRAPPER_INIT)
def _wrapper_init(self, compile_prefix="", is_encoder=False):
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
        _ORIGINAL_WRAPPER_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
    if not active:
        return
    self._ascend_lora_aot_dispatcher = _AOTVariantDispatcher(self)
    self.aot_compiled_fn = self._ascend_lora_aot_dispatcher
    self.save_aot_compiled_function = MethodType(_save_aot_compiled_function, self)


def _install():
    if getattr(TorchCompileWithNoGuardsWrapper, "_ascend_lora_aot_installed", False):
        return
    install_graph_hooks(enabled)
    TorchCompileWithNoGuardsWrapper.__init__ = _wrapper_init
    decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn
    TorchCompileWithNoGuardsWrapper._ascend_lora_aot_installed = True


_install()
