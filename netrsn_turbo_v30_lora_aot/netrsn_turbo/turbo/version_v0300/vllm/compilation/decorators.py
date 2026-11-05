# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Variant-aware vLLM AOT decorator and cache loading helpers."""

import os
from functools import wraps
from types import FunctionType


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


def wrap_try_load_aot_compiled_fn(original_load):
    """Load the selected variant while retaining the original loader in a closure."""

    @wraps(original_load)
    def try_load_aot_compiled_fn(model, path):
        variant = getattr(model, "_ascend_aot_active_variant", None)
        if variant is None:
            return original_load(model, path)
        # The upstream load fast path does not initialize the model's save paths.
        model._aot_compilation_path = path
        model._aot_cache_dir = os.path.dirname(path)
        return original_load(model, f"{path}.{variant}")

    return try_load_aot_compiled_fn
