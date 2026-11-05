# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA-aware warmup and capture wrappers for Ascend Model Runner V1."""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch

from netrsn_turbo.turbo.version_v0300.vllm.compilation.wrapper import enabled

_DUMMY_LORA_COUNT_ATTR = "_ascend_aot_dummy_lora_count"


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


def wrap_dummy_run(original_dummy_run):
    dummy_signature = inspect.signature(original_dummy_run)

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

    return dummy_run


def wrap_maybe_dummy_run_with_lora(original_dummy_lora_context):
    context_signature = inspect.signature(original_dummy_lora_context)

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

    return maybe_dummy_run_with_lora


def wrap_capture_model(original_capture_model):
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

    return capture_model
