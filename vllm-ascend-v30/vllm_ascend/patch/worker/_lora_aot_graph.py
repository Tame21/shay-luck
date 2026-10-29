# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep LoRA AOT variants aligned with graph warmup and capture."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import Any

import torch

_DUMMY_LORA_COUNT_ATTR = "_ascend_aot_dummy_lora_count"
_INSTALLED_ATTR = "_ascend_lora_aot_graph_hooks_installed"


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


def install_graph_hooks(enabled: Callable[[Any], bool]) -> None:
    """Install runner-local hooks for configurations using three LoRA AOT variants."""
    # Worker patches can be imported before the model runner is initialized.
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if NPUModelRunner.__dict__.get(_INSTALLED_ATTR, False):
        return

    original_dummy_run = NPUModelRunner._dummy_run
    original_dummy_lora_context = NPUModelRunner.maybe_dummy_run_with_lora
    original_capture_model = NPUModelRunner.capture_model
    dummy_signature = signature(original_dummy_run)
    context_signature = signature(original_dummy_lora_context)

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
                    self._dummy_run(num_tokens=self.max_num_tokens, is_profile=True, num_active_loras=0)
                self._dummy_run(num_tokens=1, is_profile=True, num_active_loras=0)
                torch.npu.synchronize()
        return original_capture_model(self, *args, **kwargs)

    NPUModelRunner._dummy_run = dummy_run
    NPUModelRunner.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora
    NPUModelRunner.capture_model = capture_model
    setattr(NPUModelRunner, _INSTALLED_ATTR, True)
