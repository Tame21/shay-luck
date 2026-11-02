# SPDX-License-Identifier: Apache-2.0
"""Resolve NetrsnTurbo's native LoRA component when a wrapper is constructed."""

from importlib import import_module

_REQUIRED_APIS = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "build_combined_lora_idx", "add_lora", "add_lora_shrink", "add_lora_expand",
    "add_lora_expand_delta", "add_lora_swiglu_quant",
)
_BACKEND = None


def initialize_lora_backend():
    """Validate the internal public APIs and native component once per worker.

    Importing this helper does not import turbo_ops, torch, or torch_npu.
    A failed initialization is not cached. Production workers must still be
    restarted after replacing their native libraries.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        backend = import_module("netrsn_turbo.turbo_ops")
        missing = [name for name in _REQUIRED_APIS if not callable(getattr(backend, name, None))]
        if missing:
            raise RuntimeError("missing public LoRA APIs: " + ", ".join(missing))
        initialize = getattr(backend, "initialize", None)
        if not callable(initialize):
            raise RuntimeError("missing netrsn_turbo.turbo_ops.initialize")
        initialize()
    except Exception as exc:
        raise RuntimeError(
            "NetrsnTurbo version_0300 DS LoRA requires its built-in turbo_ops "
            "native component compiled for the worker's CANN/torch/torch_npu "
            "environment. " + str(exc)
        ) from exc
    _BACKEND = backend
    return backend


def get_lora_ops():
    # _checked_init resolves this before execution/graph capture. The warm
    # path reads a module global without importlib or repeated validation.
    if _BACKEND is None:
        return initialize_lora_backend()
    return _BACKEND
