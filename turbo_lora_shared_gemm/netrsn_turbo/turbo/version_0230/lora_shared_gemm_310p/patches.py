# SPDX-License-Identifier: Apache-2.0
"""Build TurboManager replacements for the 310P shared-GEMM LoRA path.

This module only builds replacements. The Turbo entry point owns hardware and
version checks, registration, and application in each process.
"""

from __future__ import annotations

from functools import wraps

import torch


def _token_delta(self, x: torch.Tensor, weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # The scheduler already expands adapter IDs to tokens. Reading a repeat
    # count from the device here would synchronize during graph capture.
    indices = self._get_token_lora_indices(x)
    if weights.shape[0] == 1:
        # Keep a live view of the resident storage: loading/resetting a dynamic
        # adapter updates these weights in place, including for graph replay.
        matrix = weights[0, 0] if weights.ndim == 4 else weights[0]
        delta = x.to(dtype=dtype) @ matrix.to(dtype=dtype).T
        active = indices == 0
    else:
        selected = weights[indices.clamp_min(0)].to(dtype=dtype)
        if selected.ndim == 4:
            selected = selected.squeeze(1)
        delta = torch.einsum("bi, boi -> bo", x.to(dtype=dtype), selected)
        active = indices >= 0
    # Base-model tokens and graph padding use -1. Use where rather than
    # multiplication so even NaNs from inactive rows cannot enter the output.
    return torch.where(active.unsqueeze(1), delta, torch.zeros_like(delta))


def _shrink_prefill(self, y, x, w_t_all, scale):
    compute_dtype = y.dtype
    if w_t_all.shape[0] == 1 and x.dtype == w_t_all.dtype == torch.float16 and y.dtype == torch.float32:
        # Project the large input in FP16, then promote the small rank result.
        # Keep the FP32 scratch and apply scale in FP32 as before.
        compute_dtype = torch.float16
    delta = _token_delta(self, x, w_t_all, compute_dtype).to(dtype=y.dtype)
    y[:, : delta.shape[1]].copy_(delta * scale)


def _expand_prefill(self, y, x, w_t_all, add_inputs):
    delta = _token_delta(self, x, w_t_all, y.dtype)
    width = min(delta.shape[1], y.shape[1])
    if add_inputs:
        y[:, :width].add_(delta[:, :width])
    else:
        y[:, :width].copy_(delta[:, :width])


def _expand_slice_prefill(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs):
    delta = _token_delta(self, x, w_t_all, y.dtype)
    target = y[:, y_offset : y_offset + y_slice_size]
    if add_inputs:
        target.add_(delta)
    else:
        target.copy_(delta)


def _wrap_init(original):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._token_lora_indices.fill_(-1)

    return initialize


def _wrap_base_metadata(original):
    @wraps(original)
    def update_base_metadata(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Updates only write actual tokens; clear the rest of the preallocated
        # mapping before a larger graph bucket consumes it.
        self._token_lora_indices[self.indices_len[0] :].fill_(-1)
        return result

    return update_base_metadata


def _wrap_mcp_apply(original, base_layer):
    @wraps(original)
    def mcp_apply(x, bias, layer):
        if layer.tp_size > 1 and not layer.lora_config.fully_sharded_loras:
            # Non-sharded A is replicated. Its shrink already has the full
            # rank, so gathering rank shards would produce the wrong shape.
            return base_layer._apply_sync(layer, x, bias)
        return original(x, bias, layer)

    mcp_apply._netrsn_shared_gemm_column_310p = True
    return mcp_apply


def build_patches() -> list[tuple[str, object]]:
    """Capture current originals and return targets for TurboManager.

    Imports stay here so importing this payload does not independently load
    vLLM/Ascend before the Turbo entry point establishes their load order.
    """
    from vllm.lora.layers import column_parallel_linear
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    punica = "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"
    return [
        (
            "vllm.lora.layers.column_parallel_linear._mcp_apply",
            _wrap_mcp_apply(column_parallel_linear._mcp_apply, BaseLinearLayerWithLoRA),
        ),
        (f"{punica}.__init__", _wrap_init(PunicaWrapperNPU.__init__)),
        (
            f"{punica}._update_base_metadata",
            _wrap_base_metadata(PunicaWrapperNPU._update_base_metadata),
        ),
        (f"{punica}._shrink_prefill", _shrink_prefill),
        (f"{punica}._expand_prefill", _expand_prefill),
        (f"{punica}._expand_slice_prefill", _expand_slice_prefill),
    ]
