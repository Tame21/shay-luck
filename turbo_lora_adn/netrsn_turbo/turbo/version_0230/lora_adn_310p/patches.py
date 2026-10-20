# SPDX-License-Identifier: Apache-2.0
"""Build the v0.23 ADN LoRA replacements for TurboManager.

The bootstrap imports this module only after its version/310P checks pass.
Projection replacements are ordinary named functions; build_patches registers
the graph dispatcher and returns them directly to TurboManager.
"""

from __future__ import annotations

from functools import wraps

import torch
from adn_custom_ops import adn_batch_gather_matmul_
from vllm.logger import logger
from vllm.lora.layers import column_parallel_linear
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

_ADN_SLICE_MODE = 1
_ADN_OP_NAME = "netrsn_adn_lora_310p_0230"
_PUNICA_TARGET = "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"


def _wrap_mcp_apply(original):
    @wraps(original)
    def mcp_apply(x, bias, layer):
        if layer.tp_size > 1 and not layer.lora_config.fully_sharded_loras:
            # Ordinary LoRA A already produces the full, replicated rank.
            return BaseLinearLayerWithLoRA._apply_sync(layer, x, bias)
        return original(x, bias, layer)

    return mcp_apply


def _wrap_init(original):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._token_lora_indices.fill_(-1)
        self._seq_lengths.zero_()
        self._lora_indices_per_batch.fill_(-1)
        self._sampler_indices.fill_(-1)

    return initialize


def _wrap_base_metadata(original):
    @wraps(original)
    def update_base_metadata(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Captured buckets also consume padded rows. Clear mappings outside
        # forward/capture so replay cannot reuse an earlier request's adapter.
        self._token_lora_indices[self.indices_len[0] :].fill_(-1)
        self._sampler_indices[self.indices_len[1] :].fill_(-1)
        return result

    return update_base_metadata


def _wrap_prefill_metadata(original):
    @wraps(original)
    def update_prefill_metadata(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Retain fixed capacity: slicing by this Python batch_size inside
        # forward would freeze the group count in an unguarded compiled graph.
        self._seq_lengths[self.batch_size :].zero_()
        self._lora_indices_per_batch[self.batch_size :].fill_(-1)
        return result

    return update_prefill_metadata


def _adn_dispatch(
    y: torch.Tensor,
    x: torch.Tensor,
    weight_a: torch.Tensor | None,
    weight_b: torch.Tensor,
    indices: torch.Tensor,
    lora_order: torch.Tensor,
    lora_batch_num: torch.Tensor,
    scale: float,
    y_offset: int,
    grouped: bool,
) -> None:
    # Keep the shape choice inside the registered op so an unguarded compiled
    # range does not freeze the first choice. Small shapes use the token API.
    if grouped and lora_order.shape[0] < min(x.shape[0], y.shape[0]):
        adn_batch_gather_matmul_(
            y,
            x=x,
            weight_a=weight_a,
            weight_b=weight_b,
            indices=None,
            lora_order=lora_order,
            lora_batch_num=lora_batch_num,
            layer_idx=0,
            scale=scale,
            y_offset=y_offset,
            y_slice_size=_ADN_SLICE_MODE,
        )
    else:
        adn_batch_gather_matmul_(
            y,
            x=x,
            weight_a=weight_a,
            weight_b=weight_b,
            indices=indices,
            layer_idx=0,
            scale=scale,
            y_offset=y_offset,
            y_slice_size=_ADN_SLICE_MODE,
        )


def _adn_dispatch_fake(*args, **kwargs) -> None:
    return None


def _call_adn(self, y, x, weight_a, weight_b, scale, y_offset, *, grouped, indices=None):
    """Prepare shared metadata, then invoke the registered ADN dispatcher."""
    if indices is None:
        indices = self._get_token_lora_indices(x)
    indices = indices.to(torch.int32)
    if grouped:
        lora_order = self._lora_indices_per_batch.to(torch.int32)
        lora_batch_num = self._seq_lengths.to(torch.int32)
    else:
        # The token API ignores group arguments; avoid unused metadata casts.
        lora_order = lora_batch_num = indices
    torch.ops.vllm.netrsn_adn_lora_310p_0230(
        y, x, weight_a, weight_b, indices, lora_order, lora_batch_num, scale, y_offset, grouped
    )
    if not torch.compiler.is_compiling() and not getattr(self, "_netrsn_adn_logged_dispatch", False):
        logger.info(
            "[Turbo ADN LoRA] adn_batch_gather_matmul_ dispatch returned (eager/warmup); "
            "this is not a device-completion or ACLGraph replay counter"
        )
        self._netrsn_adn_logged_dispatch = True


def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, *, buffer=None, **kwargs):
    assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
    x = x.reshape(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    offset = 0
    for a, b, width in zip(lora_a_stacked, lora_b_stacked, output_slices):
        if not kwargs.get("add_inputs", True):
            y[:, offset : offset + width].zero_()
        # V1 also uses is_prefill=True on decode; dispatch checks actual shape.
        _call_adn(self, y, x, a, b, scale, offset, grouped=self.is_prefill)
        offset += width


def add_lora_logits(self, y, x, lora_a_stacked, lora_b_stacked, scale, *, buffer=None, **kwargs):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    if not kwargs.get("add_inputs", True):
        y[:, : lora_b_stacked.shape[-2]].zero_()
    _call_adn(
        self,
        y,
        x,
        lora_a_stacked,
        lora_b_stacked,
        scale,
        0,
        grouped=False,
        indices=self._sampler_indices[: x.shape[0]],
    )


def _shrink_prefill(self, y, x, w_t_all, scale):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    # ADN accumulates; shrink must overwrite the existing rank scratch.
    y[:, : w_t_all.shape[-2]].zero_()
    _call_adn(self, y, x, None, w_t_all, scale, 0, grouped=True)


def _shrink_decode(self, y, x, w_t_all, scale):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    y[:, : w_t_all.shape[-2]].zero_()
    _call_adn(self, y, x, None, w_t_all, scale, 0, grouped=False)


def _expand_prefill(self, y, x, w_t_all, add_inputs):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    if not add_inputs:
        y[:, : min(w_t_all.shape[-2], y.shape[-1])].zero_()
    _call_adn(self, y, x, None, w_t_all, 1.0, 0, grouped=True)


def _expand_decode(self, y, x, w_t_all, add_inputs):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    if not add_inputs:
        y[:, : min(w_t_all.shape[-2], y.shape[-1])].zero_()
    _call_adn(self, y, x, None, w_t_all, 1.0, 0, grouped=False)


def _expand_slice_prefill(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    if not add_inputs:
        y[:, y_offset : y_offset + y_slice_size].zero_()
    _call_adn(self, y, x, None, w_t_all, 1.0, y_offset, grouped=True)


def _expand_slice_decode(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs):
    y = y.view(-1, y.shape[-1])
    x = x.reshape(-1, x.shape[-1])
    if not add_inputs:
        y[:, y_offset : y_offset + y_slice_size].zero_()
    _call_adn(self, y, x, None, w_t_all, 1.0, y_offset, grouped=False)


def build_patches() -> list[tuple[str, object]]:
    """Return replacements for TurboManager after the bootstrap's gates pass.

    Keep split projections separate: fully-sharded TP collectives occur
    between shrink and expand. Resident A/B tensors are read on every call,
    preserving separately deployed, dynamically loaded adapters.
    """
    # Registration happens once here. Forward calls the torch.ops name above.
    try:
        torch._C._dispatch_find_schema_or_throw(f"vllm::{_ADN_OP_NAME}", "")
    except RuntimeError:
        direct_register_custom_op(
            op_name=_ADN_OP_NAME,
            op_func=_adn_dispatch,
            mutates_args=["y"],
            fake_impl=_adn_dispatch_fake,
            dispatch_key="PrivateUse1",
        )
    cls = PunicaWrapperNPU
    return [
        ("vllm.lora.layers.column_parallel_linear._mcp_apply", _wrap_mcp_apply(column_parallel_linear._mcp_apply)),
        (f"{_PUNICA_TARGET}.__init__", _wrap_init(cls.__init__)),
        (f"{_PUNICA_TARGET}._update_base_metadata", _wrap_base_metadata(cls._update_base_metadata)),
        (f"{_PUNICA_TARGET}._update_prefill_metadata", _wrap_prefill_metadata(cls._update_prefill_metadata)),
        (f"{_PUNICA_TARGET}.add_lora_linear", add_lora_linear),
        (f"{_PUNICA_TARGET}.add_lora_logits", add_lora_logits),
        (f"{_PUNICA_TARGET}._shrink_prefill", _shrink_prefill),
        (f"{_PUNICA_TARGET}._shrink_decode", _shrink_decode),
        (f"{_PUNICA_TARGET}._expand_prefill", _expand_prefill),
        (f"{_PUNICA_TARGET}._expand_decode", _expand_decode),
        (f"{_PUNICA_TARGET}._expand_slice_prefill", _expand_slice_prefill),
        (f"{_PUNICA_TARGET}._expand_slice_decode", _expand_slice_decode),
    ]
