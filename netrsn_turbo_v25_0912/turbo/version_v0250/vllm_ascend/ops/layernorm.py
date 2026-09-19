# SPDX-License-Identifier: Apache-2.0

"""Keep gated RMSNorm's process-local Triton state out of AOT artifacts."""

import torch
from vllm_ascend.ops import layernorm


# A newer Ascend build, or a previous import, may already own this operator.
# Reuse it so loading Turbo does not replace its kernels or fake implementation.
if hasattr(torch.ops.vllm_ascend, "rms_norm_gated"):
    _rms_norm_gated = torch.ops.vllm_ascend.rms_norm_gated.default
else:

    @torch.library.custom_op("vllm_ascend::rms_norm_gated", mutates_args=())
    def _rms_norm_gated(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None,
        eps: float,
        group_size: int | None,
        norm_before_gate: bool,
    ) -> torch.Tensor:
        return layernorm.LayerNormFn.apply(
            x, weight, bias, z, eps, group_size, norm_before_gate, True
        )

    @_rms_norm_gated.register_fake
    def _rms_norm_gated_fake(x, weight, bias, z, eps, group_size, norm_before_gate):
        return torch.empty_like(x, memory_format=torch.contiguous_format)


def forward_oot(self, x, z=None):
    return _rms_norm_gated(
        x,
        self.weight,
        self.bias,
        z,
        self.eps,
        self.group_size,
        self.norm_before_gate,
    )
