# SPDX-License-Identifier: Apache-2.0

"""Install the gated RMSNorm AOT boundary before model initialization."""

from netrsn_turbo.turbo_manager.turbo_utils import TurboManager


def apply_layernorm_aot_patch() -> None:
    from vllm_ascend.ops.layernorm import AscendRMSNormGated

    if getattr(AscendRMSNormGated, "_ascend_gated_rmsnorm_aot_patch_installed", False):
        return

    from netrsn_turbo.turbo.version_v0250.vllm_ascend.ops import (
        layernorm as layernorm_patch,
    )

    TurboManager.register_patch(
        "vllm_ascend.ops.layernorm.AscendRMSNormGated.forward_oot",
        layernorm_patch.forward_oot,
    )
    TurboManager.apply_patches()
    AscendRMSNormGated._ascend_gated_rmsnorm_aot_patch_installed = True
