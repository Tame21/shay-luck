# SPDX-License-Identifier: Apache-2.0
"""Install the v0.30.0 three-variant LoRA AOT patch through TurboManager."""


def apply_lora_aot_patch() -> bool:
    """Install after the outer Turbo router selects the runtime and patch set."""
    from netrsn_turbo.turbo.version_v0300.lora_aot.patches import build_patches
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm.logger import init_logger

    for path, replacement in build_patches():
        TurboManager.register_patch(path, replacement)
    TurboManager.apply_patches()
    init_logger(__name__).info("[Turbo AOT] v0.30.0 base/base_one/lora patches installed via TurboManager")
    return True


apply_lora_aot_patch()
