# SPDX-License-Identifier: Apache-2.0
"""Install DS LoRA v2 and three-variant AOT together through TurboManager.

The outer router selects this entry instead of the two standalone entries.
Merge both delivery trees into the host package before importing this module.
"""

from netrsn_turbo.turbo.version_0300.ds_lora import apply_lora_patch

apply_lora_patch(with_aot=True)
