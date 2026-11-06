# SPDX-License-Identifier: Apache-2.0
"""Install DS LoRA v2 and three-variant AOT together through TurboManager.

The outer router selects this entry instead of the two standalone entries.
All DS and AOT payloads are included in this delivery under version_v0300.
"""

from netrsn_turbo.turbo.version_v0300.installer import apply_lora_patch

apply_lora_patch(with_aot=True)
