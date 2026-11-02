# SPDX-License-Identifier: Apache-2.0
"""DeepSeek LoRA patches for vLLM 0.30.x; no patching on package import."""

from .installer import apply_lora_patch

__all__ = ["apply_lora_patch"]
