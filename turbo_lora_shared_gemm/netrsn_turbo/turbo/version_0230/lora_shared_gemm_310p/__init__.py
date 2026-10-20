# SPDX-License-Identifier: Apache-2.0
"""Explicit entry point for the v0.23 Ascend 310P LoRA adapter."""

from .bootstrap import apply_lora_patch

__all__ = ["apply_lora_patch"]
