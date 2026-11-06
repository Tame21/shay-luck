# SPDX-License-Identifier: Apache-2.0
"""LoRA environment addition from vllm_ascend/envs.py."""

OVERLAP_ENV = "VLLM_ASCEND_LORA_MOE_OVERLAP"


def apply_env_patch(module):
    """Extend the loaded upstream registry without evaluating any getter."""
    module.env_variables.setdefault(OVERLAP_ENV, lambda: module._strict_binary_env(OVERLAP_ENV))
