# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
"""Only the DeepSeek V4 adapter-name mapping from the DS branch."""

import re

from vllm.model_executor.models.utils import WeightsMapper

# The drafter is not LoRA-wrapped in vLLM v1. Its tensors must not reach the
# target model's unexpected-module validation.
lora_skip_prefixes = ["mtp."]
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"rotary_emb\.inv_freq"): None,
        re.compile(r"^(?!model)(?!mtp\.)"): "model.",
    },
    orig_to_new_substr={
        ".w1.": ".gate_proj.",
        ".w2.": ".down_proj.",
        ".w3.": ".up_proj.",
        "model.head.": "lm_head.",
        "model.lm_head.": "lm_head.",
        "embed.": "embed_tokens.",
        ".attn.": ".self_attn.",
        ".ffn.": ".mlp.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".attn_norm.": ".input_layernorm.",
        ".gate.bias": ".gate.e_score_correction_bias",
    },
    orig_to_new_suffix={".scale": ".weight_scale"},
)
