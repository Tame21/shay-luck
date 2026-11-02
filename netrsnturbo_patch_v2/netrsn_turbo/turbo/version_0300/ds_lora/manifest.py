# SPDX-License-Identifier: Apache-2.0
"""The LoRA-only Python delta, grouped by upstream module.

Compared source revisions:
    vllm-ascend:       5f828ad638ee4f9b12db87f9d5481f29076d37f9
    vllm-ascend-ds-v2: 5dd159d7d2acf70c0bfcec1a2055ac52c60a13bf (v30_lora_v2)
    vllm:             a00a3544b93edbd66c8eda7285e4468f1202dc4b

Excluded: C++/build/docs changes, DSpark's quantization-only name mapper,
and DeepSeek load_weights' unrelated loaded-parameter bookkeeping fix.
"""

# Each tuple is (upstream module, payload module, existing attributes, additions).
# Payload names are relative to ds_lora and mirror the upstream package path.
# Additions use setattr: a TurboManager implementation may require an original
# attribute to exist before it can propagate replacements by object identity.
PATCH_SPECS = (
    (
        "vllm_ascend.ascend_forward_context",
        "vllm_ascend.ascend_forward_context",
        ("select_moe_comm_method",),
        (),
    ),
    (
        "vllm_ascend.lora.punica_npu",
        "vllm_ascend.lora.punica_npu",
        (
            "PunicaWrapperNPU.__init__",
            "PunicaWrapperNPU.update_metadata",
            "PunicaWrapperNPU.add_shrink",
            "PunicaWrapperNPU.add_expand",
            "PunicaWrapperNPU.add_lora_embedding",
            "PunicaWrapperNPU.add_lora_linear",
            "PunicaWrapperNPU.add_lora_fused_moe",
            "PunicaWrapperNPU.add_lora_logits",
        ),
        ("build_combined_lora_idx", "PunicaWrapperNPU.add_lora"),
    ),
    (
        "vllm_ascend.lora.fused_moe",
        "vllm_ascend.lora.fused_moe",
        (
            "preprocess_lora_indices",
            "postprocess_lora_indices",
            "_recover_moe_lora_routing_allgather",
            "_recover_moe_lora_routing_all2all",
            "moe_lora_apply_w13",
            "moe_lora_apply_w2",
        ),
        (
            "_gather_by_row_permutation",
            "_build_combined_lora_idx_allgather",
            "_build_combined_lora_idx_allgather_torch",
        ),
    ),
    (
        "vllm_ascend.lora.quant_moe",
        "vllm_ascend.lora.quant_moe",
        ("_apply_dynamic_int8_moe_lora",),
        (
            "_lora_aux_stream",
            "_lora_delta_buffer",
            "_lora_overlap_eligible",
            "_fused_w13_epilogue_eligible",
            "_apply_dynamic_int8_moe_lora_overlap_delta",
            "_apply_dynamic_int8_moe_lora_overlap_phase",
        ),
    ),
    (
        "vllm_ascend.ops.fused_moe.routed_experts",
        "vllm_ascend.ops.fused_moe.routed_experts",
        (
            "AscendUnquantizedFusedMoEMethod.apply_gmm1",
            "AscendUnquantizedFusedMoEMethod.apply_gmm2",
        ),
        (),
    ),
    (
        "vllm_ascend.ops.fused_moe.token_dispatcher",
        "vllm_ascend.ops.fused_moe.token_dispatcher",
        ("TokenDispatcherWithAll2AllV.token_dispatch",),
        (),
    ),
    (
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.worker.model_runner_v1",
        ("NPUModelRunner._dummy_run",),
        (),
    ),
    (
        "vllm_ascend.models.deepseek_v4.model",
        "vllm_ascend.models.deepseek_v4.model",
        (),
        (
            "AscendDeepseekV4ForCausalLM.lora_skip_prefixes",
            "AscendDeepseekV4ForCausalLM.hf_to_vllm_mapper",
        ),
    ),
)

# These payload attributes construct a replacement from the loaded function.
# The worker factory edits only the active-LoRA argument in its existing body.
PATCH_FACTORIES = frozenset({
    "vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run",
})

REQUIRED_CUSTOM_OPS = (
    "add_lora",
    "add_lora_shrink",
    "add_lora_expand",
    "add_lora_expand_delta",
    "add_lora_swiglu_quant",
    "build_combined_lora_idx",
)
