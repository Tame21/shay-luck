# SPDX-License-Identifier: Apache-2.0
"""Explicit DS LoRA registrations for NetrsnTurbo's version_0300 entry."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps

from .vllm_ascend.envs import apply_env_patch
from .vllm_ascend.lora.ops_backend import initialize_lora_backend


def check_custom_ops():
    """Initialize the built-in native backend before constructing a wrapper."""
    initialize_lora_backend()


def _checked_init(initializer):
    @wraps(initializer)
    def initialize(self, *args, **kwargs):
        check_custom_ops()
        return initializer(self, *args, **kwargs)

    return initialize


def apply_lora_patch():
    """Register and apply DS LoRA in Netrsn's existing LoRA startup branch.

    Netrsn owns version dispatch, import ordering and duplicate registration.
    Its spawn-worker entry must call this in each worker process as usual.
    """
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager

    # ops initializes the MoE pipeline, which imports LoRA helpers itself.
    # Loading lora.fused_moe first re-enters it before prepare_lora_indices exists.
    import vllm_ascend.ops  # noqa: F401

    from vllm_ascend import envs
    from vllm_ascend.lora import fused_moe, punica_npu, quant_moe
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    from .vllm_ascend import ascend_forward_context as forward_patch
    from .vllm_ascend.lora import fused_moe as fused_patch
    from .vllm_ascend.lora import punica_npu as punica_patch
    from .vllm_ascend.lora import quant_moe as quant_patch
    from .vllm_ascend.models.deepseek_v4 import model as model_patch
    from .vllm_ascend.ops.fused_moe import routed_experts as routed_patch
    from .vllm_ascend.ops.fused_moe import token_dispatcher as dispatcher_patch
    from .vllm_ascend.worker import model_runner_v1 as worker_patch

    # This payload is a factory: retain the loaded method's body and globals.
    dummy_run = worker_patch._dummy_run(NPUModelRunner._dummy_run)

    # New attributes have no original implementation to replace or propagate.
    punica_npu.build_combined_lora_idx = punica_patch.build_combined_lora_idx
    punica_npu.PunicaWrapperNPU.add_lora = punica_patch.add_lora

    fused_moe._gather_by_row_permutation = fused_patch._gather_by_row_permutation
    fused_moe._build_combined_lora_idx_allgather = fused_patch._build_combined_lora_idx_allgather
    fused_moe._build_combined_lora_idx_allgather_torch = fused_patch._build_combined_lora_idx_allgather_torch

    quant_moe._lora_aux_stream = quant_patch._lora_aux_stream
    quant_moe._lora_delta_buffer = quant_patch._lora_delta_buffer
    quant_moe._lora_overlap_eligible = quant_patch._lora_overlap_eligible
    quant_moe._fused_w13_epilogue_eligible = quant_patch._fused_w13_epilogue_eligible
    quant_moe._apply_dynamic_int8_moe_lora_overlap_delta = quant_patch._apply_dynamic_int8_moe_lora_overlap_delta
    quant_moe._apply_dynamic_int8_moe_lora_overlap_phase = quant_patch._apply_dynamic_int8_moe_lora_overlap_phase

    AscendDeepseekV4ForCausalLM.lora_skip_prefixes = model_patch.lora_skip_prefixes
    AscendDeepseekV4ForCausalLM.hf_to_vllm_mapper = model_patch.hf_to_vllm_mapper

    apply_env_patch(envs)
    # The unchanged upstream reset_lora_indices reads this tuple.
    fused_moe._MOE_LORA_INDEX_FIELDS = (*fused_moe._MOE_LORA_INDEX_FIELDS, "combined_lora_idx")

    TurboManager.register_patch(
        "vllm_ascend.ascend_forward_context.select_moe_comm_method",
        forward_patch.select_moe_comm_method,
    )

    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.__init__",
        _checked_init(punica_patch.__init__),
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.update_metadata",
        punica_patch.update_metadata,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_shrink",
        punica_patch.add_shrink,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_expand",
        punica_patch.add_expand,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_embedding",
        punica_patch.add_lora_embedding,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_linear",
        punica_patch.add_lora_linear,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_fused_moe",
        punica_patch.add_lora_fused_moe,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.add_lora_logits",
        punica_patch.add_lora_logits,
    )

    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe.preprocess_lora_indices",
        fused_patch.preprocess_lora_indices,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe.postprocess_lora_indices",
        fused_patch.postprocess_lora_indices,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe._recover_moe_lora_routing_allgather",
        fused_patch._recover_moe_lora_routing_allgather,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe._recover_moe_lora_routing_all2all",
        fused_patch._recover_moe_lora_routing_all2all,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe.moe_lora_apply_w13",
        fused_patch.moe_lora_apply_w13,
    )
    TurboManager.register_patch(
        "vllm_ascend.lora.fused_moe.moe_lora_apply_w2",
        fused_patch.moe_lora_apply_w2,
    )

    TurboManager.register_patch(
        "vllm_ascend.lora.quant_moe._apply_dynamic_int8_moe_lora",
        quant_patch._apply_dynamic_int8_moe_lora,
    )

    TurboManager.register_patch(
        "vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod.apply_gmm1",
        routed_patch.apply_gmm1,
    )
    TurboManager.register_patch(
        "vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod.apply_gmm2",
        routed_patch.apply_gmm2,
    )
    TurboManager.register_patch(
        "vllm_ascend.ops.fused_moe.token_dispatcher.TokenDispatcherWithAll2AllV.token_dispatch",
        dispatcher_patch.token_dispatch,
    )
    TurboManager.register_patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run",
        dummy_run,
    )

    TurboManager.apply_patches()

    # Module-global propagation does not update a frozen dataclass in a dict.
    # Preserve the W8A8 validator and other fields while replacing its callable.
    key = quant_moe.QuantType.W8A8
    quant_moe._QUANT_MOE_LORA_IMPLS[key] = replace(
        quant_moe._QUANT_MOE_LORA_IMPLS[key],
        apply=quant_patch._apply_dynamic_int8_moe_lora,
    )
