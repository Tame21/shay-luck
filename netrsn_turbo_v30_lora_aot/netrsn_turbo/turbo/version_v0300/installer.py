# SPDX-License-Identifier: Apache-2.0
"""DS LoRA and optional three-variant AOT registrations from this delivery."""

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


def apply_lora_patch(*, with_aot=False):
    """Register and apply DS LoRA in Netrsn's existing LoRA startup branch.

    Netrsn owns version dispatch, import ordering and duplicate registration.
    Its spawn-worker entry must call this in each worker process as usual.
    with_aot composes the three-variant AOT payload before the single apply.
    """
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm_ascend import _ensure_global_patch

    # Model modules must capture Ascend's factory after platform initialization.
    _ensure_global_patch()

    # ops initializes the MoE pipeline, which imports LoRA helpers itself.
    # Loading lora.fused_moe first re-enters it before prepare_lora_indices exists.
    import vllm_ascend.ops  # noqa: F401

    from vllm_ascend import envs
    from vllm_ascend.lora import fused_moe, punica_npu, quant_moe
    from vllm_ascend.models.deepseek_v4 import model as deepseek_model
    from vllm_ascend.patch.platform import patch_fused_moe as moe_factory_patch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    # The platform patch only refreshes preloaded vllm.model_executor.models
    # aliases; an early Ascend model import can retain the original factory.
    # Assign only this binding: manager propagation would also replace the
    # platform wrapper's _original_FusedMoE delegate and cause recursion.
    deepseek_model.FusedMoEFactory = moe_factory_patch._ascend_FusedMoE

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
    if with_aot:
        from vllm.compilation import decorators
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        from .vllm.compilation import decorators as aot_decorators
        from .vllm.compilation import wrapper as aot_wrapper

        # Rewrite the native inference-mode method before adding AOT wrappers.
        # Register only the final callable, so both patches own one target.
        dummy_run = worker_patch.wrap_dummy_run(dummy_run)

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

    deepseek_model.AscendDeepseekV4ForCausalLM.lora_skip_prefixes = model_patch.lora_skip_prefixes
    deepseek_model.AscendDeepseekV4ForCausalLM.hf_to_vllm_mapper = model_patch.hf_to_vllm_mapper

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

    if with_aot:
        TurboManager.register_patch(
            "vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__",
            aot_wrapper.wrap_init(TorchCompileWithNoGuardsWrapper.__init__),
        )
        TurboManager.register_patch(
            "vllm.compilation.decorators._try_load_aot_compiled_fn",
            aot_decorators.wrap_try_load_aot_compiled_fn(decorators._try_load_aot_compiled_fn),
        )
        TurboManager.register_patch(
            "vllm_ascend.worker.model_runner_v1.NPUModelRunner.maybe_dummy_run_with_lora",
            worker_patch.wrap_maybe_dummy_run_with_lora(NPUModelRunner.maybe_dummy_run_with_lora),
        )
        TurboManager.register_patch(
            "vllm_ascend.worker.model_runner_v1.NPUModelRunner.capture_model",
            worker_patch.wrap_capture_model(NPUModelRunner.capture_model),
        )

    TurboManager.apply_patches()

    # Module-global propagation does not update a frozen dataclass in a dict.
    # Preserve the W8A8 validator and other fields while replacing its callable.
    key = quant_moe.QuantType.W8A8
    quant_moe._QUANT_MOE_LORA_IMPLS[key] = replace(
        quant_moe._QUANT_MOE_LORA_IMPLS[key],
        apply=quant_patch._apply_dynamic_int8_moe_lora,
    )
