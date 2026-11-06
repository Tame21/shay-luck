# SPDX-License-Identifier: Apache-2.0

"""DS LoRA changes from vllm_ascend/ascend_forward_context.py; installed by TurboManager.

Only changed/new functions are carried here. Importing this module does not
replace upstream symbols or execute registration decorators.
"""

from __future__ import annotations

from netrsn_turbo.turbo_ops._logging import log_once
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, is_moe_model
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    _MOE_COMM_SELECTORS,
    get_mc2_tokens_capacity,
)


def select_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
    draft_moe_quant_type: QuantType = QuantType.NONE,
) -> MoECommType | None:
    """Select the MoE communication method from the active hardware policy,
    parallel settings, and token count.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode.
        draft_moe_quant_type (QuantType): The draft model's MoE quantization
            type, used on A5 to decide whether the draft can use mega moe.
    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None

    mc2_tokens_capacity = get_mc2_tokens_capacity()
    moe_comm_policy = get_current_hardware_profile().moe_comm_policy
    lora_config = getattr(vllm_config, "lora_config", None)
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif lora_config is not None and vllm_config.parallel_config.enable_expert_parallel:
        if get_ascend_device_type() == AscendDeviceType.A2:
            # On A2 the only MC2 alternative in the menu is AllGather, and
            # the AllGather dispatcher already keeps routed activations in
            # BF16/FP16 for LoRA batches (quant policy) with routing
            # metadata (expanded_row_idx + topk_ids) that
            # _recover_moe_lora_routing_allgather maps to local experts.
            # AlltoAll dispatch/combine, by contrast, adds two latency-bound
            # collectives per MoE layer that dominate small-batch decode.
            moe_comm_type = MoECommType.ALLGATHER
        else:
            # A3/A5 keep AlltoAll: their selectors would otherwise pick
            # MC2/FusedMC2, and Ascend MoE LoRA cannot patch FusedMC2
            # (dispatch_ffn_combine/mega_moe is a single fused C++ op).
            # This covers both normal model forward and _dummy_run during
            # profile_run.
            moe_comm_type = MoECommType.ALLTOALL
    elif moe_comm_policy is MoECommPolicy.ALLGATHER:
        moe_comm_type = MoECommType.ALLGATHER
    else:
        moe_comm_type = _MOE_COMM_SELECTORS[moe_comm_policy](
            num_tokens,
            vllm_config,
            mc2_tokens_capacity,
            is_draft_model,
            draft_moe_quant_type,
        )
    if lora_config is not None:
        log_once(
            "moe_comm." + moe_comm_type.name,
            "[DS LoRA] rewritten select_moe_comm_method selected=%s",
            moe_comm_type.name,
        )
    logger.debug(
        "MoE comm method selected: policy=%s, method=%s, num_tokens=%d, mc2_capacity=%s, "
        "is_draft_model=%s, draft_moe_quant_type=%r",
        moe_comm_policy,
        moe_comm_type,
        num_tokens,
        mc2_tokens_capacity,
        is_draft_model,
        draft_moe_quant_type,
    )
    return moe_comm_type
