# SPDX-License-Identifier: Apache-2.0
"""Install the v0.30.0 three-variant LoRA AOT patch through TurboManager."""


def apply_lora_aot_patch() -> bool:
    """Install after the outer Turbo router selects the runtime and patch set."""
    from netrsn_turbo.turbo.version_v0300.vllm.compilation import decorators as decorators_patch
    from netrsn_turbo.turbo.version_v0300.vllm.compilation import wrapper as wrapper_patch
    from netrsn_turbo.turbo.version_v0300.vllm_ascend.worker import model_runner_v1 as runner_patch
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm.compilation import decorators
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.logger import init_logger
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    TurboManager.register_patch(
        "vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__",
        wrapper_patch.wrap_init(TorchCompileWithNoGuardsWrapper.__init__),
    )
    TurboManager.register_patch(
        "vllm.compilation.decorators._try_load_aot_compiled_fn",
        decorators_patch.wrap_try_load_aot_compiled_fn(decorators._try_load_aot_compiled_fn),
    )
    TurboManager.register_patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run",
        runner_patch.wrap_dummy_run(NPUModelRunner._dummy_run),
    )
    TurboManager.register_patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner.maybe_dummy_run_with_lora",
        runner_patch.wrap_maybe_dummy_run_with_lora(NPUModelRunner.maybe_dummy_run_with_lora),
    )
    TurboManager.register_patch(
        "vllm_ascend.worker.model_runner_v1.NPUModelRunner.capture_model",
        runner_patch.wrap_capture_model(NPUModelRunner.capture_model),
    )
    TurboManager.apply_patches()
    init_logger(__name__).info("[Turbo AOT] v0.30.0 base/base_one/lora patches installed via TurboManager")
    return True


apply_lora_aot_patch()
