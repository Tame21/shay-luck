# SPDX-License-Identifier: Apache-2.0
"""Install the v0.30.0 three-variant LoRA AOT patch through TurboManager."""

from functools import wraps

_STATE_ATTR = "_netrsn_turbo_lora_aot_v0300_state"
_ADAPT_TARGET = "vllm_ascend.utils.adapt_patch"
_NATIVE_WRAPPER_MARKER = "_ascend_lora_aot_installed"
_NATIVE_RUNNER_MARKER = "_ascend_lora_aot_graph_hooks_installed"


class _PatchState:
    def __init__(self, manager):
        self.manager = manager
        self.replacements = ()
        self.registered = False
        self.applying = False

    def apply(self):
        if self.applying:
            return
        self.applying = True
        try:
            if not self.registered:
                for path, replacement in self.replacements:
                    self.manager.register_patch(path, replacement)
                self.registered = True
            self.manager.apply_patches()
        finally:
            self.applying = False


def _wrap_adapt_patch(original, state):
    @wraps(original)
    def adapt_patch(*args, **kwargs):
        result = original(*args, **kwargs)
        state.apply()
        return result

    return adapt_patch


def apply_lora_aot_patch() -> bool:
    """Install after the outer Turbo router selects the runtime and patch set."""
    from netrsn_turbo.turbo.version_v0300.lora_aot.patches import build_patches
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm_ascend import utils
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    state = getattr(TorchCompileWithNoGuardsWrapper, _STATE_ATTR, None)
    if state is None:
        state = _PatchState(TurboManager)
        replacements = list(build_patches())
        replacements.append((_ADAPT_TARGET, _wrap_adapt_patch(utils.adapt_patch, state)))
        state.replacements = tuple(replacements)
        setattr(TorchCompileWithNoGuardsWrapper, _STATE_ATTR, state)
    state.apply()
    # Let the equivalent native patch skip installation if imported later.
    setattr(TorchCompileWithNoGuardsWrapper, _NATIVE_WRAPPER_MARKER, True)
    setattr(NPUModelRunner, _NATIVE_RUNNER_MARKER, True)

    from vllm.logger import init_logger

    init_logger(__name__).info("[Turbo AOT] v0.30.0 base/base_one/lora patches installed via TurboManager")
    return True


apply_lora_aot_patch()
