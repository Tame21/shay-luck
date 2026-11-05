# SPDX-License-Identifier: Apache-2.0
"""Install the v0.30.0 three-variant LoRA AOT patch through TurboManager."""

import sys
from functools import wraps

_STATE_ATTR = "_netrsn_turbo_lora_aot_v0300_state"
_ADAPT_TARGET = "vllm_ascend.utils.adapt_patch"
_NATIVE_WRAPPER_MARKER = "_ascend_lora_aot_installed"
_NATIVE_RUNNER_MARKER = "_ascend_lora_aot_graph_hooks_installed"


def _resolve_target(path):
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        obj = sys.modules.get(".".join(parts[:end]))
        if obj is not None:
            for name in parts[end:]:
                obj = getattr(obj, name)
            return obj
    raise RuntimeError(f"Turbo AOT target is not loaded: {path}")


def _check_legacy_conflicts(wrapper_cls):
    for name in (
        "_ascend_lora_graph_patch_installed",
        "_external_lora_acl_graph_v23_patch",
    ):
        if getattr(wrapper_cls, name, False):
            raise RuntimeError(
                "The v0.30.0 LoRA AOT patch cannot be combined with the older "
                "LoRA ACL graph patch. Disable its Turbo/native entry and restart."
            )


class _PatchState:
    def __init__(self, wrapper_cls, runner_cls, manager):
        self.wrapper_cls = wrapper_cls
        self.runner_cls = runner_cls
        self.manager = manager
        self.replacements = ()
        self.applying = False
        self.failure = None

    def ensure_applied(self):
        if self.failure is not None:
            raise RuntimeError(
                "Previous Turbo AOT installation failed; correct the cause and restart."
            ) from self.failure
        if self.applying:
            return
        _check_legacy_conflicts(self.wrapper_cls)
        if all(_resolve_target(path) is replacement for path, replacement in self.replacements):
            return
        self.applying = True
        try:
            self.manager.apply_patches()
            missing = [path for path, replacement in self.replacements if _resolve_target(path) is not replacement]
            if missing:
                raise RuntimeError(
                    "TurboManager did not install the LoRA AOT replacements: "
                    + ", ".join(missing)
                    + ". Check competing Turbo registrations."
                )
        except Exception as error:
            self.failure = error
            raise
        finally:
            self.applying = False


def _wrap_adapt_patch(original, state):
    @wraps(original)
    def adapt_patch(*args, **kwargs):
        result = original(*args, **kwargs)
        state.ensure_applied()
        return result

    return adapt_patch


def apply_lora_aot_patch() -> bool:
    """Register before model creation in every worker, including runtime-LoRA serving."""
    from packaging.version import Version

    import vllm

    if Version(vllm.__version__).release[:3] != (0, 30, 0):
        return False

    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

    if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
        return False

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    state = getattr(TorchCompileWithNoGuardsWrapper, _STATE_ATTR, None)
    if state is not None:
        state.ensure_applied()
        return True

    _check_legacy_conflicts(TorchCompileWithNoGuardsWrapper)
    if getattr(TorchCompileWithNoGuardsWrapper, _NATIVE_WRAPPER_MARKER, False) or getattr(
        NPUModelRunner, _NATIVE_RUNNER_MARKER, False
    ):
        raise RuntimeError(
            "The native patch_lora_aot is already installed. Disable its import in "
            "vllm_ascend.patch.worker and restart with the Turbo AOT entry."
        )

    from netrsn_turbo.turbo.version_v0300.lora_aot.patches import build_patches
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager
    from vllm_ascend import utils

    state = _PatchState(TorchCompileWithNoGuardsWrapper, NPUModelRunner, TurboManager)
    setattr(TorchCompileWithNoGuardsWrapper, _STATE_ATTR, state)
    try:
        replacements = list(build_patches())
        replacements.append((_ADAPT_TARGET, _wrap_adapt_patch(utils.adapt_patch, state)))
        if len({path for path, _ in replacements}) != len(replacements):
            raise RuntimeError("Duplicate Turbo AOT target registration")
        state.replacements = tuple(replacements)
        for path, replacement in state.replacements:
            TurboManager.register_patch(path, replacement)
        state.ensure_applied()
        # The equivalent native patch can be imported later by adapt_patch().
        # Its own idempotency checks must then leave the Turbo installation intact.
        setattr(TorchCompileWithNoGuardsWrapper, _NATIVE_WRAPPER_MARKER, True)
        setattr(NPUModelRunner, _NATIVE_RUNNER_MARKER, True)
    except Exception as error:
        state.failure = error
        raise

    from vllm.logger import init_logger

    init_logger(__name__).info("[Turbo AOT] v0.30.0 base/base_one/lora patches installed via TurboManager")
    return True


apply_lora_aot_patch()
