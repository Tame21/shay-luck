# SPDX-License-Identifier: Apache-2.0
"""Install the v0.23 310P LoRA adapter through TurboManager."""

from __future__ import annotations

import sys
from functools import wraps

_STATE_ATTR = "_netrsn_turbo_lora_310p_state"
_OWNER = "adn"
_ADAPT_TARGET = "vllm_ascend.utils.adapt_patch"


def _resolve_target(path):
    """Resolve an already imported module/class attribute without importing workers."""
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        obj = sys.modules.get(".".join(parts[:end]))
        if obj is not None:
            for name in parts[end:]:
                obj = getattr(obj, name)
            return obj
    raise RuntimeError(f"Turbo LoRA target is not loaded: {path}")


def _check_native_conflicts(cls):
    flags = ("_ascend_lora_prefill_310_patch", "_ascend_lora_adn_310_patch")
    if any(getattr(cls, name, False) for name in flags):
        raise RuntimeError(
            "Turbo LoRA cannot be combined with the previous handwritten VA LoRA patches. "
            "Disable the native column_parallel/prefill/adn LoRA imports and restart."
        )
    module = sys.modules.get("vllm.lora.layers.column_parallel_linear")
    if module is not None and getattr(module._mcp_apply, "_ascend_lora_column_310_patch", False):
        raise RuntimeError("Disable the previous native column-parallel LoRA patch and restart.")


class _PatchState:
    def __init__(self, cls, manager):
        self.owner = _OWNER
        self.cls = cls
        self.manager = manager
        self.replacements = ()
        self.applying = False
        self.failure = None

    def ensure_applied(self):
        if self.failure is not None:
            raise RuntimeError(
                "Previous Turbo LoRA installation failed; restart after correcting it."
            ) from self.failure
        if self.applying:
            return
        _check_native_conflicts(self.cls)
        missing = [path for path, replacement in self.replacements if _resolve_target(path) is not replacement]
        if not missing:
            return
        self.applying = True
        try:
            self.manager.apply_patches()
            missing = [path for path, replacement in self.replacements if _resolve_target(path) is not replacement]
            if missing:
                raise RuntimeError(
                    "TurboManager did not install the selected LoRA replacements: "
                    + ", ".join(missing)
                    + ". Disable conflicting older Turbo LoRA registrations."
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
        # Native platform/worker patches are imported after Turbo. Reapply the
        # same registered objects, without constructing another wrapper layer.
        state.ensure_applied()
        return result

    return adapt_patch


def apply_lora_patch() -> bool:
    """Enable this backend explicitly; runtime adapter loading needs no package-id env."""
    from packaging.version import Version

    import vllm

    if Version(vllm.__version__).release[:2] != (0, 23):
        return False

    from vllm_ascend.utils import is_310p

    if not is_310p():
        return False

    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    state = getattr(PunicaWrapperNPU, _STATE_ATTR, None)
    if state is not None:
        if state.owner != _OWNER:
            raise RuntimeError(
                f"LoRA backend {state.owner!r} is already selected; cannot also enable {_OWNER!r}. "
                "Import only one Turbo LoRA entry and restart to switch backends."
            )
        state.ensure_applied()
        return True

    _check_native_conflicts(PunicaWrapperNPU)
    # Framework and hardware checks precede imports that can load ADN or
    # register operators. Only the APIs documented in patch_mesim.md are used.
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager

    from .patches import build_patches

    state = _PatchState(PunicaWrapperNPU, TurboManager)
    setattr(PunicaWrapperNPU, _STATE_ATTR, state)
    try:
        replacements = list(build_patches())
        replacements.append((_ADAPT_TARGET, _wrap_adapt_patch(_resolve_target(_ADAPT_TARGET), state)))
        if len({path for path, _ in replacements}) != len(replacements):
            raise RuntimeError("Duplicate Turbo LoRA target registration")
        state.replacements = tuple(replacements)
        for path, replacement in state.replacements:
            TurboManager.register_patch(path, replacement)
        state.ensure_applied()
    except Exception as error:
        state.failure = error
        raise

    from vllm.logger import logger

    logger.info("[Turbo LoRA] backend=%s installed on vLLM 0.23 / Ascend310P via TurboManager", _OWNER)
    return True
