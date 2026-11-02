# SPDX-License-Identifier: Apache-2.0
"""Install DS LoRA through TurboManager, including modules imported later.

No NPU module is eagerly imported here. In particular, importing a duplicate
quant_moe module with its registration decorator would create another registry,
and eager model imports would break Ascend's worker initialization order.
"""

from __future__ import annotations

import importlib
import importlib.abc
import re
import sys
from dataclasses import replace
from functools import wraps

from .manifest import PATCH_SPECS
from .ops_backend import initialize_lora_backend

_ENV_MODULE = "vllm_ascend.envs"
_OVERLAP_ENV = "VLLM_ASCEND_LORA_MOE_OVERLAP"
_HOOK_ID = "netrsn_turbo.ds_lora.version_0300"


def check_custom_ops():
    """Initialize the built-in native backend before constructing a wrapper."""
    initialize_lora_backend()


def _checked_init(initializer):
    @wraps(initializer)
    def initialize(self, *args, **kwargs):
        check_custom_ops()
        return initializer(self, *args, **kwargs)

    return initialize


def _resolve_owner(module, path):
    parts = path.split(".")
    owner = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, hook):
        self.loader = loader
        self.hook = hook

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.hook.patch_module(module)

    def __getattr__(self, name):
        # Preserve source/code/resource access used by other import hooks.
        return getattr(self.loader, name)


class _PatchFinder(importlib.abc.MetaPathFinder):
    hook_id = _HOOK_ID

    def __init__(self, manager):
        self.manager = manager
        self.specs = {spec[0]: spec for spec in PATCH_SPECS}
        self.applied = {}
        self.in_progress = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.specs and fullname != _ENV_MODULE:
            return None
        # Delegate to the existing finder chain, including NetrsnTurbo AST
        # hooks. Calling PathFinder directly would bypass those hooks.
        after_self = False
        for finder in tuple(sys.meta_path):
            if finder is self:
                after_self = True
                continue
            if not after_self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PatchLoader(spec.loader, self)
                return spec
        return None

    def patch_module(self, module):
        name = module.__name__
        if self.applied.get(name) is module or name in self.in_progress:
            return
        self.in_progress.add(name)
        try:
            if name == _ENV_MODULE:
                # Same strict 0/1 parser and lazy evaluation as Ascend envs.
                module.env_variables.setdefault(
                    _OVERLAP_ENV, lambda: module._strict_binary_env(_OVERLAP_ENV)
                )
            else:
                self._apply_spec(module, self.specs[name])
            self.applied[name] = module
        finally:
            self.in_progress.remove(name)

    def _apply_spec(self, module, spec):
        _, payload_name, existing, additions = spec
        payload = importlib.import_module(f"{__package__}.{payload_name}")
        # Resolve and validate everything before changing attributes. Do not
        # fabricate missing classes/functions on an incompatible Ascend build.
        targets = []
        for path in existing + additions:
            owner, attribute = _resolve_owner(module, path)
            if path in existing and not hasattr(owner, attribute):
                raise RuntimeError(f"DS LoRA baseline mismatch: missing {module.__name__}.{path}")
            value = getattr(payload, attribute)
            if path == "PunicaWrapperNPU.__init__":
                value = _checked_init(value)
            targets.append((path, owner, attribute, value))

        if payload_name == "fused_moe":
            # reset_lora_indices is unchanged but reads this original constant.
            fields = module._MOE_LORA_INDEX_FIELDS
            if "combined_lora_idx" not in fields:
                module._MOE_LORA_INDEX_FIELDS = (*fields, "combined_lora_idx")
        for path, owner, attribute, value in targets:
            if path in additions:
                # Class additions can inherit a shared default (for example
                # SupportsLoRA.lora_skip_prefixes=[]). Never propagate such
                # a default, or None, into unrelated modules/classes.
                setattr(owner, attribute, value)
            else:
                self.manager.register_patch(f"{module.__name__}.{path}", value)
        self.manager.apply_patches()

        if payload_name == "quant_moe":
            # TurboManager propagates module globals, not references inside
            # frozen dataclasses stored in dictionaries. Preserve validators
            # and all other quantization entries while replacing W8A8's apply.
            key = module.QuantType.W8A8
            original = module._QUANT_MOE_LORA_IMPLS[key]
            module._QUANT_MOE_LORA_IMPLS[key] = replace(
                original, apply=payload._apply_dynamic_int8_moe_lora
            )


def apply_lora_patch():
    """Idempotently install for vLLM 0.30.x in the current process.

    Call again in each spawn worker through NetrsnCore's special_init.py import
    path. Importing a parent process's modules does not patch another process.
    Returns the finder, useful for inspecting which modules have been patched.
    """
    import vllm
    from netrsn_turbo.turbo_manager.turbo_utils import TurboManager

    if re.match(r"^0\.30\.", vllm.__version__) is None:
        raise RuntimeError(
            f"version_0300 DS LoRA targets vLLM 0.30.x, got {vllm.__version__!r}"
        )
    for finder in sys.meta_path:
        if getattr(finder, "hook_id", None) == _HOOK_ID:
            return finder
    names = (_ENV_MODULE, *(spec[0] for spec in PATCH_SPECS))
    initializing = [
        name
        for name in names
        if getattr(getattr(sys.modules.get(name), "__spec__", None), "_initializing", False)
    ]
    if initializing:
        # These loaders were selected before this hook existed, so their
        # completion cannot be intercepted retroactively. Fail instead of
        # silently leaving a partially patched process.
        raise RuntimeError(
            "Install DS LoRA after framework imports finish; modules still initializing: "
            + ", ".join(initializing)
        )
    finder = _PatchFinder(TurboManager)
    sys.meta_path.insert(0, finder)
    try:
        # Patch already imported modules as well as future imports.
        for name in names:
            module = sys.modules.get(name)
            if module is not None:
                finder.patch_module(module)
    except Exception:
        sys.meta_path.remove(finder)
        raise
    return finder
