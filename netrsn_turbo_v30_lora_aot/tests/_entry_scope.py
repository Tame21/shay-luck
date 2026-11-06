# SPDX-License-Identifier: Apache-2.0
"""Read test expectations from the explicit manager entry, without importing it."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "netrsn_turbo/turbo/version_v0300"
ENTRY = ROOT / "netrsn_turbo/turbo_manager/version_v0300/turbo_ds_lora_aot.py"
PAYLOAD_PREFIX = "netrsn_turbo.turbo.version_v0300."


def entry_function():
    tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_ds_lora_aot_patch"
    )


def registration_targets():
    targets = []
    for node in sorted(ast.walk(entry_function()), key=lambda item: getattr(item, "lineno", 0)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register_patch":
            assert isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
            targets.append(node.args[0].value)
    return tuple(targets)


def _parts(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_parts(node.value), node.attr]
    return []


def ds_patch_specs():
    """Group actual DS registrations and direct payload assignments for fixtures."""
    entry = entry_function()
    aliases = {}
    for node in ast.walk(entry):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(PAYLOAD_PREFIX):
            for imported in node.names:
                module = (node.module + "." + imported.name).removeprefix(PAYLOAD_PREFIX)
                if PAYLOAD.joinpath(*module.split(".")).with_suffix(".py").is_file():
                    aliases[imported.asname or imported.name] = module

    groups = {}
    for target in registration_targets():
        # The two additional worker hooks and compiler hooks belong to AOT.
        if not target.startswith("vllm_ascend.") or target.endswith((".maybe_dummy_run_with_lora", ".capture_model")):
            continue
        module = max((name for name in aliases.values() if target.startswith(name + ".")), key=len)
        existing, _ = groups.setdefault(module, ([], []))
        existing.append(target.removeprefix(module + "."))

    for node in ast.walk(entry):
        if not isinstance(node, ast.Assign):
            continue
        value = _parts(node.value)
        if not value or value[0] not in aliases:
            continue
        module = aliases[value[0]]
        if not module.startswith("vllm_ascend."):
            continue
        for target in node.targets:
            parts = _parts(target)
            if len(parts) > 1:
                _, additions = groups.setdefault(module, ([], []))
                additions.append(".".join(parts[1:]))
    return tuple(
        (module, module, tuple(existing), tuple(additions)) for module, (existing, additions) in groups.items()
    )
