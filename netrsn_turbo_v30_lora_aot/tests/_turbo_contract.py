# SPDX-License-Identifier: Apache-2.0
"""CPU simulation of TurboManager's documented replacement contract."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def load_payloads(monkeypatch):
    """Import the delivered package hierarchy without the private host package."""
    root = Path(__file__).resolve().parents[1]
    prefix = "netrsn_turbo.turbo.version_v0300"
    for name in tuple(sys.modules):
        if name == "netrsn_turbo" or name.startswith("netrsn_turbo."):
            monkeypatch.delitem(sys.modules, name)
    for name in ("netrsn_turbo", "netrsn_turbo.turbo"):
        module = ModuleType(name)
        module.__path__ = [str(root / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, module)
    for suffix in (
        "",
        ".vllm",
        ".vllm.compilation",
        ".vllm.compilation.decorators",
        ".vllm.compilation.wrapper",
        ".vllm_ascend",
        ".vllm_ascend.worker",
        ".vllm_ascend.worker.model_runner_v1",
    ):
        name = prefix + suffix
        # Record each import slot for monkeypatch teardown before importing it.
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    return SimpleNamespace(
        wrapper=importlib.import_module(prefix + ".vllm.compilation.wrapper"),
        decorators=importlib.import_module(prefix + ".vllm.compilation.decorators"),
        runner=importlib.import_module(prefix + ".vllm_ascend.worker.model_runner_v1"),
    )


def apply_patches(monkeypatch, replacements):
    for path, replacement in replacements:
        parts = path.split(".")
        for end in range(len(parts), 0, -1):
            target = sys.modules.get(".".join(parts[:end]))
            if target is not None:
                for name in parts[end:-1]:
                    target = getattr(target, name)
                break
        else:
            raise AssertionError(f"Missing target module: {path}")
        attr = parts[-1]
        original = getattr(target, attr)
        monkeypatch.setattr(target, attr, replacement)
        if original is replacement:
            continue
        for module in tuple(sys.modules.values()):
            if not isinstance(module, ModuleType):
                continue
            for name, value in tuple(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, name, replacement)
