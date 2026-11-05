# SPDX-License-Identifier: Apache-2.0
"""CPU simulation of TurboManager's documented replacement contract."""

import sys
from types import ModuleType


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
