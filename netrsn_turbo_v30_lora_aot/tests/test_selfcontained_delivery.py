# SPDX-License-Identifier: Apache-2.0
"""Import the joint entry from one delivery, without sibling patch packages."""

import ast
import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "netrsn_turbo"


def test_internal_imports_resolve_inside_the_delivery():
    """Only the private host's TurboManager is intentionally supplied externally."""
    external = {"netrsn_turbo.turbo_manager.turbo_utils"}
    for source in PACKAGE.rglob("*.py"):
        relative = source.relative_to(ROOT).with_suffix("")
        package = ".".join(relative.parts[:-1])
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                modules = [importlib.util.resolve_name(module, package) if node.level else module]
            else:
                continue
            for module in modules:
                if not module.startswith("netrsn_turbo") or module in external:
                    continue
                path = ROOT.joinpath(*module.split("."))
                assert path.is_dir() or path.with_suffix(".py").is_file(), (source, module)


def test_joint_entry_imports_from_an_isolated_copy(tmp_path):
    shutil.copytree(PACKAGE, tmp_path / "netrsn_turbo", ignore=shutil.ignore_patterns("__pycache__"))
    script = textwrap.dedent("""\
        import importlib
        import importlib.abc
        import pathlib
        import sys

        root = pathlib.Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(root))

        class BlockExternalDelivery(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if '.version_0300' in fullname or 'netrsnturbo_patch_v2' in fullname:
                    raise AssertionError('legacy delivery dependency: ' + fullname)
                if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'ops_repo'):
                    raise AssertionError('unexpected eager framework import: ' + fullname)
                return None

        sys.meta_path.insert(0, BlockExternalDelivery())
        prefix = 'netrsn_turbo.turbo.version_v0300'
        installer = importlib.import_module(prefix + '.installer')
        worker = importlib.import_module(prefix + '.vllm_ascend.worker.model_runner_v1')
        backend = importlib.import_module(prefix + '.vllm_ascend.lora.ops_backend')
        ops = importlib.import_module('netrsn_turbo.turbo_ops')
        assert callable(worker._dummy_run)
        assert callable(worker.wrap_dummy_run)
        assert callable(ops.add_lora)
        assert backend._BACKEND is None

        calls = []
        installer.apply_lora_patch = lambda **kwargs: calls.append(kwargs)
        entry = importlib.import_module('netrsn_turbo.turbo_manager.version_v0300.turbo_ds_lora_aot')
        assert calls == [{'with_aot': True}]
        assert entry.apply_lora_patch is installer.apply_lora_patch
        for name, module in tuple(sys.modules.items()):
            if name.startswith('netrsn_turbo') and getattr(module, '__file__', None):
                pathlib.Path(module.__file__).resolve().relative_to(root)
        assert 'netrsn_turbo.turbo_manager.version_v0300.turbo_lora_aot' not in sys.modules
        print('isolated combined entry resolved')
        """)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated combined entry resolved" in result.stdout
