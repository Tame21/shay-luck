# SPDX-License-Identifier: Apache-2.0
"""Reproduce a stale model-local MoE factory with Python's real import loader.

The reduced packages retain the relevant imports and platform-patch rebinding.
They do not exercise Torch, NPU kernels or full framework initialization.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "netrsn_turbo/turbo_manager/version_v0300/turbo_ds_lora_aot.py"

MODULES = {
    "netrsn_turbo/__init__.py": "",
    "netrsn_turbo/turbo_manager/__init__.py": "",
    "netrsn_turbo/turbo_manager/turbo_utils.py": "class TurboManager:\n    pass\n",
    "vllm/__init__.py": "",
    "vllm/model_executor/__init__.py": "",
    "vllm/model_executor/layers/__init__.py": "",
    "vllm/model_executor/layers/fused_moe/__init__.py": "from .layer import FusedMoEFactory\n",
    "vllm/model_executor/layers/fused_moe/layer.py": (
        "from types import SimpleNamespace\n"
        "calls = []\n"
        "def upstream_router():\n"
        "    raise AttributeError(\"'_OpNamespace' '_moe_C' object has no attribute 'topk_softplus_sqrt'\")\n"
        "def FusedMoEFactory(router=None):\n"
        "    calls.append(router)\n"
        "    return SimpleNamespace(router=upstream_router if router is None else router)\n"
    ),
    "vllm/model_executor/models/__init__.py": "",
    "vllm/model_executor/models/witness.py": ("from vllm.model_executor.layers.fused_moe import FusedMoEFactory\n"),
    "vllm_ascend/__init__.py": ("def _ensure_global_patch():\n    from vllm_ascend.patch import platform\n"),
    "vllm_ascend/envs.py": "",
    "vllm_ascend/ops/__init__.py": "",
    "vllm_ascend/lora/__init__.py": "",
    "vllm_ascend/lora/fused_moe.py": "",
    "vllm_ascend/lora/punica_npu.py": "",
    "vllm_ascend/lora/quant_moe.py": "",
    "vllm_ascend/worker/__init__.py": "",
    "vllm_ascend/worker/model_runner_v1.py": "class NPUModelRunner:\n    pass\n",
    "vllm_ascend/models/__init__.py": "",
    "vllm_ascend/models/deepseek_v4/__init__.py": "",
    "vllm_ascend/models/deepseek_v4/model.py": (
        "from vllm.model_executor.layers.fused_moe import FusedMoEFactory\n"
        "class AscendDeepseekV4ForCausalLM:\n    pass\n"
        "def build_runner():\n    return FusedMoEFactory()\n"
    ),
    "vllm_ascend/patch/__init__.py": "",
    "vllm_ascend/patch/platform/__init__.py": "from . import patch_fused_moe\n",
    "vllm_ascend/patch/platform/patch_fused_moe.py": (
        "import sys\n"
        "import vllm.model_executor.layers.fused_moe as package\n"
        "from vllm.model_executor.layers.fused_moe import layer\n"
        "_original_FusedMoE = layer.FusedMoEFactory\n"
        "def _ascend_FusedMoE():\n"
        "    return _original_FusedMoE(router=lambda: 'ascend')\n"
        "layer.FusedMoEFactory = _ascend_FusedMoE\n"
        "package.FusedMoEFactory = _ascend_FusedMoE\n"
        "for name, module in list(sys.modules.items()):\n"
        "    if name.startswith('vllm.model_executor.models') and module is not None:\n"
        "        if module.__dict__.get('FusedMoEFactory') is _original_FusedMoE:\n"
        "            module.FusedMoEFactory = _ascend_FusedMoE\n"
    ),
}


class RouterImportOrderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for relative, source in MODULES.items():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")

    def run_imports(self, source):
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", f"import sys\nsys.path.insert(0, {str(self.directory)!r})\n{source}"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def manager_prefix(self):
        tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
        entry = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_ds_lora_aot_patch"
        )
        prefix = []
        for node in entry.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("netrsn_turbo.turbo.version_v0300"):
                break
            prefix.append(node)
        else:
            self.fail("Manager no longer has an upstream initialization prefix")
        return ast.unparse(ast.Module(body=prefix, type_ignores=[]))

    def test_late_platform_patch_leaves_ascend_model_factory_stale(self):
        result = self.run_imports(
            "from vllm_ascend.models.deepseek_v4 import model\n"
            "from vllm.model_executor.models import witness\n"
            "original = model.FusedMoEFactory\n"
            "from vllm_ascend import _ensure_global_patch\n"
            "_ensure_global_patch()\n"
            "from vllm_ascend.patch.platform import patch_fused_moe\n"
            "assert model.FusedMoEFactory is original\n"
            "assert witness.FusedMoEFactory is patch_fused_moe._ascend_FusedMoE\n"
            "model.build_runner().router()\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("'_moe_C' object has no attribute 'topk_softplus_sqrt'", result.stderr)

    def assert_manager_routes_to_ascend(self, preload_model):
        source = (
            "from vllm.model_executor.layers.fused_moe import layer\n"
            "original = layer.FusedMoEFactory\n"
            "from vllm.model_executor.models import witness\n"
        )
        if preload_model:
            source += (
                "from vllm_ascend.models.deepseek_v4 import model as early_model\n"
                "assert early_model.FusedMoEFactory is original\n"
            )
        source += self.manager_prefix()
        source += (
            "\nfrom vllm_ascend.models.deepseek_v4 import model\n"
            "from vllm_ascend.patch.platform import patch_fused_moe\n"
            "assert model.FusedMoEFactory is patch_fused_moe._ascend_FusedMoE\n"
            "assert patch_fused_moe._original_FusedMoE is original\n"
            "assert model.build_runner().router() == 'ascend'\n"
            "assert len(layer.calls) == 1\n"
        )
        if preload_model:
            source += "assert model is early_model\n"
        result = self.run_imports(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manager_cold_start_uses_ascend_factory(self):
        self.assert_manager_routes_to_ascend(preload_model=False)

    def test_manager_refreshes_preloaded_model_without_rewriting_original(self):
        self.assert_manager_routes_to_ascend(preload_model=True)


if __name__ == "__main__":
    unittest.main()
