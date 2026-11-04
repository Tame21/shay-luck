# SPDX-License-Identifier: Apache-2.0
"""Exercise the upstream MoE import cycle without importing torch or CANN."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "netrsn_turbo/turbo/version_0300/ds_lora/installer.py"

# Keep the import edges and their order from the Ascend baseline. The bodies
# are reduced to inert definitions so Python's real package loader can expose
# the cycle without framework or hardware dependencies.
MODULES = {
    "vllm_ascend/__init__.py": "def _ensure_global_patch():\n    pass\n",
    "vllm_ascend/envs.py": "",
    "vllm_ascend/lora/__init__.py": "",
    "vllm_ascend/lora/punica_npu.py": "",
    "vllm_ascend/lora/quant_moe.py": "",
    "vllm_ascend/lora/fused_moe.py": (
        "from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all\n"
        "def prepare_lora_indices():\n    return 'prepared'\n"
    ),
    "vllm_ascend/ops/__init__.py": (
        "import vllm_ascend.ops.fused_moe.fused_moe\n"
    ),
    "vllm_ascend/ops/fused_moe/__init__.py": "",
    "vllm_ascend/ops/fused_moe/fused_moe.py": (
        "from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method\n"
    ),
    "vllm_ascend/ops/fused_moe/moe_comm_method.py": (
        "from vllm_ascend.ops.fused_moe import moe_utils\n"
        "from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalize\n"
        "def get_moe_comm_method():\n    return PrepareAndFinalize\n"
    ),
    "vllm_ascend/ops/fused_moe/prepare_finalize.py": (
        "from vllm_ascend.lora.fused_moe import prepare_lora_indices\n"
        "class PrepareAndFinalize:\n    pass\n"
    ),
    "vllm_ascend/ops/fused_moe/moe_utils.py": (
        "def async_all_to_all():\n    pass\n"
    ),
}


class ImportOrderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for relative, source in MODULES.items():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")

    def run_imports(self, source):
        # A fresh interpreter is essential: a preloaded ops package would hide
        # the failure, which is why the registration stubs cannot cover it.
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             f"import sys\nsys.path.insert(0, {str(self.directory)!r})\n{source}"],
            capture_output=True, text=True, timeout=30,
        )

    def test_lora_first_reproduces_partially_initialized_error(self):
        result = self.run_imports(
            "from vllm_ascend.lora import fused_moe, punica_npu, quant_moe\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "cannot import name 'prepare_lora_indices' from partially initialized "
            "module 'vllm_ascend.lora.fused_moe'",
            result.stderr,
        )

    def test_installer_import_prefix_completes_cold_start(self):
        tree = ast.parse(INSTALLER.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "apply_lora_patch")
        imports = []
        for node in entry.body:
            if isinstance(node, ast.Import):
                if all(alias.name.startswith("vllm_ascend") for alias in node.names):
                    imports.append(node)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module and node.module.startswith("vllm_ascend"):
                    imports.append(node)
                    if node.module == "vllm_ascend.lora":
                        break
        else:
            self.fail("Installer no longer imports the upstream LoRA modules")

        source = ast.unparse(ast.Module(body=imports, type_ignores=[]))
        source += (
            "\nfrom vllm_ascend.ops.fused_moe import prepare_finalize\n"
            "assert prepare_finalize.prepare_lora_indices is fused_moe.prepare_lora_indices\n"
            "assert prepare_finalize.prepare_lora_indices() == 'prepared'\n"
        )
        result = self.run_imports(source)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
