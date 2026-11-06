# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for version_v0300's built-in turbo_ops integration."""

import ast
import builtins
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "netrsn_turbo/turbo/version_v0300"
LORA_PAYLOAD = PAYLOAD / "vllm_ascend/lora"
PACKAGE = "_turbo_ops_integration_fixture"
LORA_PACKAGE = PACKAGE + ".vllm_ascend.lora"


def load_backend():
    module = types.ModuleType(LORA_PACKAGE + ".ops_backend")
    module.__package__ = LORA_PACKAGE
    source = LORA_PAYLOAD / "ops_backend.py"
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    return module


def fake_backend(helper):
    return types.SimpleNamespace(
        initialize=mock.Mock(),
        **{name: mock.Mock(name=name) for name in helper._REQUIRED_APIS},
    )


class BackendInitializationTests(unittest.TestCase):
    def test_import_is_lightweight_and_does_not_load_frameworks_or_backend(self):
        real_import = builtins.__import__
        imports = []

        def checked_import(name, *args, **kwargs):
            imports.append(name)
            if name.split(".")[0] in {"torch", "torch_npu", "netrsn_turbo", "ops_repo"}:
                self.fail("Importing the helper eagerly loaded " + name)
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=checked_import):
            helper = load_backend()
        self.assertIsNone(helper._BACKEND)
        self.assertEqual(imports, ["importlib"])

    def test_success_initializes_internal_backend_once_and_caches_hot_path(self):
        helper = load_backend()
        backend = fake_backend(helper)
        with mock.patch.object(helper, "import_module", return_value=backend) as importer:
            self.assertIs(helper.initialize_lora_backend(), backend)
            self.assertIs(helper.initialize_lora_backend(), backend)
            self.assertIs(helper.get_lora_ops(), backend)
            importer.assert_called_once_with("netrsn_turbo.turbo_ops")
        backend.initialize.assert_called_once_with()

    def test_get_lora_ops_initializes_on_first_use(self):
        helper = load_backend()
        backend = fake_backend(helper)
        with mock.patch.object(helper, "import_module", return_value=backend) as importer:
            self.assertIs(helper.get_lora_ops(), backend)
            importer.assert_called_once_with("netrsn_turbo.turbo_ops")
        backend.initialize.assert_called_once_with()

    def test_missing_public_apis_are_reported_and_failure_is_not_cached(self):
        helper = load_backend()
        backend = fake_backend(helper)
        backend.add_lora = None
        del backend.sgmv_shrink
        with mock.patch.object(helper, "import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "missing public LoRA APIs") as raised:
                helper.initialize_lora_backend()
        self.assertIn("add_lora", str(raised.exception))
        self.assertIn("sgmv_shrink", str(raised.exception))
        backend.initialize.assert_not_called()
        self.assertIsNone(helper._BACKEND)

        repaired = fake_backend(helper)
        with mock.patch.object(helper, "import_module", return_value=repaired):
            self.assertIs(helper.initialize_lora_backend(), repaired)

    def test_missing_initializer_and_native_failure_are_actionable(self):
        helper = load_backend()
        backend = fake_backend(helper)
        backend.initialize = None
        with mock.patch.object(helper, "import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "netrsn_turbo.turbo_ops.initialize"):
                helper.initialize_lora_backend()
        self.assertIsNone(helper._BACKEND)

        native_error = ImportError("kernel library missing")
        backend.initialize = mock.Mock(side_effect=native_error)
        with mock.patch.object(helper, "import_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "built-in turbo_ops") as raised:
                helper.initialize_lora_backend()
        self.assertIs(raised.exception.__cause__, native_error)
        self.assertIn("kernel library missing", str(raised.exception))
        self.assertIsNone(helper._BACKEND)


class DenseAdapterTests(unittest.TestCase):
    def setUp(self):
        helper = load_backend()
        self.backend = fake_backend(helper)
        helper.get_lora_ops = mock.Mock(return_value=self.backend)
        packages = {}
        for name in (PACKAGE, PACKAGE + ".vllm_ascend", LORA_PACKAGE):
            package = types.ModuleType(name)
            package.__path__ = []
            packages[name] = package
        self.module = types.ModuleType(LORA_PACKAGE + ".lora_ops")
        self.module.__package__ = LORA_PACKAGE
        source = LORA_PAYLOAD / "lora_ops.py"
        with mock.patch.dict(sys.modules, {**packages, helper.__name__: helper}):
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), self.module.__dict__)

    def test_bgmv_signatures_and_argument_order(self):
        x, weight, indices = object(), object(), object()
        y = types.SimpleNamespace(size=lambda dim: 192)
        result = self.module.bgmv_shrink(x, weight, y, indices)
        self.backend.bgmv_shrink.assert_called_once_with(x, weight, indices, y, 1.0)
        self.assertIs(result, self.backend.bgmv_shrink.return_value)
        result = self.module.bgmv_expand(x, weight, y, indices, add_inputs=False)
        self.backend.bgmv_expand.assert_called_once_with(x, weight, indices, y, 0, 192)
        self.assertIs(result, self.backend.bgmv_expand.return_value)
        self.module.bgmv_expand_slice(x, weight, y, indices, 32, 64, add_inputs=True)
        self.backend.bgmv_expand.assert_called_with(x, weight, indices, y, 32, 64)

    def test_sgmv_reorders_vllm_metadata_for_the_native_api(self):
        x, weight, start, lengths, indices = (object() for _ in range(5))
        y = types.SimpleNamespace(size=lambda dim: 256)
        metadata = (start, lengths, indices, 3, 7, 21)
        self.module.sgmv_shrink(x, weight, y, *metadata, 0.25)
        self.backend.sgmv_shrink.assert_called_once_with(x, weight, indices, lengths, y, 0.25)
        result = self.module.sgmv_expand(x, weight, y, *metadata)
        self.backend.sgmv_expand.assert_called_once_with(x, weight, indices, lengths, y, 0, 256)
        self.assertIs(result, self.backend.sgmv_expand.return_value)
        self.module.sgmv_expand_slice(x, weight, y, *metadata, 48, 80, add_inputs=True)
        self.backend.sgmv_expand.assert_called_with(x, weight, indices, lengths, y, 48, 80)


class IntegrationScopeTests(unittest.TestCase):
    def test_public_ten_api_signatures_and_forwarding_match_native_schemas(self):
        component = ROOT / "netrsn_turbo/turbo_ops"
        bindings = (component / "csrc/lora/torch_binding.cpp").read_text(encoding="utf-8")
        schemas = {}
        for match in re.finditer(r'ops\.def\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)\);', bindings):
            parts = re.findall(r'"(?:\\.|[^"\\])*"', match[1])
            schema = "".join(ast.literal_eval(part) for part in parts)
            schemas[schema.split("(", 1)[0]] = schema
        functions = {
            node.name: node
            for node in ast.parse((component / "lora.py").read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
        }
        required = set(load_backend()._REQUIRED_APIS)
        self.assertEqual(set(functions), required)
        self.assertEqual(set(schemas), required)
        for name in sorted(required):
            with self.subTest(operator=name):
                arguments = re.fullmatch(r"\w+\((.*)\)\s*->.*", schemas[name])[1]
                parameters, defaults = [], []
                for declaration in arguments.split(","):
                    declaration, separator, default = declaration.strip().partition("=")
                    parameters.append(declaration.rsplit(None, 1)[-1])
                    if separator:
                        defaults.append(ast.literal_eval(default))
                function = functions[name]
                self.assertEqual([argument.arg for argument in function.args.args], parameters)
                self.assertEqual([ast.literal_eval(value) for value in function.args.defaults], defaults)
                self.assertFalse(
                    function.args.posonlyargs or function.args.kwonlyargs or function.args.vararg or function.args.kwarg
                )
                expected = ast.parse("return call(%r, %s)" % (name, ", ".join(parameters))).body[0]
                self.assertEqual(len(function.body), 1)
                self.assertEqual(ast.dump(function.body[0]), ast.dump(expected))

    def test_six_new_native_apis_route_to_internal_backend(self):
        expected = {
            "build_combined_lora_idx",
            "add_lora",
            "add_lora_shrink",
            "add_lora_expand",
            "add_lora_expand_delta",
            "add_lora_swiglu_quant",
        }
        calls = set()
        for name in ("fused_moe.py", "punica_npu.py", "quant_moe.py"):
            tree = ast.parse((LORA_PAYLOAD / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(ast.unparse(node), "torch.ops._C_ascend")
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "get_lora_ops":
                            calls.add(node.attr)
        self.assertEqual(calls, expected)

    def test_no_external_ops_repo_runtime_imports(self):
        for path in PAYLOAD.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.split(".")[0] == "ops_repo" for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "ops_repo")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "import_module" and node.args and isinstance(node.args[0], ast.Constant):
                        self.assertNotEqual(node.args[0].value.split(".")[0], "ops_repo")

    def test_punica_native_branch_uses_adapters_and_keeps_torch_fallback(self):
        tree = ast.parse((LORA_PAYLOAD / "punica_npu.py").read_text(encoding="utf-8"))
        initializer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        selection = next(
            n for n in initializer.body if isinstance(n, ast.If) and "LORA_CUSTOM_OPS" in ast.unparse(n.test)
        )
        self.assertIn("max_lora_rank >= 128", ast.unparse(selection.test))
        self.assertEqual(selection.body[0].module, "vllm.lora.ops.torch_ops")
        self.assertEqual(selection.orelse[0].module, "lora_ops")
        self.assertEqual(selection.orelse[0].level, 1)
        self.assertEqual(
            {a.name for a in selection.orelse[0].names},
            {
                "bgmv_shrink",
                "bgmv_expand",
                "bgmv_expand_slice",
                "sgmv_shrink",
                "sgmv_expand",
                "sgmv_expand_slice",
            },
        )


if __name__ == "__main__":
    unittest.main()
