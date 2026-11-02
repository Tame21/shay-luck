# SPDX-License-Identifier: Apache-2.0
"""Self-contained source/contract checks; no torch, pytest, CANN or NPU needed.

Optional comparisons use the original ops-repo migration source and the new
vllm-ascend-ds-v2 checkout, and are skipped when those siblings are unavailable.
These checks do not substitute for native hardware tests.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import sys
import types
import unittest
from unittest.mock import patch


DELIVERY = Path(__file__).resolve().parents[1]
WORKSPACE = DELIVERY.parent
PACKAGE = DELIVERY / "netrsn_turbo/turbo_ops"
NPU_TESTS = Path(__file__).resolve().parent / "npu"
OP_NAMES = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "build_combined_lora_idx", "add_lora", "add_lora_shrink", "add_lora_expand",
    "add_lora_expand_delta", "add_lora_swiglu_quant",
)
NPU_MODULES = (
    "test_moe_lora.py", "test_swiglu_quant.py", "test_moe_lora_graph.py",
    "test_fully_sharded_lora_graph.py", "test_v1_ep_lora_graph.py",
)


def read(path):
    return path.read_text(encoding="utf-8")


def source_hash(path):
    return hashlib.sha256(read(path).encode("utf-8")).hexdigest()


def dispatcher_schemas(text):
    result = {}
    for match in re.finditer(r"ops\.def\(\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+)\);", text):
        schema = "".join(ast.literal_eval(part) for part in re.findall(r'"(?:\\.|[^"\\])*"', match[1]))
        name = schema.split("(", 1)[0]
        if name in result:
            raise AssertionError(f"Duplicate schema for {name}")
        result[name] = schema
    return result


class TurboOpsSourceTests(unittest.TestCase):
    def test_provenance_covers_all_native_files_and_matches_destination_hashes(self):
        provenance = json.loads(read(PACKAGE / "provenance.json"))
        self.assertEqual(provenance["component"], "netrsn_turbo.turbo_ops")
        self.assertEqual(provenance["migration_date"], "2026-11-02")
        self.assertEqual(provenance["upstream_repository"], "https://github.com/dghuawei/vllm-ascend.git")
        self.assertEqual(provenance["upstream_branch"], "v30_lora_v2")
        self.assertEqual(provenance["upstream_commit"], "5dd159d7d2acf70c0bfcec1a2055ac52c60a13bf")
        verification = provenance["upstream_verification"]
        self.assertEqual(verification["previous_branch"], "v30_lora_temp")
        self.assertEqual(verification["previous_commit"], "782eec35e340c87434f7fd2e8ce8e06a11149116")
        self.assertIn("ops-repo", provenance["origin"]["repository"])
        entries = provenance["files"]
        paths = [entry["destination_path"] for entry in entries]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), {
            path.relative_to(PACKAGE).as_posix()
            for path in (PACKAGE / "csrc").rglob("*") if path.is_file()
        })
        self.assertEqual(len(paths), 17)
        self.assertEqual(sum("upstream_source_path" in entry for entry in entries), 16)
        for entry in entries:
            with self.subTest(file=entry["destination_path"]):
                self.assertEqual(source_hash(PACKAGE / entry["destination_path"]), entry["destination_sha256"])
                self.assertRegex(entry["source_sha256"], r"^[0-9a-f]{64}$")
                if "upstream_source_path" in entry:
                    self.assertRegex(entry["upstream_source_sha256"], r"^[0-9a-f]{64}$")

    def test_ten_privateuse1_and_meta_contracts_are_registered(self):
        host = read(PACKAGE / "csrc/lora/torch_binding.cpp")
        meta = read(PACKAGE / "csrc/lora/torch_binding_meta.cpp")
        schemas = dispatcher_schemas(host)
        self.assertEqual(set(schemas), set(OP_NAMES))
        registrations = re.findall(
            r'ops\.impl\("(\w+)",\s*torch::kPrivateUse1,\s*&turbo_ops_lora::(\w+)\);', host,
        )
        meta_registrations = re.findall(
            r'ops\.impl\("(\w+)",\s*&turbo_ops_lora::meta::(\w+)_meta\);', meta,
        )
        self.assertCountEqual(registrations, [(name, name) for name in OP_NAMES])
        self.assertCountEqual(meta_registrations, [(name, name) for name in OP_NAMES])
        self.assertIn("TORCH_LIBRARY_FRAGMENT(turbo_ops, ops)", host)
        self.assertIn("TORCH_LIBRARY_IMPL(turbo_ops, Meta, ops)", meta)
        for name in OP_NAMES:
            self.assertRegex(meta, rf"\b{name}_meta\s*\(")
        for name in ("bgmv_expand", "sgmv_expand"):
            self.assertIn("Tensor(a!) y", schemas[name])
            self.assertTrue(schemas[name].endswith("-> Tensor(a!)"))
            self.assertRegex(meta, rf"{name}_meta\([^{{]+\{{\s*return y;\s*\}}")

    def test_public_wrappers_follow_schema_and_return_dispatch_result(self):
        schemas = dispatcher_schemas(read(PACKAGE / "csrc/lora/torch_binding.cpp"))
        tree = ast.parse(read(PACKAGE / "lora.py"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual({node.name for node in functions}, set(OP_NAMES))
        calls = []
        result = object()

        def dispatch(name, *args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        namespace = {"call": dispatch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "lora.py", "exec"), namespace)
        for name in OP_NAMES:
            with self.subTest(operator=name):
                schema_args = re.fullmatch(r"\w+\((.*)\)\s*->.*", schemas[name])[1].split(",")
                parameters, defaults = [], []
                for argument in schema_args:
                    declaration, separator, default = argument.strip().partition("=")
                    parameters.append(declaration.rsplit(None, 1)[-1])
                    if separator:
                        defaults.append(ast.literal_eval(default))
                function = namespace[name]
                signature = inspect.signature(function)
                self.assertEqual(list(signature.parameters), parameters)
                actual_defaults = [value.default for value in signature.parameters.values()
                                   if value.default is not inspect.Parameter.empty]
                self.assertEqual(actual_defaults, defaults)
                values = [object() for _ in parameters]
                self.assertIs(function(*values), result)
                self.assertEqual(calls.pop(), (name, tuple(values), {}))
                required = values[:len(values) - len(defaults)]
                self.assertIs(function(*required), result)
                self.assertEqual(calls.pop(), (name, tuple(required + defaults), {}))
        exports = ast.parse(read(PACKAGE / "__init__.py"))
        imported = {alias.name for node in exports.body if isinstance(node, ast.ImportFrom)
                    and node.module == "lora" for alias in node.names}
        self.assertEqual(imported, set(OP_NAMES))

    def test_native_names_and_build_paths_are_isolated(self):
        kernels = PACKAGE / "csrc/kernels/lora"
        self.assertEqual(len(list(kernels.glob("*.cpp"))), 7)
        for path in (PACKAGE / "csrc").rglob("*"):
            if not path.is_file():
                continue
            with self.subTest(file=path.name):
                text = read(path)
                self.assertNotIn("ops_repo", text)
                self.assertNotIn("vllm_ascend", text)
                for symbol in re.findall(r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+(\w+)', text):
                    self.assertTrue(symbol.startswith("turbo_ops_"), symbol)
        host = read(PACKAGE / "csrc/lora/torch_binding.cpp")
        self.assertIn("PyInit__C()", host)
        self.assertNotIn("PyInit__lora_C", host)
        cmake = read(PACKAGE / "csrc/CMakeLists.txt")
        for required in (
            'set(KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/lora")',
            '"${CMAKE_CURRENT_SOURCE_DIR}/lora/*.cpp"',
            "ascendc_library(turbo_ops_lora_kernels SHARED", "add_library(_C SHARED",
            "COMPONENT turbo_ops)", 'INSTALL_RPATH "$ORIGIN"', '"-Wl,-rpath,$ORIGIN"',
        ):
            self.assertIn(required, cmake)
        self.assertNotIn("REPO_ROOT", cmake)

    @unittest.skipUnless((WORKSPACE / "ops-repo").is_dir(), "Optional ops-repo source checkout is unavailable")
    def test_local_origin_hashes_and_normalized_native_sources(self):
        origin = WORKSPACE / "ops-repo"
        provenance = json.loads(read(PACKAGE / "provenance.json"))
        for entry in provenance["files"]:
            with self.subTest(file=entry["destination_path"]):
                original = read(origin / entry["source_path"])
                self.assertEqual(source_hash(origin / entry["source_path"]), entry["source_sha256"])
                if entry["destination_path"].endswith("CMakeLists.txt"):
                    continue  # layout changes are checked separately
                expected = original.replace("ops_repo_lora", "turbo_ops_lora").replace("ops_repo_", "turbo_ops_")
                expected = re.sub(r"\bops_repo\b", "turbo_ops", expected).replace("_lora_C", "_C")
                self.assertEqual(read(PACKAGE / entry["destination_path"]), expected)

    @unittest.skipUnless((WORKSPACE / "vllm-ascend-ds-v2").is_dir(), "Optional DS v2 source checkout is unavailable")
    def test_local_ds_v2_kernel_alignment_and_upstream_hashes(self):
        source = WORKSPACE / "vllm-ascend-ds-v2"
        provenance = json.loads(read(PACKAGE / "provenance.json"))
        for entry in provenance["files"]:
            if "upstream_source_path" not in entry:
                continue
            with self.subTest(file=entry["destination_path"]):
                upstream = source / entry["upstream_source_path"]
                self.assertEqual(source_hash(upstream), entry["upstream_source_sha256"])
                if "/kernels/" in entry["destination_path"]:
                    restored = read(PACKAGE / entry["destination_path"])
                    restored = restored.replace("turbo_ops_lora", "vllm_ascend").replace("turbo_ops_", "")
                    self.assertEqual(restored.rstrip("\n"), read(upstream).rstrip("\n"))


class HardwareTestCollectionTests(unittest.TestCase):
    def test_all_five_modules_skip_before_hardware_imports_without_opt_in(self):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "torch_npu", "netrsn_turbo", "ops_repo", "vllm", "vllm_ascend"}:
                self.fail(f"Disabled hardware collection imported {name}")
            return real_import(name, *args, **kwargs)

        def skip(message, *, allow_module_level):
            self.assertTrue(allow_module_level)
            raise unittest.SkipTest(message)

        fake_pytest = types.ModuleType("pytest")
        fake_pytest.skip = skip
        for filename in NPU_MODULES:
            with self.subTest(module=filename), \
                    patch.dict(os.environ, {"TURBO_OPS_RUN_NPU_TESTS": "0"}), \
                    patch.dict(sys.modules, {"pytest": fake_pytest}), \
                    patch("builtins.__import__", side_effect=guard), \
                    self.assertRaisesRegex(unittest.SkipTest, "TURBO_OPS_RUN_NPU_TESTS=1"):
                runpy.run_path(str(NPU_TESTS / filename))

    def test_every_module_initializes_after_device_selection_and_has_no_external_backend_import(self):
        for filename in NPU_MODULES:
            with self.subTest(module=filename):
                text = read(NPU_TESTS / filename)
                self.assertTrue(text.startswith("# SPDX-License-Identifier: Apache-2.0"))
                self.assertIn("Vendored from ops-repo/tests/npu/", text)
                tree = ast.parse(text)
                initializers = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(alias.name.split(".")[0] in {"ops_repo", "vllm", "vllm_ascend"}
                                             for alias in node.names))
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotIn((node.module or "").split(".")[0], {"ops_repo", "vllm", "vllm_ascend"})
                    elif isinstance(node, ast.FunctionDef):
                        calls = [statement.value for statement in node.body
                                 if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)]
                        call_names = [ast.unparse(call.func) for call in calls]
                        if "torch.npu.set_device" in call_names:
                            initializers.append(node)
                            index = call_names.index("torch.npu.set_device")
                            self.assertEqual(call_names[index + 1], "turbo_ops.initialize")
                self.assertEqual(len(initializers), 1)


if __name__ == "__main__":
    unittest.main()
