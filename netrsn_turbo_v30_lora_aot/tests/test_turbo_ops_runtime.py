"""Loader/ABI regressions that run without torch, CANN or an NPU."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "netrsn_turbo" / "turbo_ops"


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        package_name = "_turbo_runtime_under_test"
        package = ModuleType(package_name)
        package.__path__ = [str(PACKAGE)]
        sys.modules[package_name] = package
        self.runtime = importlib.import_module(package_name + "._runtime")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.info = {
            "schema_version": 1,
            "component": "ds_lora",
            "namespace": "turbo_ops",
            "python_version": "%d.%d" % sys.version_info[:2],
            "machine": platform.machine(),
            "torch_version": "2.9.0",
            "torch_npu_version": "2.9.0.post1",
            "soc": "ascend910_93",
            "soc_version": "ascend910_9391",
        }
        self.write_info()
        (self.directory / "_C.so").touch()
        (self.directory / "libturbo_ops_lora_kernels.so").touch()
        self.namespace = SimpleNamespace(**{name: Mock(name=name) for name in self.runtime.OP_NAMES})
        self.torch = SimpleNamespace(
            __version__="2.9.0",
            compiler=SimpleNamespace(is_compiling=Mock(return_value=False)),
            npu=SimpleNamespace(
                is_initialized=Mock(return_value=True), is_current_stream_capturing=Mock(return_value=False)
            ),
            ops=SimpleNamespace(turbo_ops=self.namespace, load_library=Mock()),
        )
        self.torch_npu = SimpleNamespace(
            __version__="2.9.0.post1", npu=SimpleNamespace(get_soc_version=Mock(return_value=250))
        )
        self.modules = {"torch": self.torch, "torch_npu": self.torch_npu}
        sys.modules.update(self.modules)
        patches = (
            patch.object(self.runtime, "_package_dir", return_value=self.directory),
            patch.object(self.runtime.importlib, "import_module", side_effect=self.modules.__getitem__),
        )
        self.locator, self.importer = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)

    def write_info(self):
        (self.directory / "_build_info.json").write_text(json.dumps(self.info), encoding="utf-8")

    def test_is_built_does_not_import_or_load_framework(self):
        self.assertTrue(self.runtime.is_built())
        self.importer.assert_not_called()
        self.torch.ops.load_library.assert_not_called()

    def test_initialize_once_and_dispatch_preserves_objects(self):
        x, y = object(), object()
        result = self.runtime.call("bgmv_shrink", x, y, scale=0.5)
        self.assertIs(result, self.namespace.bgmv_shrink.return_value)
        self.namespace.bgmv_shrink.assert_called_once_with(x, y, scale=0.5)
        self.runtime.initialize()
        self.torch.ops.load_library.assert_called_once_with(str(self.directory / "_C.so"))
        self.assertEqual(self.importer.call_count, 2)

    def test_concurrent_initialization_registers_once(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: self.runtime.initialize(), range(32)))
        self.torch.ops.load_library.assert_called_once()

    def test_load_and_dispatch_have_separate_first_use_messages(self):
        x, y = object(), object()
        with self.assertLogs("vllm.netrsn_turbo", level="INFO") as captured:
            first = self.runtime.call("bgmv_shrink", x, y, scale=0.5)
            second = self.runtime.call("bgmv_shrink", x, y, scale=0.75)
            self.runtime.call("add_lora", x)
        messages = [record.getMessage() for record in captured.records]
        loaded = [message for message in messages if "native library loaded" in message]
        shrink = [message for message in messages if "torch.ops.turbo_ops.bgmv_shrink" in message]
        add_lora = [message for message in messages if "torch.ops.turbo_ops.add_lora" in message]
        self.assertEqual(len(loaded), 1)
        self.assertIn(str(self.directory), loaded[0])
        self.assertIn(self.info["soc_version"], loaded[0])
        self.assertEqual(len(shrink), 1)
        self.assertEqual(len(add_lora), 1)
        self.assertIn("dispatch custom op", shrink[0])
        self.assertIn("C++ selects the kernel or skip path", shrink[0])
        self.assertIs(first, self.namespace.bgmv_shrink.return_value)
        self.assertIs(second, self.namespace.bgmv_shrink.return_value)
        self.assertEqual(self.namespace.bgmv_shrink.call_count, 2)
        self.namespace.bgmv_shrink.assert_called_with(x, y, scale=0.75)

    def test_compiled_dispatch_skips_logging_without_skipping_operation(self):
        self.runtime.initialize()
        self.torch.compiler.is_compiling.return_value = True
        marker = object()
        with self.assertNoLogs("vllm.netrsn_turbo", level="INFO"):
            result = self.runtime.call("bgmv_shrink", marker)
        self.assertIs(result, self.namespace.bgmv_shrink.return_value)
        self.namespace.bgmv_shrink.assert_called_once_with(marker)
        self.torch.compiler.is_compiling.return_value = False
        with self.assertLogs("vllm.netrsn_turbo", level="INFO") as captured:
            self.runtime.call("bgmv_shrink", marker)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("torch.ops.turbo_ops.bgmv_shrink", captured.output[0])

    def test_incomplete_or_ambiguous_artifacts_fail_before_torch_import(self):
        for filename in ("_build_info.json", "_C.so", "libturbo_ops_lora_kernels.so"):
            with self.subTest(missing=filename):
                path = self.directory / filename
                contents = path.read_bytes()
                path.unlink()
                self.assertFalse(self.runtime.is_built())
                with self.assertRaisesRegex(RuntimeError, "TURBO_OPS_BUILD=1"):
                    self.runtime.initialize()
                path.write_bytes(contents)
        (self.directory / "_C.extra.so").touch()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.runtime.initialize()
        self.importer.assert_not_called()

    def test_metadata_and_abi_mismatch_do_not_load_library(self):
        for key, value in (
            ("schema_version", 2),
            ("component", "other"),
            ("namespace", "ops_repo"),
            ("python_version", "0.0"),
            ("machine", "other-machine"),
            ("soc", "ascend310p"),
            ("torch_version", "0.0"),
            ("torch_npu_version", "0.0"),
        ):
            with self.subTest(key=key):
                original = self.info[key]
                self.info[key] = value
                self.write_info()
                with self.assertRaisesRegex(RuntimeError, "Rebuild/reinstall"):
                    self.runtime.initialize()
                self.info[key] = original
        self.torch.ops.load_library.assert_not_called()

    def test_invalid_metadata_reports_actionable_error(self):
        for contents in ("not json", "{}", "null", "[]"):
            with self.subTest(contents=contents):
                (self.directory / "_build_info.json").write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Cannot load NetrsnTurbo"):
                    self.runtime.initialize()
        self.importer.assert_not_called()

    def test_wrong_hardware_family_is_rejected_before_native_load(self):
        for runtime_soc in (220, 225, 200, 205, 260, -1):
            with self.subTest(runtime_soc=runtime_soc):
                self.torch_npu.npu.get_soc_version.return_value = runtime_soc
                with self.assertRaisesRegex(RuntimeError, "select a matching A2/A3 build"):
                    self.runtime.initialize()
        self.torch.ops.load_library.assert_not_called()

    def test_a2_build_accepts_a2_device(self):
        self.info["soc"] = "ascend910b"
        self.info["soc_version"] = "ascend910b1"
        self.write_info()
        self.torch_npu.npu.get_soc_version.return_value = 220
        self.runtime.initialize()
        self.torch.ops.load_library.assert_called_once()

    def test_missing_dependency_is_reported_without_native_load(self):
        self.importer.side_effect = ImportError("No module named torch_npu")
        with self.assertRaisesRegex(RuntimeError, "torch_npu"):
            self.runtime.initialize()
        self.torch.ops.load_library.assert_not_called()

    def test_failed_library_load_can_retry(self):
        self.torch.ops.load_library.side_effect = OSError("libopapi.so missing")
        with self.assertRaisesRegex(RuntimeError, "libopapi.so missing"):
            self.runtime.initialize()
        self.assertFalse(self.runtime._loaded)
        self.torch.ops.load_library.side_effect = None
        self.runtime.initialize()
        self.assertEqual(self.torch.ops.load_library.call_count, 2)

    def test_missing_registration_does_not_register_twice(self):
        del self.namespace.add_lora_expand_delta
        with self.assertRaisesRegex(RuntimeError, "missing add_lora_expand_delta"):
            self.runtime.initialize()
        self.namespace.add_lora_expand_delta = Mock()
        self.runtime.initialize()
        self.torch.ops.load_library.assert_called_once()


class PublicApiTests(unittest.TestCase):
    def test_import_has_no_framework_or_external_ops_dependency(self):
        program = """
import importlib.abc
import sys
class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'ops_repo'):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockFramework())
from netrsn_turbo import turbo_ops
assert callable(turbo_ops.initialize)
assert callable(turbo_ops.add_lora)
"""
        subprocess.run([sys.executable, "-B", "-c", program], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_ten_wrappers_preserve_argument_order_defaults_and_results(self):
        package_name = "_turbo_public_api_test"
        spec = importlib.util.spec_from_file_location(
            package_name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
        )
        api = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = api
        try:
            spec.loader.exec_module(api)
            lora = sys.modules[package_name + ".lora"]
            runtime = sys.modules[package_name + "._runtime"]
            self.assertEqual(set(api.__all__) - {"initialize", "is_built"}, set(runtime.OP_NAMES))
            for name in runtime.OP_NAMES:
                function = getattr(api, name)
                parameters = list(inspect.signature(function).parameters.values())
                values = [object() for _ in parameters]
                with self.subTest(name=name), patch.object(lora, "call") as dispatch:
                    self.assertIs(function(*values), dispatch.return_value)
                    dispatch.assert_called_once_with(name, *values)
                required = [object() for p in parameters if p.default is inspect.Parameter.empty]
                defaults = [p.default for p in parameters if p.default is not inspect.Parameter.empty]
                with self.subTest(name=name, defaults=True), patch.object(lora, "call") as dispatch:
                    function(*required)
                    dispatch.assert_called_once_with(name, *required, *defaults)
        finally:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    del sys.modules[key]


if __name__ == "__main__":
    unittest.main()
