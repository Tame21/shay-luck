# SPDX-License-Identifier: Apache-2.0
"""Build integration tests using stdlib mocks and synthetic, unexecuted binaries.

setuptools/wheel are packaging dependencies. No torch, CANN or NPU is needed.
Native archive fixtures exercise packaging only, not compiler or kernel success.
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from setuptools import Extension


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_support import turbo_ops as build


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("TURBO_OPS_")}


class BuildConfigurationTests(unittest.TestCase):
    def test_merge_preserves_parent_extensions_commands_and_package_data(self):
        class ParentExt:
            pass

        class ParentPython:
            pass

        class ParentSdist:
            pass

        other_extension = Extension("netrsn_turbo.other", sources=["other.cpp"])
        unrelated_command = object()
        original = {
            "packages": ["netrsn_turbo", "netrsn_turbo.other"],
            "ext_modules": [other_extension],
            "package_data": {"netrsn_turbo": ["other.json"], "netrsn_turbo.turbo_ops": ["custom.dat"]},
            "cmdclass": {"build_ext": ParentExt, "build_py": ParentPython,
                         "sdist": ParentSdist, "custom": unrelated_command},
            "install_requires": ["existing-dependency"],
        }
        with mock.patch.dict(os.environ, {"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend910b"}, clear=True):
            result = build.configure_turbo_ops(original)
            repeated = build.configure_turbo_ops(result)
        self.assertEqual(original["ext_modules"], [other_extension])
        self.assertEqual(original["packages"], ["netrsn_turbo", "netrsn_turbo.other"])
        self.assertEqual(original["package_data"]["netrsn_turbo.turbo_ops"], ["custom.dat"])
        self.assertIs(result["ext_modules"][0], other_extension)
        self.assertEqual([e.name for e in result["ext_modules"]], ["netrsn_turbo.other", build.EXTENSION_NAME])
        self.assertIn("netrsn_turbo.turbo_ops", result["packages"])
        self.assertEqual(result["package_data"]["netrsn_turbo.turbo_ops"], ["custom.dat", "LICENSE", "provenance.json"])
        self.assertEqual(result["install_requires"], original["install_requires"])
        for name, parent in (("build_ext", ParentExt), ("build_py", ParentPython), ("sdist", ParentSdist)):
            self.assertTrue(issubclass(result["cmdclass"][name], parent))
            self.assertIs(repeated["cmdclass"][name], result["cmdclass"][name])
        self.assertIs(result["cmdclass"]["custom"], unrelated_command)
        self.assertEqual(len(repeated["ext_modules"]), 2)

    def test_default_pure_configuration_needs_no_framework_import(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torch_npu"):
                raise AssertionError("Unexpected framework import: " + name)
            return original(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(builtins, "__import__", side_effect=guarded):
            result = build.configure_turbo_ops({})
        self.assertEqual(result["ext_modules"], [])
        self.assertNotIn("packages", result)

    def test_implicit_main_package_discovery_is_never_replaced_with_only_turbo_ops(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("packages", build.configure_turbo_ops({}))
            self.assertIsNone(build.configure_turbo_ops({"packages": None})["packages"])
        for kwargs in ({}, {"packages": None}):
            with self.subTest(kwargs=kwargs), mock.patch.dict(
                    os.environ, {"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend910b"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "explicit packages list.*main NetrsnTurbo"):
                    build.configure_turbo_ops(kwargs)

    def test_invalid_build_flag_soc_and_editable_are_rejected(self):
        cases = (
            ({"TURBO_OPS_BUILD": "yes"}, [], "must be 0 or 1"),
            ({"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend950"}, [], "A2/A3"),
            ({"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend910b",
              "TURBO_OPS_SOC_VERSION": "ascend910_9391"}, [], "belonging"),
            ({"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend910b",
              "TURBO_OPS_SOC_VERSION": "../../outside"}, [], "belonging"),
            ({"TURBO_OPS_BUILD": "1", "TURBO_OPS_SOC": "ascend910_93"}, ["editable_wheel"], "regular wheel"),
        )
        for environment, arguments, expected in cases:
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment, clear=True), \
                    mock.patch.object(sys, "argv", ["setup.py", *arguments]):
                with self.assertRaisesRegex(RuntimeError, expected):
                    build.configure_turbo_ops({"packages": ["netrsn_turbo"]})

    def test_parent_extension_dispatch_and_outputs_remain_intact(self):
        class Parent:
            def build_extension(self, extension):
                self.parent_calls.append(extension)

            def get_outputs(self):
                return ["existing-output"]

        native = build.TurboOpsExtension(ROOT)
        other = Extension("other", sources=[])
        command = build.turbo_ops_build_ext(Parent)()
        command.parent_calls = []
        command.extensions = [other, native]
        command.get_ext_fullpath = lambda name: str(ROOT / "build-fixture" / "_C.fixture.so")
        with mock.patch.object(build, "build_turbo_ops") as compile_native:
            command.build_extension(other)
            command.build_extension(native)
        self.assertEqual(command.parent_calls, [other])
        compile_native.assert_called_once_with(command, native)
        self.assertEqual(command.get_outputs(), ["existing-output",
                         str(ROOT / "build-fixture" / build.KERNEL_LIBRARY),
                         str(ROOT / "build-fixture" / build.BUILD_INFO)])


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="turbo-build-", dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "build-lib/netrsn_turbo/turbo_ops/_C.fixture.so"
        self.command = SimpleNamespace(build_temp=str(self.root / "build-temp"),
                                       get_ext_fullpath=lambda name: str(self.output))
        self.extension = build.TurboOpsExtension(self.root)
        package = self.root / "netrsn_turbo/turbo_ops"
        write(package / "csrc/CMakeLists.txt", "# synthetic CMake input\n")
        write(package / "provenance.json", json.dumps({"upstream_commit": "fixture-source-revision"}))
        self.environment = {"TURBO_OPS_SOC": "ascend910_93", "ASCEND_HOME_PATH": str(self.root / "cann"), "MAX_JOBS": "3"}
        self.frameworks = {"torch": SimpleNamespace(__version__="2.10.0+fixture"),
                           "torch_npu": SimpleNamespace(__version__="2.10.0.post4+fixture")}

    def test_install_component_extension_suffix_and_runtime_metadata(self):
        calls = []

        def cmake(arguments, **kwargs):
            calls.append(arguments)
            if "--install" in arguments:
                write(self.output.parent / "_C.so", b"synthetic ELF, never executed")
                write(self.output.parent / build.KERNEL_LIBRARY, b"synthetic kernel, never executed")

        with mock.patch.dict(os.environ, dict(self.environment, TURBO_OPS_SOC_VERSION="Ascend910_9381"), clear=True), \
                mock.patch.dict(sys.modules, self.frameworks), mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(build.subprocess, "run", side_effect=cmake):
            build.build_turbo_ops(self.command, self.extension)
        self.assertEqual(len(calls), 3)
        self.assertIn("-DSOC_VERSION=ascend910_9381", calls[0])
        self.assertEqual(calls[1][-2:], ["--parallel", "3"])
        self.assertEqual(calls[2][-2:], ["--component", "turbo_ops"])
        self.assertTrue(self.output.is_file())
        self.assertFalse((self.output.parent / "_C.so").exists())
        info = json.loads((self.output.parent / build.BUILD_INFO).read_text(encoding="utf-8"))
        self.assertEqual(info["component"], "ds_lora")
        self.assertEqual(info["namespace"], "turbo_ops")
        self.assertEqual(info["python_version"], "%d.%d" % sys.version_info[:2])
        self.assertEqual(info["torch_version"], "2.10.0+fixture")
        self.assertEqual(info["torch_npu_version"], "2.10.0.post4+fixture")
        self.assertEqual(info["soc_version"], "ascend910_9381")
        self.assertEqual(info["source_commit"], "fixture-source-revision")

    def test_policies_fail_before_compiler_or_framework_import(self):
        cases = (("win32", self.environment, "Linux"),
                 ("linux", dict(self.environment, MAX_JOBS="0"), "positive integer"),
                 ("linux", dict(self.environment, MAX_JOBS="invalid"), "positive integer"),
                 ("linux", {"TURBO_OPS_SOC": "ascend910b"}, "ASCEND_HOME_PATH"))
        for host, environment, expected in cases:
            with self.subTest(host=host, environment=environment), \
                    mock.patch.dict(os.environ, environment, clear=True), \
                    mock.patch.object(sys, "platform", host), \
                    mock.patch.object(build.subprocess, "run") as cmake:
                with self.assertRaisesRegex(RuntimeError, expected):
                    build.build_turbo_ops(self.command, self.extension)
                cmake.assert_not_called()

    def test_pure_build_cleans_stale_payload_before_and_after_parent_copy(self):
        build_lib = self.root / "build-lib"
        package = build_lib / "netrsn_turbo/turbo_ops"
        unrelated = build_lib / "netrsn_turbo/other/_C.keep.so"
        write(package / "_C.old.so", b"stale")
        write(unrelated, b"keep")

        class Parent:
            def run(parent_self):
                self.assertFalse((package / "_C.old.so").exists())
                write(package / "_C.copied.pyd", b"copied stale source payload")
                write(package / build.KERNEL_LIBRARY, b"copied stale kernel")
                write(package / build.BUILD_INFO, "{}")
                write(package / "include/turbo_ops_lora_kernels/stale.h", "stale")
                write(package / "include/keep/header.h", "keep")
                write(package / "lib" / build.KERNEL_LIBRARY, b"stale duplicate")
                write(package / "lora.py", "# keep\n")

        command = build.turbo_ops_build_py(Parent)()
        command.build_lib = str(build_lib)
        command.distribution = SimpleNamespace(ext_modules=[])
        command.run()
        self.assertFalse(list(package.glob("_C*")))
        for name in (build.KERNEL_LIBRARY, build.BUILD_INFO, "include/turbo_ops_lora_kernels", "lib/" + build.KERNEL_LIBRARY):
            self.assertFalse((package / name).exists(), name)
        self.assertTrue(unrelated.is_file())
        self.assertTrue((package / "include/keep/header.h").is_file())
        self.assertTrue((package / "lora.py").is_file())


@unittest.skipUnless(importlib.util.find_spec("wheel"), "wheel is needed for archive checks")
class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="turbo-archives-", dir=ROOT.parent)
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        shutil.copy2(ROOT / "setup_turbo_ops.py", cls.source)
        for relative in ("build_support", "netrsn_turbo/turbo_ops"):
            shutil.copytree(ROOT / relative, cls.source / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.pyd", "_build_info.json"))
        cls.environment = clean_environment()
        cls.environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
        cls.native = cls.wheel("native", native=True)

    @classmethod
    def setup(cls, arguments, *, native=False):
        bootstrap = textwrap.dedent('''\
            import os, pathlib, runpy, sys
            from types import SimpleNamespace
            from unittest import mock
            source = pathlib.Path(sys.argv[1])
            native = sys.argv[2] == 'native'
            arguments = sys.argv[3:]
            sys.path.insert(0, str(source))
            if native:
                import build_support.turbo_ops as build
                original = build.build_turbo_ops
                frameworks = {'torch': SimpleNamespace(__version__='2.10.0+fixture'),
                              'torch_npu': SimpleNamespace(__version__='2.10.0.post4+fixture')}
                def synthetic_build(command, extension):
                    def fake_cmake(args, **kwargs):
                        if '--install' in args:
                            if args[-2:] != ['--component', 'turbo_ops']:
                                raise AssertionError('Unscoped CANN install')
                            output = pathlib.Path(command.get_ext_fullpath(extension.name)).resolve().parent
                            output.mkdir(parents=True, exist_ok=True)
                            for name in ('_C.so', build.KERNEL_LIBRARY):
                                (output / name).write_bytes(b'\\x7fELF synthetic fixture; never execute')
                    with mock.patch.object(sys, 'platform', 'linux'), \\
                            mock.patch.dict(sys.modules, frameworks), \\
                            mock.patch.object(build.subprocess, 'run', side_effect=fake_cmake):
                        original(command, extension)
                build.build_turbo_ops = synthetic_build
            sys.argv = [str(source / 'setup_turbo_ops.py'), *arguments]
            runpy.run_path(str(source / 'setup_turbo_ops.py'), run_name='__main__')
        ''')
        environment = dict(cls.environment, TURBO_OPS_BUILD="1" if native else "0",
                           TURBO_OPS_SOC="ascend910_93", ASCEND_HOME_PATH=str(cls.root / "synthetic-cann"))
        completed = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(cls.source),
                                    "native" if native else "pure", *arguments],
                                   cwd=cls.root, env=environment, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            raise AssertionError(completed.stdout + completed.stderr)

    @classmethod
    def wheel(cls, name, *, native=False):
        output = cls.root / name
        cls.setup(["build", "--build-lib", "shared-build", "bdist_wheel", "--dist-dir", str(output)], native=native)
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError("Expected one addon wheel")
        return wheels[0]

    def test_native_archive_contains_only_the_owned_runtime_package_and_libraries(self):
        with zipfile.ZipFile(self.native) as archive:
            names = archive.namelist()
            native = [name for name in names if name.endswith((".so", ".pyd"))]
            self.assertEqual(len(native), 2, native)
            self.assertIn("netrsn_turbo/turbo_ops/" + build.KERNEL_LIBRARY, names)
            self.assertIn("netrsn_turbo/turbo_ops/LICENSE", names)
            self.assertIn("netrsn_turbo/turbo_ops/provenance.json", names)
            self.assertIn("netrsn_turbo/turbo_ops/_logging.py", names)
            self.assertNotIn("netrsn_turbo/__init__.py", names)
            self.assertFalse(any(name.startswith(("build_support/", "netrsn_turbo/turbo/", "netrsn_turbo/turbo_ops/csrc/")) for name in names))
            info = json.loads(archive.read("netrsn_turbo/turbo_ops/" + build.BUILD_INFO))
            self.assertEqual(info["namespace"], "turbo_ops")
            self.assertEqual(info["component"], "ds_lora")
            wheel = next(name for name in names if name.endswith(".dist-info/WHEEL"))
            self.assertIn("Root-Is-Purelib: false", archive.read(wheel).decode())

    def test_pure_rebuild_cannot_leak_native_outputs(self):
        pure = self.wheel("pure")
        with zipfile.ZipFile(pure) as archive:
            names = archive.namelist()
            self.assertIn("netrsn_turbo/turbo_ops/lora.py", names)
            self.assertFalse(any(name.endswith((".so", ".pyd")) or name.endswith(build.BUILD_INFO) for name in names))
        self.assertIn("py3-none-any", pure.name)

    def test_source_archive_has_build_support_native_inputs_and_license(self):
        output = self.root / "source-archive"
        self.setup(["sdist", "--dist-dir", str(output)])
        archives = list(output.glob("*.tar.gz"))
        self.assertEqual(len(archives), 1)
        with tarfile.open(archives[0]) as archive:
            names = {name.split("/", 1)[-1] for name in archive.getnames()}
        for name in ("setup_turbo_ops.py", "build_support/__init__.py", "build_support/turbo_ops.py",
                     "netrsn_turbo/turbo_ops/LICENSE", "netrsn_turbo/turbo_ops/provenance.json",
                     "netrsn_turbo/turbo_ops/csrc/CMakeLists.txt"):
            self.assertIn(name, names)
        expected = {path.relative_to(self.source).as_posix()
                    for path in (self.source / "netrsn_turbo/turbo_ops/csrc").rglob("*")
                    if path.is_file() and path.suffix in (".cpp", ".h", ".hpp")}
        self.assertTrue(expected)
        self.assertTrue(expected <= names, sorted(expected - names))
        self.assertFalse(any(name.endswith((".so", ".pyd")) for name in names))


if __name__ == "__main__":
    unittest.main()
