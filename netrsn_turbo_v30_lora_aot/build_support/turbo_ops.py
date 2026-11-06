# SPDX-License-Identifier: Apache-2.0
"""Compose the built-in LoRA extension with NetrsnTurbo's existing setup.

    from build_support import configure_turbo_ops
    setup(**configure_turbo_ops(existing_setup_kwargs))

The helper preserves existing extensions and command classes. Native framework
imports happen only while building TURBO_OPS_BUILD=1; pure wheels need neither
torch nor torch_npu. No NetrsnTurbo setup.py is supplied or overwritten here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


EXTENSION_NAME = "netrsn_turbo.turbo_ops._C"
PACKAGE_PARTS = ("netrsn_turbo", "turbo_ops")
KERNEL_LIBRARY = "libturbo_ops_lora_kernels.so"
BUILD_INFO = "_build_info.json"
SOC_DEFAULTS = {"ascend910b": "ascend910b1", "ascend910_93": "ascend910_9391"}
SOC_PATTERNS = {"ascend910b": r"ascend910b[1-4](?:-1|c)?", "ascend910_93": r"ascend910_93[0-9]{2}"}


def _enabled():
    value = os.environ.get("TURBO_OPS_BUILD", "0")
    if value not in ("0", "1"):
        raise RuntimeError("TURBO_OPS_BUILD must be 0 or 1")
    return value == "1"


def _target():
    soc = os.environ.get("TURBO_OPS_SOC")
    if soc not in SOC_DEFAULTS:
        raise RuntimeError("TURBO_OPS_SOC must be ascend910b or ascend910_93 (A2/A3)")
    version = os.environ.get("TURBO_OPS_SOC_VERSION", SOC_DEFAULTS[soc]).lower()
    if re.fullmatch(SOC_PATTERNS[soc], version) is None:
        raise RuntimeError("TURBO_OPS_SOC_VERSION must be a supported concrete model belonging to TURBO_OPS_SOC")
    return soc, version


class TurboOpsExtension(Extension):
    def __init__(self, project_root):
        super().__init__(EXTENSION_NAME, sources=[])
        self.project_root = Path(project_root).resolve()


def _remove_owned_path(path, boundary):
    """Only remove this component's named artifacts within the chosen build tree."""
    path, boundary = Path(path), Path(boundary).resolve()
    # A linked parent could redirect even an unlink into a different checkout.
    path.parent.resolve().relative_to(boundary)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        path.resolve().relative_to(boundary)
        shutil.rmtree(path)


def clean_turbo_ops_payload(build_lib):
    """Drop only owned binaries/receipts; retain other NetrsnTurbo artifacts."""
    build_lib = Path(build_lib).resolve()
    package = build_lib.joinpath(*PACKAGE_PARTS)
    package.resolve().relative_to(build_lib)
    _clean_package(package, build_lib)


def _clean_package(package, boundary):
    if not package.exists():
        return
    owned = list(package.glob("_C*.so")) + list(package.glob("_C*.pyd"))
    owned += [
        package / KERNEL_LIBRARY,
        package / BUILD_INFO,
        package / "include" / "turbo_ops_lora_kernels",
        package / "lib" / KERNEL_LIBRARY,
    ]
    for path in owned:
        _remove_owned_path(path, boundary)


def build_turbo_ops(command, extension):
    """Build and install exactly this component into setuptools' output tree."""
    if sys.platform != "linux":
        raise RuntimeError("TURBO_OPS_BUILD=1 requires a Linux CANN build environment")
    if getattr(command, "inplace", False) or getattr(command, "editable_mode", False):
        raise RuntimeError("turbo_ops native payload requires a regular wheel, not an editable/in-place build")
    soc, soc_version = _target()
    toolkit = os.environ.get("ASCEND_HOME_PATH")
    if not toolkit:
        raise RuntimeError("Source CANN set_env.sh so ASCEND_HOME_PATH is set before building turbo_ops")
    try:
        jobs = int(os.environ.get("MAX_JOBS", "4"))
    except ValueError as exc:
        raise RuntimeError("MAX_JOBS must be a positive integer") from exc
    if jobs < 1:
        raise RuntimeError("MAX_JOBS must be a positive integer")
    package_source = extension.project_root.joinpath(*PACKAGE_PARTS)
    cmake_source = package_source / "csrc"
    if not (cmake_source / "CMakeLists.txt").is_file():
        raise RuntimeError("Missing turbo_ops CMakeLists.txt: " + str(cmake_source))
    provenance = json.loads((package_source / "provenance.json").read_text(encoding="utf-8"))
    source_commit = provenance["upstream_commit"]
    if not isinstance(source_commit, str) or not source_commit:
        raise RuntimeError("turbo_ops provenance must provide upstream_commit")

    # Kept after policy/source checks: pure packaging and invalid targets must
    # not require or initialize either native framework.
    import torch
    import torch_npu

    destination = Path(command.get_ext_fullpath(extension.name)).resolve()
    output = destination.parent
    _clean_package(output, output)
    build_parent = Path(command.build_temp).resolve()
    build = build_parent / ("turbo_ops_" + soc_version)
    if build.is_symlink():
        raise RuntimeError("turbo_ops build directory must not be a symbolic link")
    build.resolve().relative_to(build_parent)
    if build.exists():
        shutil.rmtree(build)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(cmake_source),
            "-B",
            str(build),
            "-DSOC_VERSION=" + soc_version,
            "-DASCEND_CANN_PACKAGE_PATH=" + toolkit,
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(output),
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    subprocess.run(["cmake", "--install", str(build), "--component", "turbo_ops"], check=True)
    native = output / "_C.so"
    if not native.is_file() or not (output / KERNEL_LIBRARY).is_file():
        raise RuntimeError("CMake must install _C.so and " + KERNEL_LIBRARY)
    if native != destination:
        native.replace(destination)
    info = {
        "schema_version": 1,
        "component": "ds_lora",
        "namespace": "turbo_ops",
        "python_version": "%d.%d" % sys.version_info[:2],
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "machine": platform.machine(),
        "soc": soc,
        "soc_version": soc_version,
        "source_commit": source_commit,
    }
    (output / BUILD_INFO).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def turbo_ops_build_ext(base=build_ext):
    if getattr(base, "_turbo_ops_composed", False):
        return base

    class BuildNative(base):
        _turbo_ops_composed = True

        def build_extension(self, extension):
            if isinstance(extension, TurboOpsExtension):
                build_turbo_ops(self, extension)
            else:
                super().build_extension(extension)

        def get_outputs(self):
            outputs = list(super().get_outputs())
            for extension in self.extensions or ():
                if isinstance(extension, TurboOpsExtension):
                    package = Path(self.get_ext_fullpath(extension.name)).parent
                    outputs += [str(package / KERNEL_LIBRARY), str(package / BUILD_INFO)]
            return outputs

    return BuildNative


def turbo_ops_build_py(base=build_py):
    if getattr(base, "_turbo_ops_composed", False):
        return base

    class BuildPython(base):
        _turbo_ops_composed = True

        def run(self):
            clean_turbo_ops_payload(self.build_lib)
            super().run()
            # A parent build_py may copy stale source-tree package_data.
            # Native build_ext runs afterwards; pure builds must remove it.
            if not any(isinstance(ext, TurboOpsExtension) for ext in self.distribution.ext_modules or ()):
                clean_turbo_ops_payload(self.build_lib)

    return BuildPython


def _source_inputs(project_root):
    """Supply native/build helper inputs to sdist without putting them in wheels."""
    root = Path(project_root).resolve()
    selected = [
        root / "setup_turbo_ops.py",
        root.joinpath(*PACKAGE_PARTS) / "provenance.json",
        root.joinpath(*PACKAGE_PARTS) / "LICENSE",
    ]
    for directory in (root / "build_support", root.joinpath(*PACKAGE_PARTS) / "csrc"):
        if directory.is_symlink():
            raise RuntimeError("turbo_ops source directories must not be symbolic links")
        if directory.exists():
            selected.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".hpp", ".cmake", ".txt", ".json"}
            )
    result = []
    for path in selected:
        if path.is_symlink():
            raise RuntimeError("turbo_ops source inputs must not be symbolic links: " + str(path))
        if path.is_file():
            path.resolve().relative_to(root)
            result.append(path.relative_to(root).as_posix())
    return result


def turbo_ops_sdist(base=sdist, *, project_root):
    if getattr(base, "_turbo_ops_composed", False):
        return base

    class SourceDistribution(base):
        _turbo_ops_composed = True

        def make_release_tree(self, base_dir, files):
            super().make_release_tree(base_dir, sorted(set(files) | set(_source_inputs(project_root))))

    return SourceDistribution


def configure_turbo_ops(setup_kwargs, *, project_root=None):
    """Return setup kwargs with composed build commands and optional native code.

    This does not mutate setup_kwargs or its ext_modules/cmdclass containers.
    The runtime package is added to an explicit package list. Pure builds keep
    omitted/None packages untouched so setuptools can discover the main project.
    Native builds require that explicit list: setuptools otherwise suppresses
    discovery when ext_modules is nonempty. Dependencies and unrelated commands
    are not replaced. This helper expects netrsn_turbo at the project root.
    """
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    enabled = _enabled()
    extensions = list(setup_kwargs.get("ext_modules") or ())
    if enabled:
        _target()
        if setup_kwargs.get("packages") is None:
            raise RuntimeError(
                "TURBO_OPS_BUILD=1 requires an explicit packages list containing the main "
                "NetrsnTurbo packages; pass the project's existing package discovery result"
            )
        if any(arg in ("develop", "editable_wheel") for arg in sys.argv[1:]):
            raise RuntimeError("turbo_ops native payload requires a regular wheel installation")
        matching = [extension for extension in extensions if extension.name == EXTENSION_NAME]
        if matching and not all(isinstance(extension, TurboOpsExtension) for extension in matching):
            raise RuntimeError(EXTENSION_NAME + " is already owned by another extension")
        if not matching:
            extensions.append(TurboOpsExtension(root))
    else:
        extensions = [extension for extension in extensions if not isinstance(extension, TurboOpsExtension)]
    commands = dict(setup_kwargs.get("cmdclass") or {})
    commands["build_ext"] = turbo_ops_build_ext(commands.get("build_ext", build_ext))
    commands["build_py"] = turbo_ops_build_py(commands.get("build_py", build_py))
    commands["sdist"] = turbo_ops_sdist(commands.get("sdist", sdist), project_root=root)
    package_name = ".".join(PACKAGE_PARTS)
    package_data = dict(setup_kwargs.get("package_data") or {})
    own_data = list(package_data.get(package_name) or ())
    for name in ("LICENSE", "provenance.json"):
        if name not in own_data:
            own_data.append(name)
    package_data[package_name] = own_data
    result = dict(setup_kwargs, ext_modules=extensions, cmdclass=commands, package_data=package_data)
    if setup_kwargs.get("packages") is not None:
        packages = list(setup_kwargs["packages"])
        if package_name not in packages:
            packages.append(package_name)
        result["packages"] = packages
    result.setdefault("zip_safe", False)
    return result
