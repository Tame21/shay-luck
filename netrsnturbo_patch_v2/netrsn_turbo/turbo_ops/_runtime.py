"""Load the package-local DS LoRA library once, in the calling worker."""

import importlib
import json
from pathlib import Path
import platform
import sys
import threading

from ._logging import log_once

OP_NAMES = (
    "bgmv_shrink", "bgmv_expand", "sgmv_shrink", "sgmv_expand",
    "build_combined_lora_idx", "add_lora", "add_lora_shrink", "add_lora_expand",
    "add_lora_expand_delta", "add_lora_swiglu_quant",
)
_lock = threading.Lock()
_loaded = False
_ops = None


def _package_dir():
    return Path(__file__).resolve().parent


def is_built():
    """Report whether native artifacts exist, without importing torch or probing NPU."""
    package = _package_dir()
    return ((package / "_build_info.json").is_file()
            and len(list(package.glob("_C*.so"))) == 1
            and (package / "libturbo_ops_lora_kernels.so").is_file())


def initialize():
    """Validate the build ABI and register all ten ops in torch.ops.turbo_ops.

    Call after selecting the worker's device, before graph capture/compilation.
    Ordinary Python imports do not invoke this function.
    """
    global _loaded, _ops
    if _ops is not None:
        return
    with _lock:
        if _ops is not None:
            return
        package = _package_dir()
        if not is_built():
            raise RuntimeError(
                "NetrsnTurbo turbo_ops DS LoRA is not built or is incomplete. "
                "Build NetrsnTurbo with TURBO_OPS_BUILD=1 and "
                "TURBO_OPS_SOC=ascend910b (A2) or ascend910_93 (A3) in the "
                "matching Linux/CANN/torch/torch_npu environment."
            )
        try:
            info = json.loads((package / "_build_info.json").read_text(encoding="utf-8"))
            if (info["schema_version"] != 1 or info["component"] != "ds_lora"
                    or info["namespace"] != "turbo_ops"
                    or info["soc"] not in ("ascend910b", "ascend910_93")):
                raise ValueError("unsupported build metadata or target")
            expected = {"python_version": "%d.%d" % sys.version_info[:2],
                        "machine": platform.machine()}
            for name, actual in expected.items():
                if info[name] != actual:
                    raise ValueError("built for %s=%s, running %s" % (name, info[name], actual))
            torch = importlib.import_module("torch")
            torch_npu = importlib.import_module("torch_npu")
            for name, module in (("torch_version", torch), ("torch_npu_version", torch_npu)):
                if info[name] != str(module.__version__):
                    raise ValueError("built for %s=%s, running %s" % (name, info[name], module.__version__))
            # Numeric family IDs match the pinned vLLM Ascend device/hardware.py.
            # This runs after device selection, never during a Python import.
            runtime_soc = torch_npu.npu.get_soc_version()
            family = ("ascend910b" if 220 <= runtime_soc <= 225 else
                      "ascend910_93" if 250 <= runtime_soc <= 256 else None)
            if family != info["soc"]:
                raise ValueError("built for soc=%s, running NPU soc=%s; select a matching A2/A3 build"
                                 % (info["soc"], runtime_soc))
            if not _loaded:
                torch.ops.load_library(str(next(package.glob("_C*.so"))))
                _loaded = True
            namespace = torch.ops.turbo_ops
            missing = [name for name in OP_NAMES if not callable(getattr(namespace, name, None))]
            if missing:
                raise ValueError("native extension is missing " + ", ".join(missing))
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "Cannot load NetrsnTurbo turbo_ops DS LoRA: %s. Rebuild/reinstall "
                "for this environment and restart the worker." % exc
            ) from exc
        _ops = namespace
        log_once(
            "turbo_ops.loaded",
            "[turbo_ops] native library loaded and APIs registered: path=%s soc=%s",
            package,
            info["soc_version"],
        )


def call(name, *args, **kwargs):
    initialize()
    log_once(
        "turbo_ops.call." + name,
        "[turbo_ops] dispatch custom op: torch.ops.turbo_ops.%s "
        "(C++ selects the kernel or skip path)",
        name,
    )
    return getattr(_ops, name)(*args, **kwargs)
