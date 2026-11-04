# SPDX-License-Identifier: Apache-2.0
"""First-use diagnostics shared by the Python operators and DS LoRA patches."""

import logging
import os


# Reuse vLLM's configured handler without importing the framework here.
_logger = logging.getLogger("vllm.netrsn_turbo")
_logged = set()


def log_once(key, message, *args):
    # Logging and Python bookkeeping must stay outside compiled graphs.
    # The operator/patch caller already uses torch; package imports stay lazy.
    import torch

    if torch.compiler.is_compiling():
        return
    pid = os.getpid()
    event = (pid, key)
    if event in _logged:
        return
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_initialized() and npu.is_current_stream_capturing():
        return
    _logged.add(event)
    _logger.info("[NetrsnTurbo][pid=%d] " + message, pid, *args)
