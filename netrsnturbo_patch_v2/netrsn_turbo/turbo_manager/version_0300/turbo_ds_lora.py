# SPDX-License-Identifier: Apache-2.0
"""NetrsnTurbo integration entry point for DS LoRA v2 on vLLM 0.30.x.

Source: vllm-ascend-ds-v2 v30_lora_v2, commit 5dd159d7d2ac.
This delivery replaces the previous DS LoRA payload in a fresh worker.

Merge this delivery's netrsn_turbo/ tree into NetrsnTurbo (preserve existing
parent __init__.py files). In turbo_manager/__init__.py, place a 0.30 branch
before the existing >=0.23 branch, retaining other existing patch imports:

    if Version("0.30.0") <= Version(vllm.__version__) < Version("0.31.0"):
        from .version_0300 import turbo_ds_lora
    elif ...:
        ...

Version is packaging.version.Version. The normal startup order remains:
vllm -> vllm_ascend -> netrsn_turbo.turbo_manager. NetrsnCore special_init.py
must repeat that import in spawn workers, as described in patch_mesim.md.

ADAPTATION_PKG_ID follows the documented early LoRA gate. For deployments
whose LoRA flag is provided by GlobalContext instead, call apply_lora_patch()
explicitly in that existing LoRA branch before constructing the engine.
The patch supports A2 and A3 deployments; its communication policy selects
AllGather for A2 LoRA + EP and AlltoAll for A3 LoRA + EP.

Build and package netrsn_turbo.turbo_ops in the worker's matching
CANN/torch/torch_npu environment. This built-in native component supplies all
ten LoRA APIs; no external ops_repo package is required. The first Punica
wrapper validates its public APIs and calls turbo_ops.initialize(), then
caches the backend before model execution or graph capture.
No-LoRA imports do not load turbo_ops. The native component currently targets
A2/A3; this integration does not add A5 or 310P kernel support.
"""

import os

from netrsn_turbo.turbo.version_0300.ds_lora import apply_lora_patch

if os.getenv("ADAPTATION_PKG_ID", ""):
    apply_lora_patch()
