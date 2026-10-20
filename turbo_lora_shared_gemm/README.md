# v0.23 / 310P 共享 GEMM 的 Turbo 适配

此目录是按 `patch_mesim.md` 中 TurboManager 机制组织的独立交付目录。将其中 `netrsn_turbo/` 下的文件按相对路径合并进 NetrsnTurbo 项目，再在 v0.23 分支导入本方案入口。

本目录不修改 TurboManager 引擎，不导入之前在 vllm-ascend 中自建的 LoRA patch 文件，也不依赖另一个 ADN 适配目录。

## 1. 文件布局

```text
turbo_lora_shared_gemm/
├── netrsn_turbo/
│   ├── turbo/version_0230/lora_shared_gemm_310p/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   └── patches.py
│   └── turbo_manager/
│       └── turbo_lora_shared_gemm_0230_310p.py
└── tests/test_turbo_lora_integration.py
```

- [patches.py](netrsn_turbo/turbo/version_0230/lora_shared_gemm_310p/patches.py)：计算及元数据实现，`build_patches()` 只返回目标路径和替换函数。
- [bootstrap.py](netrsn_turbo/turbo/version_0230/lora_shared_gemm_310p/bootstrap.py)：设备/版本门禁、TurboManager 注册、原生补丁之后重应用、幂等与方案互斥。
- [Turbo 入口](netrsn_turbo/turbo_manager/turbo_lora_shared_gemm_0230_310p.py)：导入时立即调用 `apply_lora_patch()`。

交付目录没有提供已有上级包的 `__init__.py`，以免覆盖 NetrsnTurbo 的现有路由。它需要合并到实际项目中，不能只把整个交付目录加到 PYTHONPATH 就假定覆盖已安装的同名普通包。

## 2. 接入方式

1. 将 `netrsn_turbo/` 内新增文件合并到实际 NetrsnTurbo 项目，保留原有上级 `__init__.py`。
2. 在当前 v0.23 的版本分发/LoRA 条件分支中，停用旧的同目标 LoRA 注册入口，只导入本方案：

   ```python
   from netrsn_turbo.turbo_manager import turbo_lora_shared_gemm_0230_310p  # noqa: F401
   ```

3. 保留框架加载顺序：`vllm` → `vllm_ascend` → `turbo_manager`。入口内部会先检查 vLLM 与设备，再导入 payload。
4. 保留 `special_init.py` 在 spawn 子进程中重新导入 `netrsn.core.turbo` 的现有机制，确保子进程也经过上述版本入口。
5. 重启服务。单槽优化对应实际权重槽位数为 1，启动配置通常为 `--enable-lora --max-loras 1`，rank 按 adapter 实际配置。

两个方案只能选择一个。切换到 ADN 时，更换入口并启动新进程，不能在同一运行进程里叠加两套注册。

本入口不依赖 `ADAPTATION_PKG_ID` 非空：先启用 LoRA 服务、再通过运行时 API 加载 adapter 的场景也需要先打补丁。如果项目已有可靠的 `GlobalContext.get_is_lora()` 条件，可以继续把入口放在该条件内。

## 3. 迁移时停用之前的原生 LoRA 导入

使用原始 vllm-ascend 时无需此步骤。如果已经应用本次会话之前的原生 VA LoRA 补丁，应停用 `vllm_ascend/patch/worker/__init__.py` 中新增的这三个导入，再重启：

```python
# import vllm_ascend.patch.worker.patch_lora_column_parallel_310
# import vllm_ascend.patch.worker.patch_lora_prefill_310
# import vllm_ascend.patch.worker.patch_lora_adn_310
```

其他模型启动修复，例如 VL、RoPE，不由本目录替换。这里仅迁移 LoRA 计算及其公共兼容处理。

代码会检测之前原生 LoRA 补丁的标记并报告冲突，防止悄悄叠加包装或把 GEMM 测试跑成 ADN。已有的旧 Turbo LoRA 注册也应在版本分支停用；其他量化、权重加载等无关注册继续保留。

## 4. 适配内容

| TurboManager 目标 | 行为 |
| --- | --- |
| `vllm.lora.layers.column_parallel_linear._mcp_apply` | 普通非 fully-sharded、TP>1 使用已有同步路径，避免重复聚合完整 rank |
| `PunicaWrapperNPU.__init__` | token-to-slot 缓冲区初始化为 `-1` |
| `PunicaWrapperNPU._update_base_metadata` | 有效 token 之后清尾为 `-1` |
| `PunicaWrapperNPU._shrink_prefill` | 单槽共享 A 矩阵，保留 FP32 scratch、scale 和覆写语义 |
| `PunicaWrapperNPU._expand_prefill` | 单槽共享 B 矩阵，保留累加/覆写语义 |
| `PunicaWrapperNPU._expand_slice_prefill` | 保留输出切片和基础模型/padding 掩码 |
| `vllm_ascend.utils.adapt_patch` | 原生 platform/worker 补丁返回后，检查并恢复上述已注册替换 |

完整 Punica 类路径为 `vllm_ascend.lora.punica_npu.PunicaWrapperNPU`。

所有目标函数通过 `TurboManager.register_patch()` 和 `TurboManager.apply_patches()` 应用，利用其已有的 `sys.modules` 引用传播能力。每个进程只构建并注册一套稳定替换对象；后续重应用不会再套一层包装。

## 5. 行为边界

- 仅在 vLLM `0.23.x` 且 `vllm_ascend.utils.is_310p()` 为真时应用。0.22、0.24 以及其他设备直接跳过，不按 `>=0.23` 无限扩展。
- 单槽条件是 `weights.shape[0] == 1`；多槽仍保留 gather/einsum。
- 不需要 `adn_custom_ops`，没有直接写入 A/B 融合算子的路径。
- 基础 W 不变；动态 adapter 权重继续由 vLLM 原地更新，forward 读取当前 A/B。
- 不启用 LoRA 时不进入这些投影；启用 LoRA 后请求基础模型，仍有先计算投影、再掩码清零的开销。
- v0.23 V1 的 `is_prefill=True` 也覆盖 decode，这些方法名称不表示仅处理真实 prefill。
- fully-sharded 路由及其通信保留，不跨 A/B 中间通信做融合。

## 6. 检查与验证

每个实际执行模型的进程应完成本方案安装。安装日志：

```text
[Turbo LoRA] backend=shared_gemm installed on vLLM 0.23 / Ascend310P via TurboManager
```

可在目标进程检查所选方案：

```python
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

state = PunicaWrapperNPU._netrsn_turbo_lora_310p_state
assert state.owner == "shared_gemm"
```

在本交付目录 `turbo_lora_shared_gemm/` 下运行测试：

```bash
pytest --noconftest -q --import-mode=importlib tests/test_turbo_lora_integration.py
```

本地结果：18 项通过，5 项 ADN 专属用例跳过。覆盖注册门禁、重复应用、原生补丁晚覆盖、方案互斥、TP 分支、元数据清尾和 CPU 数值行为。

测试使用 CPU Torch，并模拟文档中 TurboManager 的目标替换/引用传播契约。当前工作区没有 NetrsnTurbo 的完整引擎源码，也没有真实 310P，因此需要在目标环境继续验证实际 TurboManager、spawn 加载链、ACLGraph 和吞吐。此前 VA 方案的测试结果不能直接当作这次 Turbo 接入已经在设备上验收。
