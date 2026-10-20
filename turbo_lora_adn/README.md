# v0.23 / 310P ADN LoRA 的 Turbo 适配

此目录是按 `patch_mesim.md` 中 TurboManager 机制组织的独立交付目录。将其中 `netrsn_turbo/` 下的文件按相对路径合并进 NetrsnTurbo 项目，再在 v0.23 分支导入 ADN 入口。

本目录自包含 TP、token/group/sampler 元数据兼容及 ADN 投影，不需要复制共享 GEMM 交付目录，也不导入之前在 vllm-ascend 中自建的 LoRA patch 文件。

## 1. 文件布局

```text
turbo_lora_adn/
├── netrsn_turbo/
│   ├── turbo/version_0230/lora_adn_310p/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   └── patches.py
│   └── turbo_manager/
│       └── turbo_lora_adn_0230_310p.py
└── tests/test_turbo_lora_integration.py
```

- [patches.py](netrsn_turbo/turbo/version_0230/lora_adn_310p/patches.py)：ADN 直写投影、运行时 shape 调度及公共兼容处理。
- [bootstrap.py](netrsn_turbo/turbo/version_0230/lora_adn_310p/bootstrap.py)：版本/310P 门禁、TurboManager 注册、后置重应用、幂等及方案互斥。
- [Turbo 入口](netrsn_turbo/turbo_manager/turbo_lora_adn_0230_310p.py)：导入时立即调用 `apply_lora_patch()`。

交付目录未提供已有上级包的 `__init__.py`，不会用空文件替换现有路由。需要合并到实际项目，不能只增加 PYTHONPATH 就假定覆盖已安装的同名普通包。

## 2. 接入方式

1. 将 `netrsn_turbo/` 内新增文件合并到实际 NetrsnTurbo 项目，保留现有上级 `__init__.py`。
2. 确保已安装与目标 CANN 环境匹配的 `adn_custom_ops` 及对应算子实现。
3. 在当前 v0.23 版本分发/LoRA 条件分支中停用旧的同目标 LoRA 入口，仅导入：

   ```python
   from netrsn_turbo.turbo_manager import turbo_lora_adn_0230_310p  # noqa: F401
   ```

4. 保留 `vllm` → `vllm_ascend` → `turbo_manager` 的框架加载顺序。
5. 保留 `special_init.py` 在 spawn 子进程中重新导入 `netrsn.core.turbo` 的机制，确保每个子进程也执行所选入口。
6. 重启服务，再测试 adapter 动态加载、卸载和请求切换。

不要同时导入共享 GEMM 入口。切换后端需要更换入口并启动新进程；代码会在投影替换及 ADN 注册前检查进程级方案归属。

本入口不要求 `ADAPTATION_PKG_ID` 非空，避免服务先启动、adapter 后动态加载时漏打补丁。项目已有的可靠 `GlobalContext.get_is_lora()` 路由条件可继续用于控制入口导入。

## 3. 停用旧的原生 LoRA 导入

如果已使用前一轮 VA 原生补丁，在 `vllm_ascend/patch/worker/__init__.py` 停用之前新增的三个导入并重启：

```python
# import vllm_ascend.patch.worker.patch_lora_column_parallel_310
# import vllm_ascend.patch.worker.patch_lora_prefill_310
# import vllm_ascend.patch.worker.patch_lora_adn_310
```

使用未安装这些手工补丁的原始 VA 时无需此步骤。VL、RoPE 等模型启动兼容修复仍可保留；本目录只迁移 LoRA 范围。

代码会报告已检测到的旧原生 LoRA 标记，避免元数据和投影重复包装。版本分支里其他旧 Turbo LoRA 注册也应停用；不需要停用无关的量化或权重加载补丁。

## 4. 通过 TurboManager 注册的目标

| 目标 | 适配行为 |
| --- | --- |
| `column_parallel_linear._mcp_apply` | 修复普通 LoRA TP rank 重复聚合 |
| `PunicaWrapperNPU.__init__` | 初始化 token、group、sampler 缓冲区 |
| `PunicaWrapperNPU._update_base_metadata` | 清理 token/sampler 尾部为 `-1` |
| `PunicaWrapperNPU._update_prefill_metadata` | 清理无效 group：count=0、slot=-1 |
| `PunicaWrapperNPU.add_lora_linear` | 联合 A/B 投影直接更新最终 y，保留真实 offset/scale |
| `PunicaWrapperNPU.add_lora_logits` | 使用 sampler 映射的联合投影 |
| `PunicaWrapperNPU._shrink_prefill/_decode` | 单矩阵投影，清已有 rank 缓冲区以满足覆写语义 |
| `PunicaWrapperNPU._expand_prefill/_decode` | 单矩阵投影，保留累加或覆写 |
| `PunicaWrapperNPU._expand_slice_prefill/_decode` | 单矩阵投影，保留目标切片和实际偏移 |
| `vllm_ascend.utils.adapt_patch` | 原生补丁返回后恢复上述稳定替换对象 |

完整目标前缀分别为 `vllm.lora.layers.column_parallel_linear` 和 `vllm_ascend.lora.punica_npu.PunicaWrapperNPU`。

目标属性通过 `TurboManager.register_patch()`、`TurboManager.apply_patches()` 应用。已有引用的传播由 TurboManager 提供；每个进程只构建一次替换对象，后续恢复不会增加 wrapper 层数。

投影直接注册同名函数，例如：

```python
((f"{_PUNICA_TARGET}.add_lora_linear", add_lora_linear),)
((f"{_PUNICA_TARGET}.add_lora_logits", add_lora_logits),)
```

没有 `_get_adn_operator()`、工厂生成的 dispatch 或投影 wrapper。`build_patches()` 就地注册模块级 `_adn_dispatch`；投影通过公共 `_call_adn()` 准备元数据，再显式调用 `torch.ops.vllm.netrsn_adn_lora_310p_0230`。只有初始化、元数据更新和 TP 路由需要保留原函数，因此仍使用包装。

## 5. ADN 直写与图适配

普通累加路径直接把最终 `y`、真实 adapter 索引、A/B、scale 和 offset 交给 ADN，不建立宽临时 delta，不在外部执行 where 和 add_。

运行时沿用原 ADN 两种 API：

```text
grouped 候选且 G < min(x行数, y行数)
    → ADN grouped API（lora_order / lora_batch_num）
其他情况
    → ADN token API（真实 indices，含 -1）
```

G 是固定 group 数组容量，不是实际 adapter 数。该规则保留此前分组形状报错的修复，避免小 token 图预热触发 ADN tiling 参数错误；转 token API 仍然使用 ADN。

Turbo 版本注册的调度算子名为：

```text
torch.ops.vllm.netrsn_adn_lora_310p_0230
```

它声明 `mutates_args=["y"]`、PrivateUse1 实现和 fake 实现。shape 判断位于不透明算子的真实执行函数内，避免无 guard 编译将首次选择的分支固定到所有尺寸。它不是另写的 ADN 计算内核，内部仍调用安装的 `adn_batch_gather_matmul_`。

原 VA 版本的 `ascend_adn_lora_310` 名称没有复用，避免与旧注册发生名称碰撞。图中应检查新的 Turbo 调度节点名。

## 6. 行为边界

- 门禁仅允许 vLLM `0.23.x` 和 310P；检查通过之后才构建 payload、导入 ADN 和注册算子。
- 未增加单槽位或 FP16 的 Python 限制，但实际 dtype、布局和槽位支持仍由安装的私有 ADN 版本决定。
- `y_slice_size=1` 保留原私有 ABI 约定，不当成输出列数。
- 动态加载继续更新驻留 A/B；不合并基础 W，不缓存另一份推理权重。
- 保留真实 scale 和 `add_inputs=False` 的目标切片覆写语义。
- fully-sharded shrink 与 expand 之间的通信保持，不能跨通信融合。
- v0.23 的 `is_prefill=True` 也用于 decode，因此还需由实际 shape 决定能否使用 grouped。
- 无 GEMM 自动回退；ADN 缺包、符号缺失或真实内核异常会按原错误上报。
- 没有修改 ADN whl 或 CANN run 包中的二进制实现。

## 7. 日志和验证

每个模型执行进程应完成安装：

```text
[Turbo LoRA] backend=adn installed on vLLM 0.23 / Ascend310P via TurboManager
```

eager/warmup 首次调用返回后，会有一次 `[Turbo ADN LoRA] ... dispatch returned`。编译期间抑制该日志；它不证明设备异步完成，也不是图重放计数。

进程级检查：

```python
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

state = PunicaWrapperNPU._netrsn_turbo_lora_310p_state
assert state.owner == "adn"
```

替换版本后首次验证可临时设置 `VLLM_DISABLE_COMPILE_CACHE=1`，这不关闭 ACLGraph。

在本交付目录 `turbo_lora_adn/` 下运行测试：

```bash
pytest --noconftest -q --import-mode=importlib tests/test_turbo_lora_integration.py
```

本地结果：33 项通过，1 项 GEMM 专属用例跳过。覆盖注册生命周期、310P/版本门禁、直接更新输出、grouped/token 形状调度、split/logits、元数据清尾和动态权重，以及直接函数注册和全部 split 阶段的覆写/累加行为。

本地测试使用 CPU Torch 和模拟 ADN、TurboManager 契约。当前工作区没有完整 NetrsnTurbo 源码和 310P，实际 Turbo 注册、spawn、CANN/ADN 数值与 ACLGraph 均需在目标环境继续验证。

性能测试应与共享 GEMM 服务分开使用同等设备和请求配置；4K 输入、4K 输出、2 并发分别记录 TTFT、TPOT、总吞吐和完成时间。此前旧 ADN bridge 的 10.4 tokens/s 不代表这次 Turbo 迁移版本的结果。
