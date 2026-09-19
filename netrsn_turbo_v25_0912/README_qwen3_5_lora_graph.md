# v0.25.1 Qwen3.5/Magistral LoRA 与 ACL graph 运行时补丁

根据 `patch_mesim.md` 的 TurboManager 机制，将
`patch_qwen3_5_dense_lora.py` 和 `patch_lora_acl_graph.py` 转换成按真实
vLLM / vLLM Ascend 源码层级组织的运行时补丁。原始 `vllm/`、
`vllm-ascend/` 和测试文件均不修改。

## 接入

将 `netrsn_turbo_v25_0912` 目录下的文件按同名路径合入实际工程的 `netrsn_turbo`
包。复用已有的
`__init__.py`；新建子包按工程打包规则补齐 `__init__.py`，不要覆盖原有
初始化逻辑。

在 TurboManager 的 `version_v0250` 分发入口中只增加统一入口：

```python
import netrsn_turbo.turbo_manager.version_v0250.turbo_lora_acl_graph  # noqa: F401
```

ACL graph 入口会先安装 Dense LoRA 依赖及 gated RMSNorm 算子边界，再安装图与
AOT 补丁，调用方不应再单独自动调用 dense LoRA 补丁。

Qwen3.5/Magistral LoRA 和 ACL graph 的自动启用条件沿用 v0.23 示例：
`ADAPTATION_PKG_ID` 非空且卡型为 `910B`。已有工程使用其他 LoRA 开关时，
可以在该开关中依次显式调用：

```python
from netrsn_turbo.turbo_manager.version_v0250.turbo_lora_acl_graph import (
    apply_lora_acl_graph_patch,
)

apply_lora_acl_graph_patch()
```

显式调用不检查环境变量或卡型，由调用方保证版本、硬件和 LoRA 场景正确。
补丁必须在模型实例、LoRA manager、compile wrapper 和 ACL graph 状态初始化之前应用。
spawn 子进程仍需通过工程已有的 `special_init.py` 重新导入。

## Qwen3.5/Magistral dense LoRA 映射

| 原补丁目标 | Turbo 实现位置 |
| --- | --- |
| `PunicaWrapperNPU` 的 metadata 与 no-LoRA guard | `turbo/version_v0250/vllm_ascend/lora/punica_npu.py` |
| `LoRAModelManager.__init__` 与多模态模块映射 | `turbo/version_v0250/vllm/lora/model_manager.py` |
| `WorkerLoRAManager._load_adapter` | `turbo/version_v0250/vllm/lora/worker_manager.py` |
| `AscendAttentionBackendImpl.update_graph_params` | `turbo/version_v0250/vllm_ascend/attention/attention_v1.py` |

管理入口
`turbo_manager/version_v0250/turbo_qwen3_5_dense_lora.py` 按样例在
`apply_dense_lora_patch()` 中统一调用 `TurboManager.register_patch()`；旧名称
`apply_qwen3_5_dense_lora_patch` 仅作为兼容别名保留。入口最后调用
`TurboManager.apply_patches()`
应用注册的方法包装。attention 方法
使用 `staticmethod` 保留描述符语义。MM key remap、decode `no_lora` 更新、
Base 图的 no-LoRA guard 和配置限制均保留。

模型启用条件如下：

- Qwen3.5 dense：`hf_text_config.model_type == "qwen3_5_text"`。
- Magistral/Mistral3：外层 `hf_config.model_type == "mistral3"` 且语言层
  `hf_text_config.model_type == "mistral"`。

普通 `MistralForCausalLM` 不会进入该补丁。Qwen3.5 为混合 attention metadata
增加的过滤只对 Qwen3.5 生效；Magistral 仅复用多模态语言前缀、no-LoRA guard、
ACL graph 隔离和三变体 AOT，不改变其 attention metadata。

已移除用于规避算子内部对齐检查报错的 shrink/expand rank padding，以及
`PunicaWrapperNPU.__init__` 的算子包装。shrink、expand 和 expand slice 均沿用
vLLM Ascend 原生实现，参数与算子报错原样传递。

## LoRA ACL graph 映射

| 原补丁目标 | Turbo 实现位置 |
| --- | --- |
| `GraphParams` 与 `weak_ref_workspaces` | `turbo/version_v0250/vllm_ascend/compilation/acl_graph.py` |
| `TorchCompileWithNoGuardsWrapper` | `turbo/version_v0250/vllm/compilation/wrapper.py` |
| `GPUModelRunner` 的 LoRA dummy run、warmup 与 capture | `turbo/version_v0250/vllm/v1/worker/gpu_model_runner.py` |
| `AscendRMSNormGated.forward_oot` 的 AOT 算子边界 | `turbo/version_v0250/vllm_ascend/ops/layernorm.py` |

管理入口 `turbo_manager/version_v0250/turbo_lora_acl_graph.py` 使用
`TurboManager.register_patch()` 统一注册 `GraphParams`、`weak_ref_workspaces`、
compile wrapper、AOT 加载和 model runner 方法，再调用
`TurboManager.apply_patches()` 应用。已导入的同对象引用由 TurboManager
同步传播；辅助逻辑在补丁模块内直接调用，无需向上游类新增方法。
若任一 ACL graph 状态已经初始化则拒绝应用，避免丢失状态。

设置 `VLLM_USE_AOT_COMPILE=1` 时，compile wrapper 会分别编译、加载并原子保存
LoRA、Base 和单 token Base 三份 AOT 产物（`model.lora`、`model.base`、
`model.base_one`）。缓存只命中部分变体时，其余变体会在首次使用时补编译并保存。

## gated RMSNorm AOT 修复

根据 `LAYERNORM_AOT_CHANGE_SUMMARY.md`，新增
`vllm_ascend::rms_norm_gated` 自定义算子及连续布局的 FakeTensor 实现。
`AscendRMSNormGated.forward_oot` 经由该算子调用原有 `LayerNormFn.apply`，
使 AOT 图保存稳定的 `vllm_ascend.rms_norm_gated.default` 节点，避免保存
内部 Triton wrapper 的进程内编号。普通 RMSNorm、Gemma RMSNorm、原有
Triton 内核及三种 AOT 变体的分发逻辑保持原实现。

管理入口为 `turbo_manager/version_v0250/turbo_layernorm_aot.py`，已接入
`apply_lora_acl_graph_patch()`，无需增加另一条自动导入。重复调用会跳过已安装
的方法补丁；若 Ascend 已注册同名算子，则复用其实现。需要单独启用此修复时，
可在模型创建前显式调用：

```python
from netrsn_turbo.turbo_manager.version_v0250.turbo_layernorm_aot import (
    apply_layernorm_aot_patch,
)

apply_layernorm_aot_patch()
```

此修复改变编译图，应在实际启动容器中重新生成缓存，例如使用新的持久化子目录：

```bash
export VLLM_USE_AOT_COMPILE=1
export VLLM_DISABLE_COMPILE_CACHE=0
export VLLM_CACHE_ROOT=/root/.cache/vllm/gated-rmsnorm-v1
unset VLLM_FORCE_AOT_LOAD
```

首次启动应确认每个 TP rank 保存 `model.lora`、`model.base`、`model.base_one`。
再次启动保持目录、代码、模型和配置一致，可临时设置
`VLLM_FORCE_AOT_LOAD=1`，确认三种产物均出现
`Directly load AOT compilation from path` 日志。日常运行可取消强制加载。
spawn worker 必须同样加载 Turbo，以便在读取缓存前注册该算子。

CPU 回归脚本（需要 PyTorch 2.10，使用参考内核和 TurboManager 替身）：

```bash
python netrsn_turbo_v25_0912/tests/check_layernorm_aot.py
```

该脚本检查参数与数值、输入不被修改、FakeTensor/schema、编译图边界、重复安装
及新进程中的三种 AOT 产物加载。当前 PyTorch 2.10.0 CPU 环境下，5 项单元测试
和三种测试产物的跨进程保存、加载、执行均通过；AOT 使用测试用
`SerializableCallable` 后端。CPU 检查不能代替真实 NPU 的数值、ACL Graph
捕获/重放、AscendCompiler、TP4 和 base/LoRA 混合请求验证，也不证明启动加速比例。

## 兼容性边界

不要同时加载 vLLM Ascend 原生的
`patch_qwen3_5_dense_lora.py`、`patch_lora_acl_graph.py` 与本运行时补丁；
两套实现会重复包装方法。该补丁只面向当前 vLLM
0.25.1 / vLLM Ascend 源码组合，其他版本必须重新核对方法签名和图描述符。

当前工作区已完成所有 Python 文件的语法解析与行长检查，并使用 TurboManager
替身检查注册目标、加载顺序、重复应用、静态方法调用、原始算子错误透传和
LoRA/Base 变体选择；注册路径已与上游源码核对。由于没有真实 NetrsnTurbo
与 NPU 环境，部署前仍需
验证 LoRA/Base 交替请求、FULL graph 捕获与重放、MTP、MM LoRA key remap、
无 LoRA decode、图 workspace 弱引用以及 spawn worker 重打补丁。
