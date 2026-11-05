# v0.30.0 LoRA 三路 AOT 的 NetrsnTurbo 适配

此目录把当前 `patch_lora_aot.py` 的 `base / base_one / lora` 三路 AOT 迁移到
NetrsnTurbo 的 `TurboManager.register_patch()` / `apply_patches()` 机制，包含
DeepSeek-V4 的启用条件。它是可合入现有 NetrsnTurbo 工程的增量交付目录。

## 文件布局

```text
netrsn_turbo_v30_lora_aot/
├── netrsn_turbo/
│   ├── turbo/version_v0300/lora_aot/patches.py
│   └── turbo_manager/version_v0300/turbo_lora_aot.py
├── tests/
├── pyproject.toml
└── README.md
```

- `patches.py` 合并 AOT 分流、缓存读写和图预热逻辑；导入不会安装补丁，
  `build_patches()` 只构造替换对象。
- `turbo_lora_aot.py` 是统一入口：通过 TurboManager 注册、应用补丁，
  并处理重复加载和 VA 原生补丁的加载顺序。版本、硬件和补丁组合由外层选择。
- 不需要部署或导入 `vllm_ascend.patch.worker.patch_lora_aot`，也不需要
  `_lora_aot_graph.py`。

## 接入现有工程

1. 将本目录的 `netrsn_turbo/` 内容按相对路径合入实际 NetrsnTurbo 包。
   保留工程已有的上级 `__init__.py` 和版本分发逻辑；若 `version_v0300`
   已存在，也保留其原有初始化内容。本目录没有包含完整 TurboManager 引擎。
2. 在实际工程的 **vLLM 0.30.0 版本分支**加入：

   ```python
   import netrsn_turbo.turbo_manager.version_v0300.turbo_lora_aot  # noqa: F401
   ```

3. 保持 `vllm → vllm_ascend → turbo_manager` 的加载顺序，在模型和编译 wrapper
   创建之前加载入口。spawn worker 也必须通过工程已有的 `special_init.py`
   重新走到这个入口；父进程打补丁不会自动覆盖新解释器。
4. 如果之前手工安装了 VA 版本，从 `vllm_ascend/patch/worker/__init__.py`
   移除它的 `import vllm_ascend.patch.worker.patch_lora_aot`，然后重启。
   原生与 Turbo 入口的选择由外层负责，本入口不再检查原生安装状态或补丁冲突。
   Turbo 先安装、同版原生文件后导入时，其幂等标记会使原生安装跳过。
5. 停用此前 v0.23/v0.25 的 LoRA ACL graph 注册入口；其 compiler/graph
   改写与本版本不能叠加。其他无关的 Turbo 补丁可保留。

入口导入时自动执行 `apply_lora_aot_patch()`；若工程使用显式注册，也可调用该函数，
重复调用不会增加包装层。不依赖 `ADAPTATION_PKG_ID`，因此服务先启动、运行时再加载
adapter 的场景也能安装补丁。实际是否启用三路分流仍由下述模型配置决定。

把交付目录简单添加到 `PYTHONPATH`，不一定能覆盖已安装的同名普通 Python 包；
应把新增文件合入实际使用的 `netrsn_turbo` 包。

## 启用范围

实现适配 vLLM v0.30.0 的编译接口；入口不再读取版本号或检查硬件能力，由工程外层
决定是否导入。模型级 `enabled(config)` 保留以下实际运行条件。

模型白名单：

| 模型 | 配置判定 |
| --- | --- |
| Qwen3.5 dense | `hf_text_config.model_type == "qwen3_5_text"` |
| DeepSeek-V4 | `hf_text_config.model_type == "deepseek_v4"` |
| Magistral/Mistral3 | 文本类型为 `mistral`，且外层类型为 `mistral3` 或模型路径包含 `magistral` |

同时需要：LoRA 已启用、`VLLM_USE_AOT_COMPILE=1`、Model Runner V1、
`fully_sharded_loras=False`、`enforce_eager=False`、编译模式 `VLLM_COMPILE`、
编译后端非 `eager`、图模式非 `NONE`、`cudagraph_specialize_lora=True`。
只处理 `backbone`，不扩展 encoder、draft 模型或 MRV2。

```bash
export VLLM_USE_AOT_COMPILE=1
export VLLM_DISABLE_COMPILE_CACHE=0
export VLLM_CACHE_ROOT=/path/to/new/turbo-lora-aot-v0300-cache
```

沿用已验证的 MRV1 启动配置，增加 `--enable-lora`、adapter 参数和以下编译配置：

```text
--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_specialize_lora": true}'
```

## 分流和注册目标

| 当前 forward 的 batch | AOT 产物 |
| --- | --- |
| 全部为 base，输入 token 数为 1 | `model.base_one` |
| 全部为 base，输入 token 数大于 1 | `model.base` |
| 含任意 adapter 请求，包括 base/LoRA 混合 batch | `model.lora` |

token 数以实际 forward 输入张量为准，不是请求数。各变体独立编译、保存和加载；
部分缓存命中时，仅补编译缺失变体。普通 profiling 保留 dummy adapter，图捕获前
以零活跃 LoRA 预热两个 base 变体，捕获时保持 descriptor 对应的 LoRA 状态。

TurboManager 注册以下六个目标：

| 目标 | 用途 |
| --- | --- |
| `vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__` | 安装模型级三路 dispatcher |
| `vllm.compilation.decorators._try_load_aot_compiled_fn` | 按变体加载 AOT 缓存 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run` | 传递预热/捕获的 LoRA 数量 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner.maybe_dummy_run_with_lora` | 设置匹配的 dummy LoRA 状态 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner.capture_model` | 捕获前预热 base/base_one |
| `vllm_ascend.utils.adapt_patch` | 原生 platform/worker 补丁返回后重应用同一组已注册对象 |

原始函数保存在闭包或 dispatcher 实例属性内，避免 TurboManager 的 `sys.modules`
引用传播把原始函数替换成补丁自身。每个进程只构造一组稳定的替换对象；重复调用
重应用原有对象，不增加包装层。注册和应用异常原样抛出，重试复用已构建的对象。
入口不额外校验重复目标或安装结果。

安装成功日志：

```text
[Turbo AOT] v0.30.0 base/base_one/lora patches installed via TurboManager
LoRA AOT variant lora is ready (compiled)
LoRA AOT variant base is ready (compiled)
LoRA AOT variant base_one is ready (compiled)
```

实际变体日志顺序取决于预热过程；复用缓存时为 `(loaded)`。

## 验证和边界

在本目录运行 CPU 契约测试：

```bash
python -m pytest --noconftest -q tests
python -m ruff check .
python -m ruff format --check .
```

本地结果：115 项 CPU 测试通过。测试模拟 TurboManager 文档规定的属性替换和模块引用传播，覆盖三路分流、
缓存重载、编译失败恢复、图预热、模型运行条件、外层运行环境选择、重复安装、原生晚覆盖及安装失败重试。
当前工作区没有完整私有 NetrsnTurbo 引擎，也没有 NPU；仍需在目标环境验证真实
TurboManager、spawn 加载、AOT 保存/加载、ACL graph 捕获/重放、输出正确性和性能。

DeepSeek-V4 的接入只扩展三路 AOT 分流。adapter 的具体 `target_modules`、
量化及 MoE 通信组合必须本来就被所用 VA 版本正确支持。此适配不新增模型层的 LoRA
计算能力，也不迁移旧版本的 attention、GraphParams 或 RMSNorm 补丁。

验收至少包含 base → adapter A → adapter B → base、单 token base、混合 batch
以及重启加载三份缓存。性能比较应使用相同设备、模型、请求分布和并行参数，分别
记录纯 base batch 和混合 batch；混合 batch 仍走 LoRA 分支，三路 AOT 本身不保证
所有场景的 base 请求延迟与未开启 LoRA 的服务完全一致。
