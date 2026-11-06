# v0.30.0 DS LoRA 与三路 AOT 的 NetrsnTurbo 适配

此目录包含 DS LoRA v2 的 Python 补丁、内置 `turbo_ops` 算子及
`base / base_one / lora` 三路 AOT，统一通过 NetrsnTurbo 的
`TurboManager.register_patch()` / `apply_patches()` 安装。
部署只需本目录，不再需要同时合入另一份 DS 交付目录。

## 文件布局

```text
netrsn_turbo_v30_lora_aot/
├── netrsn_turbo/
│   ├── turbo/version_v0300/
│   │   ├── vllm/compilation/
│   │   │   ├── wrapper.py
│   │   │   └── decorators.py
│   │   └── vllm_ascend/
│   │       ├── ascend_forward_context.py / envs.py
│   │       ├── lora/
│   │       ├── models/deepseek_v4/model.py
│   │       ├── ops/fused_moe/
│   │       └── worker/model_runner_v1.py
│   ├── turbo_manager/version_v0300/
│   │   ├── turbo_lora_aot.py
│   │   └── turbo_ds_lora_aot.py
│   └── turbo_ops/
│       ├── __init__.py / lora.py / _runtime.py / _logging.py
│       ├── csrc/
│       └── provenance.json / LICENSE
├── build_support/
├── setup_turbo_ops.py
├── tests/
├── pyproject.toml
└── README.md
```

- 补丁实现按被修改模块在 vLLM / vLLM Ascend 项目中的原层级放置，
  不再使用 `lora_aot/patches.py` 聚合文件。导入实现模块不会安装补丁，
  各 `wrap_*()` 函数只构造对应的替换方法。
- `turbo_lora_aot.py` 是独立 AOT 入口：直接构造五个替换对象，逐项注册，再调用
  `TurboManager.apply_patches()`。版本、硬件、补丁组合和调用次数由外层管理。
- `turbo_ds_lora_aot.py` 是联合入口，直接展开所有属性赋值和 24 个目标的
  `TurboManager.register_patch()` 调用，组合 DS worker 改写和 AOT 包装后统一应用一次。
  注册逻辑全部位于 `turbo_manager`，没有独立安装器或补丁映射清单。
- DS 和 AOT 实现共用 `version_v0300`，DS 的源码改写与 AOT wrapper 位于同一个
  `vllm_ascend/worker/model_runner_v1.py`。内部导入不经过旧版本目录或额外的
  `ds_lora` 包。
- 不需要部署或导入 `vllm_ascend.patch.worker.patch_lora_aot`，也不需要
  `_lora_aot_graph.py`。

相对于 `netrsn_turbo/turbo/version_v0300/` 的实现路径与原项目对应关系：

| 原项目源码 | Turbo 实现路径 | 内容 |
| --- | --- | --- |
| `vllm/vllm/compilation/wrapper.py` | `vllm/compilation/wrapper.py` | 三路 dispatcher、编译与保存、wrapper 初始化 |
| `vllm/vllm/compilation/decorators.py` | `vllm/compilation/decorators.py` | 单 token 动态标记适配、按变体加载缓存 |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | `vllm_ascend/worker/model_runner_v1.py` | DS 数量参数改写、dummy LoRA 状态和图捕获前预热 |
| `vllm-ascend/vllm_ascend/lora/` | `vllm_ascend/lora/` | DS Punica、MoE、量化和算子适配 |
| `vllm-ascend/vllm_ascend/models/deepseek_v4/model.py` | `vllm_ascend/models/deepseek_v4/model.py` | adapter 名称映射与跳过前缀 |
| `vllm-ascend/vllm_ascend/ops/fused_moe/` | `vllm_ascend/ops/fused_moe/` | 专家计算和 token 分发适配 |

## 接入现有工程

1. 将本目录的 `netrsn_turbo/` 和 `build_support/` 内容按相对路径合入实际工程。
   保留工程已有的上级 `__init__.py` 和版本分发逻辑；若 `version_v0300`
   已存在，也保留其原有初始化内容。本目录没有包含完整 TurboManager 引擎。
2. 在实际工程的 **vLLM 0.30.0 版本分支**加入：

   ```python
   import netrsn_turbo.turbo_manager.version_v0300.turbo_ds_lora_aot  # noqa: F401
   ```

3. 保持 `vllm → vllm_ascend → turbo_manager` 的加载顺序，在模型和编译 wrapper
   创建之前加载入口。spawn worker 也必须通过工程已有的 `special_init.py`
   重新走到这个入口；父进程打补丁不会自动覆盖新解释器。
4. 如果之前手工安装了 VA 版本，从 `vllm_ascend/patch/worker/__init__.py`
   移除它的 `import vllm_ascend.patch.worker.patch_lora_aot`，然后重启。
   原生与 Turbo 入口的选择、加载顺序由外层负责，本入口不检查或设置原生安装标记。
5. 停用原 DS 独立入口、AOT 独立入口及此前 v0.23/v0.25 的 LoRA ACL graph
   注册入口，统一由上述联合入口安装。其他无关的 Turbo 补丁可保留。
6. 按下文将 `turbo_ops` 构建 helper 接入主仓 `setup.py`，在匹配运行环境中构建 wheel。

联合入口导入时自动执行本文件内的 `apply_ds_lora_aot_patch()`。外层负责每个 worker 的单次安装，
不再额外调用该函数或 reload 入口。不依赖 `ADAPTATION_PKG_ID`，因此服务先启动、运行时再加载
adapter 的场景也能安装补丁。实际是否启用三路分流仍由下述模型配置决定。

把交付目录简单添加到 `PYTHONPATH`，不一定能覆盖已安装的同名普通 Python 包；
应把新增文件合入实际使用的 `netrsn_turbo` 包。

## 联合安装的行为

联合入口先对原始 `_dummy_run` 做 DS 单点源码改写，再构造 AOT 包装，
最终 `_dummy_run` 只注册一次。其余 19 个 DS 目标和 4 个 AOT 目标一起注册，
最后执行一次 `TurboManager.apply_patches()`，随后更新 W8A8 的冻结注册项。
仅使用三路 AOT 的模型可由外层选择 `turbo_lora_aot` 入口，安装五个 AOT 目标。
这两个入口由外层择一使用；入口不添加版本、硬件或重复安装检查。

LoRA 数量统一按以下优先级传递给 `_dummy_run` 方法体及 dummy adapter context：

| 场景 | `num_active_loras` |
| --- | --- |
| capture 前显式预热 `base` / `base_one` | 0，优先于 `is_profile=True` |
| 普通首次 profiling | `lora_config.max_loras` |
| 普通图预热、捕获 | 调用方传入的数量，缺省为 0 |
| 模型配置未启用 AOT | 保持 DS 改写后的行为 |

普通 profiling 因而包含 LoRA 编译和内存估算；base 两个变体仍在图捕获前预热。
方法体的图描述与实际 dummy adapter 使用同一个数量，不依赖原 VA 强制激活
`max_loras` 的旧逻辑。先加载独立 AOT 再做 DS 源码改写会破坏 DS 要求的包装层结构；
使用联合入口避免这一顺序冲突。

## 内置算子构建

DS LoRA 的十个原生接口通过 `netrsn_turbo.turbo_ops` 调用，算子源码、Meta 注册、
运行时加载、来源记录和许可证均已包含在本目录，无须另装 `ops_repo`。
主仓 `setup.py` 保留现有参数，将构建 helper 合入：

```python
from setuptools import setup
from build_support import configure_turbo_ops

setup(**configure_turbo_ops(existing_setup_kwargs))
```

`existing_setup_kwargs` 应包含主仓完整的 `packages` 列表，涵盖本次新增的
`version_v0300` 子包。helper 保留已有扩展和构建命令，并补充 `turbo_ops` 的分发内容。
它按项目根目录下的 `netrsn_turbo/turbo_ops/` 定位源码；采用 `src/` 布局的主仓应调整定位。

在目标 Linux 运行环境构建，需匹配的 torch、torch_npu、CANN AscendC 工具链、
CMake ≥ 3.19、C++17 编译器、setuptools 和 wheel。A2/A3 应分别构建，芯片型号
按实际设备选择。主仓构建示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# A2 示例；910B2 设备改为 ascend910b2。
TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910b \
  TURBO_OPS_SOC_VERSION=ascend910b1 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist/ascend910b

# A3 示例。
TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910_93 \
  TURBO_OPS_SOC_VERSION=ascend910_9391 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist/ascend910_93
```

在本交付目录单独验证算子构建，可使用相同环境变量运行
`python setup_turbo_ops.py bdist_wheel`。该验证入口只打包算子组件；生产部署应将
补丁和算子一并构建到真实 NetrsnTurbo 的 wheel 中。

未设置 `TURBO_OPS_BUILD=1` 时不编译内核，所得纯 Python 包不能运行 DS LoRA 原生路径。
当前算子构建支持 A2/A3 的普通 wheel，不支持 A5/310P、editable 或 `build_ext --inplace`。
运行时库及 `_build_info.json` 位于 `netrsn_turbo/turbo_ops/`，CANN 和 torch_npu
的动态库仍由目标环境提供。第一个 Punica wrapper 初始化原生组件，须在编译或图捕获前完成。

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

AOT 涉及以下五个目标；联合入口将共享的 `_dummy_run` 合并为一次注册：

| 目标 | 用途 |
| --- | --- |
| `vllm.compilation.wrapper.TorchCompileWithNoGuardsWrapper.__init__` | 安装模型级三路 dispatcher |
| `vllm.compilation.decorators._try_load_aot_compiled_fn` | 按变体加载 AOT 缓存 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run` | 传递预热/捕获的 LoRA 数量 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner.maybe_dummy_run_with_lora` | 设置匹配的 dummy LoRA 状态 |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner.capture_model` | 捕获前预热 base/base_one |

原始函数保存在闭包或 dispatcher 实例属性内，避免 TurboManager 的 `sys.modules`
引用传播把原始函数替换成补丁自身。入口不保存安装状态，不处理重入或重复安装，
也不包装 `adapt_patch` 做晚覆盖后的重应用。注册和应用异常原样抛出。

AOT 变体日志示例：

```text
LoRA AOT variant lora is ready (compiled)
LoRA AOT variant base is ready (compiled)
LoRA AOT variant base_one is ready (compiled)
```

实际变体日志顺序取决于预热过程；复用缓存时为 `(loaded)`。

## 验证和边界

在本目录运行 CPU 契约测试：

```bash
python -m pytest --noconftest --ignore=tests/npu -q tests
python -m ruff check .
python -m ruff format --check .
```

测试模拟 TurboManager 文档规定的属性替换和模块引用传播，覆盖三路分流、
缓存重载、编译失败恢复、图预热、模型运行条件、外层运行环境选择、直接注册及安装异常传播。
联合回归还执行真实 DS worker 源码改写与 AOT 包装，检查 profiling、base 预热、
LoRA 图捕获的数量一致性，以及单次注册和统一应用。DS 与算子测试直接读取本目录中的
实现；外部来源仓库仅用于可选的源码对照，不是运行时依赖。
当前工作区没有完整私有 NetrsnTurbo 引擎，也没有 NPU；仍需在目标环境验证真实
TurboManager、spawn 加载、AOT 保存/加载、ACL graph 捕获/重放、输出正确性和性能。

本次目录融合保留 DS v2 已有的 LoRA 计算、量化和 MoE 通信适配，AOT 部分负责三路
编译与图预热。adapter 的 `target_modules` 及并行配置仍需符合 DS 补丁和目标 VA
版本的支持范围；不迁移旧版 attention、GraphParams 或 RMSNorm 补丁。

在目标设备安装编译后的 wheel 后，可从交付目录之外执行 NPU 回归，避免源码目录
遮蔽 wheel 内的原生库：

```bash
cd /tmp
TURBO_OPS_RUN_NPU_TESTS=1 TURBO_OPS_TEST_DEVICE=npu:0 \
  python -m pytest /path/to/netrsn_turbo_v30_lora_aot/tests/npu --import-mode=importlib -q
```

本地 CPU 测试不能替代原生编译、算子数值、真实多卡模型和性能验证。

验收至少包含 base → adapter A → adapter B → base、单 token base、混合 batch
以及重启加载三份缓存。性能比较应使用相同设备、模型、请求分布和并行参数，分别
记录纯 base batch 和混合 batch；混合 batch 仍走 LoRA 分支，三路 AOT 本身不保证
所有场景的 base 请求延迟与未开启 LoRA 的服务完全一致。
