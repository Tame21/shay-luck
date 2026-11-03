# NetrsnTurbo DS LoRA v2：version_0300 + turbo_ops

这是可合入 NetrsnTurbo 的交付目录。当前没有 NetrsnTurbo 主仓源码，因此在此提供
补丁、内置算子包和构建合并入口；真实主仓的版本分发及 `setup.py` 需要按下文接入。
本目录 `netrsnturbo_patch_v2` 基于新增的 `vllm-ascend-ds-v2`，原
`netrsnturbo_patch` 保留。两份交付使用相同的 `netrsn_turbo` 包路径，合入时选择一份，
替换后重启 Worker。

本轮内置 DS LoRA 的 10 个接口，面向 vLLM **0.30.x**、Ascend **A2/A3**。
调用链为：

```text
version_0300 DS LoRA patch
  → netrsn_turbo.turbo_ops（Python API）
  → torch.ops.turbo_ops（PrivateUse1 / Meta 注册）
  → _C*.so + libturbo_ops_lora_kernels.so
  → AscendC direct launch / CANN 基础 GMM
```

算子源码随 NetrsnTurbo 构建、打包，不需要单独安装 `ops_repo`，也不需要它的 Add OPP。
保留 vLLM / vLLM Ascend 本身的正常依赖；仅 DS LoRA 的原生调用切到 `turbo_ops`。
设备入口、C++ namespace、dispatcher namespace 和动态库名称均独立，可与原框架库共存。

## v2 相对上一份交付的变化

- 单 LoRA 槽位且存在活跃 adapter 时，`add_lora_linear` 优先走原生 `lora_linear`
  registry 分派，复用上游的 matmul / A3 Triton 路径。显式 buffer 和其他配置维持各自原路径。
- `NPUModelRunner._dummy_run` 使用调用方给定的 `num_active_loras`，移除强制替换为
  `max_loras` 的旧逻辑，使 LoRA dummy-run 与图捕获描述使用同一个数量。
- 量化 MoE 从原 `quant_moe` 模块动态读取 routing、w13、w2 helper，保持 v2 的可替换
  模块属性语义。这里采用运行时导入，避免 helper 依赖形成循环导入。
- C++ / AscendC 源码与上一版相同，仍为 10 个接口、7 个内核；已重新核对 v2 的来源哈希。

`_lora_linear_kernel`、`_get_shrink_buffer`、`_lora_linear_matmul` 在目标基线已有，
与 v2 的 AST 一致，直接复用。`token_dispatcher.py` 在两个 DS 分支间仅有排版差异。
已有 fully-sharded W2 all-reduce 和 LoRA slot 精度边界修复继续保留；DSpark 的量化映射
及 `load_weights` 的非 LoRA 记账差异仍不纳入本补丁。

## 目录与接口

```text
netrsn_turbo/
  turbo_ops/
    __init__.py / lora.py        十个公开接口
    _runtime.py                 延迟加载与 ABI 检查
    csrc/CMakeLists.txt         原生构建入口
    csrc/kernels/lora/          七个 AscendC 内核
    csrc/lora/                  Host、Meta、ACLNN adapter
    provenance.json / LICENSE  来源、规范化 SHA256 和许可
  turbo/version_0300/ds_lora/   DS LoRA Python 补丁及接口适配
    __init__.py / installer.py / manifest.py
    vllm_ascend/
      ascend_forward_context.py
      envs.py
      lora/
        fused_moe.py
        punica_npu.py
        quant_moe.py
        lora_ops.py / ops_backend.py
      ops/fused_moe/
        routed_experts.py
        token_dispatcher.py
      models/deepseek_v4/
        model.py
      worker/
        model_runner_v1.py
  turbo_manager/version_0300/  TurboManager 接入入口
build_support/turbo_ops.py
setup_turbo_ops.py             独立构建验证入口
tests/                        无硬件测试及可选 NPU 回归
```

`ds_lora` 下的补丁文件与上游 Python 模块路径对应，例如
`vllm_ascend/lora/punica_npu.py`、`vllm_ascend/models/deepseek_v4/model.py`。
`installer.py` / `manifest.py` 保留为公共安装入口和补丁清单；`lora/ops_backend.py`
是内部 `turbo_ops` 的连接层。当前补丁目标全部属于 `vllm_ascend`，因此没有空的
`vllm/` 目录。各级 `__init__.py` 不提前加载算子或模型，原 `apply_lora_patch()`
导入路径不变。

`apply_lora_patch()` 逐条调用 `TurboManager.register_patch()`，共 **20 个现有函数或
方法替换**，最后统一调用一次 `TurboManager.apply_patches()`。**13 个新增属性**
直接赋值；环境变量字典、MoE 缓存字段和 W8A8 冻结注册项分别更新。
`manifest.py` 仅作补丁范围核对，不再驱动安装循环。入口不安装 import hook，
也不维护已应用模块或初始化状态；版本分发、导入时序和防重复注册由 Netrsn 负责。
调用入口时会加载所需的原模块和补丁模块，应接入 Netrsn 已有的 LoRA 初始化阶段。

Worker 补丁在注册时对已加载方法做单点 AST 改写，只替换 `maybe_dummy_run_with_lora`
调用的 `num_active_loras` 实参；保留原模块 globals、方法签名和 `torch.inference_mode()`。
目标环境需保留 vLLM Ascend 的 Python 源文件；遇到不匹配的方法布局或包装层会明确报错。

公开接口：`bgmv_shrink`、`bgmv_expand`、`sgmv_shrink`、`sgmv_expand`、
`build_combined_lora_idx`、`add_lora`、`add_lora_shrink`、`add_lora_expand`、
`add_lora_expand_delta`、`add_lora_swiglu_quant`。
后六个对应 DS 新增接口，前四个是其所需的 BG/SG 基础算子。
参数顺序、返回值及原地写入行为与已迁入 `ops-repo` 的 DS 接口保持一致。

## 合入真实 NetrsnTurbo

1. 合并本目录的 `netrsn_turbo/` 子树和 `build_support/` 到主仓，保留主仓现有的
   父级 `__init__.py`、构建配置及其他版本补丁。
2. 在主仓 `setup.py` 中，用构建 helper 包装现有参数：

   ```python
   from setuptools import setup
   from build_support import configure_turbo_ops

   # 保留主仓完整的 packages 列表及其余 setup 参数。
   setup(**configure_turbo_ops(existing_setup_kwargs))
   ```

   helper 合并原有扩展与构建命令。实际分发使用主仓的 wheel 名称和版本。
   下方独立入口用于先验证算子包，正式合入后不需要额外的算子 wheel。
   原生构建要求参数中显式提供主仓完整的 `packages` 列表；若主仓依赖自动发现，
   应先沿用主仓的发现规则生成该列表，避免 setuptools 因新增扩展而漏打包主仓代码。
   当前 helper 对应本文的仓库根目录布局：`netrsn_turbo/turbo_ops/`；采用 `src/`
   布局的主仓需同步调整 helper 的源码定位。
3. 在 `netrsn_turbo/turbo_manager/__init__.py` 的旧 `>=0.23` 分支之前添加：

   ```python
   from packaging.version import Version
   import vllm

   if Version("0.30.0") <= Version(vllm.__version__) < Version("0.31.0"):
       from .version_0300 import turbo_ds_lora
       # 保留主仓在 0.30 下仍需执行的其他补丁入口。
   elif ...:
       ...
   ```

   按根目录 `patch_mesim.md`，保持 `vllm → vllm_ascend → turbo_manager` 的加载顺序。
   入口以非空 `ADAPTATION_PKG_ID` 启用 LoRA；若部署使用
   `GlobalContext.get_is_lora()`，在主仓已有 LoRA 分支显式调用
   `from netrsn_turbo.turbo.version_0300.ds_lora import apply_lora_patch` 后执行
   `apply_lora_patch()`，时机应在构建引擎之前。
4. `special_init.py` 的 spawn Worker 仍需按原机制重新导入
   `netrsn.core.turbo`，从而在每个 Worker 安装补丁。

导入 `turbo_ops` 不导入 torch / torch_npu，也不加载原生库。第一个 Punica wrapper
构造前会调用 `turbo_ops.initialize()`，校验 Python 小版本、架构、torch 和 torch_npu
版本、NPU 硬件族及十个注册接口；后续调用复用结果。加载失败会报出构建信息，替换 wheel 后须重启 Worker。
自行调用算子时，先选择设备，再 `turbo_ops.initialize()`，之后再开始图捕获或编译。

## 构建

在目标 **Linux** 设备的运行环境构建，需已有匹配的 torch、torch_npu、CANN AscendC
开发工具链、CMake ≥ 3.19、C++17 编译器、setuptools、wheel。先加载 CANN 的环境脚本。
不支持 A5/310P；A2 和 A3 应分别构建，`SOC_VERSION` 按实际芯片选择。

独立验证入口（在本目录执行）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# A2 示例；实际芯片为 910B2 时改为 ascend910b2。
TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910b \
  TURBO_OPS_SOC_VERSION=ascend910b1 MAX_JOBS=4 \
  python setup_turbo_ops.py bdist_wheel --dist-dir dist/ascend910b

# A3 示例。
TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910_93 \
  TURBO_OPS_SOC_VERSION=ascend910_9391 MAX_JOBS=4 \
  python setup_turbo_ops.py bdist_wheel --dist-dir dist/ascend910_93
```

在真实主仓合并 helper 后，用相同环境变量运行主仓原有 wheel 构建命令，例如：

```bash
TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910_93 \
  TURBO_OPS_SOC_VERSION=ascend910_9391 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist/ascend910_93
```

未设置 `TURBO_OPS_BUILD=1` 时不编译内核，这种包不能运行本补丁的 DS LoRA。
原生构建使用普通 wheel 安装，当前不支持 editable 或 `build_ext --inplace`。
运行时共享库与 `_build_info.json` 位于 `netrsn_turbo/turbo_ops/` 内，kernel 库经
`$ORIGIN` 查找；CANN / torch_npu 自身的动态库仍依赖目标环境正常配置。

## 验证及范围

无需 torch / NPU 的加载、构建、源码一致性及补丁测试：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

安装编译后的 wheel，在目标设备运行（额外需要 pytest）。从交付目录之外执行，
避免当前源码目录遮蔽 wheel 内的原生库：

```bash
cd /tmp
TURBO_OPS_RUN_NPU_TESTS=1 TURBO_OPS_TEST_DEVICE=npu:0 \
  python -m pytest /path/to/netrsnturbo_patch_v2/tests/npu --import-mode=importlib -q
```

硬件用例包含数值参考、窄 rank 写回、LoRA slot 更新、图回放及 fully-sharded / EP
局部算子路径；这些测试不能替代真实多卡 vLLM 0.30 模型验证。
本次开发环境为 Windows，没有 torch / CANN / NPU，因此未执行原生编译或 NPU 测试。
无硬件测试覆盖 v2 单槽位分派、Worker 活跃 LoRA 数量、量化 helper 动态替换、
目录映射、显式注册、包入口懒加载，以及 setuptools wheel / sdist 内容和 native → pure 重建残留。
本地完整运行 **91 项测试，全部通过**，包括 Worker 生成方法与真实 v2 方法的执行码比对。
打包测试中的原生文件为合成样本，仅验证分发机制。

源码固定于 DS v2 `v30_lora_v2` 提交 `5dd159d7d2acf70c0bfcec1a2055ac52c60a13bf`，
目标基线为 `vllm-ascend` 的 `5f828ad638ee4f9b12db87f9d5481f29076d37f9`，
vLLM 为本地 0.30.x。原生代码经过与 v2 的逐文件比对，
并保留 `ops-repo` 中 BGMV/SGMV mutation/alias 与 Meta 适配。来源详情见
`netrsn_turbo/turbo_ops/provenance.json`。
Python 补丁保留 fully-sharded W2 shrink 的必要 all-reduce，以及 LoRA slot packing
的精度边界处理；本轮融合未更改计算算法。
旧 `ops-repo/integrations/netrsnturbo/version_0300-ops-repo.patch` 是原 Python-only
交付的外部 wheel 方案，不要叠加到当前内置版本。
