# 完整模型资源准备

核对日期：2026-10-03。本指南面向当前发行的 `optimized` 完整融合与 `legacy` 真 JEPA 路径，不是所有历史研究分支的训练环境说明。

## 1. 先确认发行与许可范围

仓库只提供项目所有者明确授权的两个去私人元信息的下游定位模型发布副本：`models/public/locator.json` 和 `models/public/motion.json`，及其 registry、许可和公开合成样例生成器。这是特定发布例外，不授权上传原始模型 JSON、第三方 encoder/checkpoint、真实视频、标注、缓存或私有报告。发布副本保留学习参数、特征顺序和解码规则；合成样例用于体验，不是独立评测。

项目自有代码和获授权 readout 使用 MIT；**I-JEPA 固定版本源码是 CC BY-NC 4.0，完整推理链路不能据此宣称统一 MIT 或可商用**。先阅读 [第三方说明](../THIRD_PARTY_NOTICES.md) 与各上游许可、权重条款，再决定是否获取资源。`--accept-noncommercial` 只记录对 I-JEPA 源码非商业限制的明确确认，不替代许可审查，也不授予商业权利。

基础 `optimized_fast` 学习式 CPU 模式不需要下述大模型资源；它不是 JEPA，也不是规则回退。见 [运行指南](RUNNING.md)。前端启动、模型导入、model doctor 和 Worker 启动**都不会自动下载资源**。

## 2. 固定资源与完整性声明

声明以 [发布 registry](../models/public/registry.json)、[模型说明](../models/README.md) 和发布模型的 feature profile 为准；不要跟随上游 main/latest，也不要编辑 metadata、profile 或 registry 来绕过失败。

### 官方源码

| 资源 | 官方固定提交 | 推理源码树 SHA-256 |
| --- | --- | --- |
| V-JEPA 2.1 | [facebookresearch/vjepa2 @ 204698b45b3712590f06245fbfba32d3be539812](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812) | `0397155c0fda831ba92ebf0e838d9367a29f53daab355ac87dc5fdadf71f3971` |
| I-JEPA | [facebookresearch/ijepa @ 52c1ae95d05f743e000e8f10a1f3a79b10cff048](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048) | `aa751b5762f09555a606060106ad341929fa3a3f6146a2b174d560ca369d8d1a` |

这里的 source-tree SHA 是工具对 `src/`、`app/` 中所有 Python 文件的相对路径与文件 SHA 按固定次序计算的摘要，**不是 Git tree object SHA 或 ZIP SHA**。

资源工具会读取并保存官方固定提交 ZIP，但只展开 `src/`、`app/` 与根层 LICENSE/NOTICE/COPYING、README、requirements、pyproject.toml、setup.py 等许可和说明。不展开训练 configs 等其余目录，不克隆/安装整个训练仓库，也不执行训练。此裁剪不改变 registry 固定的推理源码树摘要，且工具要求保留上游 `LICENSE`。自行取得源码时也必须保持相同的文件字节、相对路径及许可。

### 官方 checkpoint 与本地转换

**以下官方 checkpoint 链接指向权重文件，点击可能直接开始大下载；它们不是本仓库分发的文件。**

| 文件 / 输入 | 官方路线或本地生成规则 | 固定 SHA-256 |
| --- | --- | --- |
| V-JEPA full / predictor，`vjepa2_1_vitg_384.pt` | 固定提交 [V-JEPA README](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md) 中 ViT-g 384；[官方下载（大文件）](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt) | `b417628f1618c8bd52c0f419800b802f65794288c6ef2ff85c1341a1ae587cba` |
| V-JEPA encoder，`encoder_only.pt` | 从上述 full 的 `target_encoder` 分支生成 | `897b3578e3d9a9df1b72aecc1f79026de61cb03ce28cb0ea56f56fa30e3a2f94` |
| I-JEPA full，`IN22K-vit.g.16-600e.pth.tar` | 固定提交 [I-JEPA README](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/README.md) 的 ViT-g/16（224px）、ImageNet-22K 表项；[官方下载（大文件）](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar) | `a373c7b16165092e29bd5714c6ce79c574c4a52b01b64bf4c82dc81e6a82cf0b` |
| I-JEPA BF16，`ijepa_true_slim_bf16.pt` | 从上述 full 保留三分支后生成 | `8e3e9a1fd5aece75357e87644e55e2d345415ad1170ce55ab24f2cd33361c050` |
| RGB R3D-18，`r3d_18-b3b3357e.pth` | torchvision 0.27.0 的 `R3D_18_Weights.KINETICS400_V1`；[官方定义](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/video/resnet.py)、[官方下载](https://download.pytorch.org/models/r3d_18-b3b3357e.pth) | `b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d` |

I-JEPA 固定 README 的该表项标为 44 epochs，但链接文件名含 600e；本指南按官方链接与 registry SHA 固定资源，不依据文件名推断训练轮次，也不另选“相近”模型。

固定转换规则：

- V：取 `target_encoder`，清理参数键中的 `module.` 与 `backbone.`，保持原 FP32 张量，保存为只含 `encoder` 的字典。
- I：保留 `encoder`、`target_encoder`、`predictor`，清理键中的 `module.`，全部转为 BF16。
- 使用 `torch==2.12.0+cu132` 和工具固定的输出文件名；Torch 序列化版本、文件名和分支选择都影响文件级一致性，不要自行换序列化格式。

registry 记录的既有转换验证中，转换输出逐张量及文件 SHA 均与本地基线一致。这是资源重建证据，**不是本指南完成了 GPU 推理验收，也不是复现了私有训练数据或上文实验成绩**。

## 3. 选择 Windows 前端 + WSL，或原生 Linux

以下命令均从项目根目录执行；示例 checkout 路径请换成自己的绝对路径。需要可用的 NVIDIA CUDA GPU/驱动；Windows 使用 WSL 2，参照 [Microsoft WSL 安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [NVIDIA CUDA on WSL 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 配置。不要把 CPU 构建或无 CUDA 环境当作完整模式等价环境。

### A. Windows 前端 + WSL 模型环境

PowerShell 准备前端依赖，并查看实际发行版名称（此时不启动服务）：

```powershell
Set-Location "C:/src/jepa-system"
py -3.12 -m venv .venv-front
./.venv-front/Scripts/python.exe -m pip install -r requirements-demo.txt
wsl.exe --list --verbose
```

Windows 前端可以使用 Python 3.12.x；**完整模型解释器必须是下一节的 3.12.3**。进入选定 WSL 发行版，在 Bash 中设置：

```bash
cd /mnt/c/src/jepa-system
TRANSPORT=wsl
RESOURCE_DIR="$PWD/resources"
```

使用 Windows 与 WSL 共享的同一个 checkout 和同一份 TOML。模型 venv 放在 WSL Linux 文件系统（如下一节的 `$HOME/.venvs/`），不要拿 Windows venv 给 Linux 使用。项目在其他盘时，相应使用 `/mnt/<盘符>/...`。

### B. 原生 Linux

在 Bash 中设置：

```bash
cd "$HOME/src/jepa-system"
TRANSPORT=native
RESOURCE_DIR="$PWD/resources"
```

前端和模型可使用同一个 Linux venv；`requirements-models.txt` 已包含 `requirements-demo.txt`。以下第 4、5 节对两种方式相同，均在 Linux/WSL Bash 内执行。

## 4. 安装严格匹配的完整 RGB profile

| 模型侧运行版本 | 必须匹配 |
| --- | --- |
| Python | `3.12.3`（不是任意 3.12.x） |
| Torch | `2.12.0+cu132` |
| torchvision | `0.27.0+cu132` |
| NumPy | `2.4.4` |
| OpenCV 的 `cv2.__version__` | `4.13.0`（pip 包 `opencv-python==4.13.0.92`） |

固定版本服务于现有 feature profile，不是对旧 Python 的通用安全/生产部署建议。Python 的官方来源为 [Python 3.12.3 发布页](https://www.python.org/downloads/release/python-3123/)；安装方法见 [官方 Unix 安装说明](https://docs.python.org/3.12/using/unix.html)。如 Linux 已有恰好 3.12.3，可直接令 `PY3123="$(command -v python3.12)"`。

否则，可先按 [CPython 官方构建依赖说明](https://devguide.python.org/getting-started/setup-building/#linux) 准备编译器与开发库，再在用户目录单独安装；不要替换系统 Python，也不要覆盖已有、仍被其他任务使用的同名安装：

```bash
PROJECT_DIR="$PWD"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/jepa-python-3.12.3.XXXXXX")"
cd "$BUILD_DIR"
curl -fLO https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
tar -xzf Python-3.12.3.tgz
cd Python-3.12.3
./configure --prefix="$HOME/.local/cpython-3.12.3" --with-ensurepip=install
make -j2
make altinstall
PY3123="$HOME/.local/cpython-3.12.3/bin/python3.12"
cd "$PROJECT_DIR"
```

上述构建最后回到原 checkout。已有 3.12.3 时可跳过构建；下面安装始终在项目根目录执行：

```bash
# 保留第 3 节的 TRANSPORT、RESOURCE_DIR；PY3123 指向已核对的 3.12.3。
"$PY3123" -c "import sys; assert sys.version.split()[0] == '3.12.3', sys.version"
MODEL_PY="$HOME/.venvs/jepa-model-py3123/bin/python"
"$PY3123" -m venv "$HOME/.venvs/jepa-model-py3123"
"$MODEL_PY" -m pip install "torch==2.12.0+cu132" "torchvision==0.27.0+cu132" --index-url https://download.pytorch.org/whl/cu132
"$MODEL_PY" -m pip install -r requirements-models.txt
```

官方 [cu132 Torch 索引](https://download.pytorch.org/whl/cu132/torch/) 与 [torchvision 索引](https://download.pytorch.org/whl/cu132/torchvision/) 包含上述版本的 CPython 3.12 Linux x86_64 wheel。Torch/CUDA pip 安装本身也会下载较大软件包，这与下一节的模型权重下载开关是两件事。

除 Torch/torchvision 外，沿用既有 `requirements-models.txt`：Pillow 12.2.0、matplotlib 3.10.9、PyYAML 6.0.3、tqdm 4.67.3、psutil 7.2.2，及 demo 的 Flask 3.1.3、NumPy 2.4.4、opencv-python 4.13.0.92。不需要 `pip install -e` 上游仓库，也不以其训练 requirements 替换本项目依赖。

## 5. 显式准备资源（二选一）

先审阅许可。如果只想准备源码、暂不取得权重，可执行：

```bash
"$MODEL_PY" -B code/prepare_model_resources.py --directory "$RESOURCE_DIR" --sources-only --accept-noncommercial
```

这仍会取得固定源码归档，但**不下载 checkpoint、不转换模型、不写完整模式配置**。`--sources-only` 不能与 `--download-checkpoints` 或 `--write-config` 合用。

### 方式 1：使用自己取得的官方 full checkpoint

按第 2 节官方路线自行取得三份输入：V full、I full、R3D。必须都是固定 SHA 的完整文件，不能拿 encoder-only 代替 V full，或拿已转换 BF16 代替 I full；R3D 也必须显式提供。然后执行（将输入路径换成自己的 Linux 路径）：

```bash
"$MODEL_PY" -B code/prepare_model_resources.py \
  --directory "$RESOURCE_DIR" \
  --vjepa-full "$HOME/checkpoints/vjepa2_1_vitg_384.pt" \
  --ijepa-full "$HOME/checkpoints/IN22K-vit.g.16-600e.pth.tar" \
  --r3d "$HOME/checkpoints/r3d_18-b3b3357e.pth" \
  --accept-noncommercial --transport "$TRANSPORT" \
  --write-config "$PWD/jepa-runtime.toml"
```

该方式不请求工具下载权重，但若固定源码尚未准备，仍会下载源码 ZIP。输入文件先验 SHA 校验通过后才转换；原输入保留，输出写在 `$RESOURCE_DIR/checkpoints/`。

### 方式 2：明确允许工具下载大权重

**执行下一条即主动允许大额网络传输。** V full 单文件为 16,878,318,788 字节（约 16.88 GB），I full 也是大模型 checkpoint；工具还下载 R3D（133,546,016 字节），并生成约 4.05 GB 的 V encoder 与约 4.10 GB 的 I BF16 文件。需为原输入、转换副本、源码归档、Python 环境预留充足磁盘、内存、时间和流量。未对 I full 的大小或总下载量作未核实保证。

确认许可、空间和流量后再运行；**两个显式标志都不能省略**：

```bash
"$MODEL_PY" -B code/prepare_model_resources.py \
  --directory "$RESOURCE_DIR" \
  --accept-noncommercial --download-checkpoints \
  --transport "$TRANSPORT" --write-config "$PWD/jepa-runtime.toml"
```

工具不会覆盖已有配置；使用 `--write-config` 前不要先复制示例 TOML。如果已配置，去掉此参数并核对现有路径，或写入另一个尚不存在的本地文件，再用 `JEPA_CONFIG` 指向它。已有资源只有匹配 SHA/源码摘要、包含 LICENSE 且不存在未完成准备标记才复用；不匹配或中断残留不覆盖，失败文件可能留存用于诊断。转换还需要同一资源磁盘容纳暂存与目标副本，请预留足够空间；发生 I/O/中断后使用新的输出目录重新准备，不要更改 SHA 或让半成品冒充成功。不要改 registry、关闭校验或把来源不明的下载重命名后当作通过。

## 6. 配置、核查与启动

准备工具成功时只表示源码、输入与转换资源通过其完整性检查；它不检查驱动就绪或执行 GPU 推理。按本指南在 Linux/WSL 内运行时，新 TOML 记录实际模型解释器和资源路径；若从 Windows 宿主指定 WSL transport，工具只填安全的 `python3` 默认值，必须再改为已经核对的 WSL 模型 venv 路径。新 TOML 设置 `default_algorithm = "optimized"`。

- 原生 Linux：`runtime.transport = "native"`，`runtime.python` 为上述 Linux venv 解释器。
- Windows + WSL：`runtime.transport = "wsl"`，`runtime.wsl_python` 为 WSL venv 的绝对路径。工具未写 `wsl_distro` 时默认 `Ubuntu-24.04`；如实际名称不同，在已有 `[runtime]` 下补 `wsl_distro = "实际发行版名"`，不要另建重复 table。
- `[resources]` 路径归模型环境解释；WSL 使用 Linux 绝对路径。不要把 WSL 根目录 `/home/...` 当成 Windows 普通目录。相对资源路径基于项目 checkout，而非当前终端目录。
- 已有原始本地模型包时，兼容默认可能优先原始包。为了明确核查本次发行，在已有 `[demo]` 中补如下两项（不要重复 table）：

```toml
full_bundle = "models/public/locator.json"
motion_bundle = "models/public/motion.json"
```

配置可参考 [示例 TOML](../jepa-runtime.example.toml)。不要将本地真实配置、资源或原模型提交到 Git。

### Windows 前端核查与启动

在同一 checkout 的 PowerShell 中执行，`JEPA_CONFIG` 指向共享 TOML 的 Windows 路径；启动前确认自己需要使用的 localhost 端口没有被其他服务占用，不停止他人的服务：

```powershell
$env:JEPA_CONFIG = (Join-Path (Get-Location).Path "jepa-runtime.toml")
./.venv-front/Scripts/python.exe -B code/model_doctor.py
./.venv-front/Scripts/python.exe -B code/model_doctor.py --model-runtime
if ($LASTEXITCODE -ne 0) { throw "模型资源检查失败；不启动完整模式" }
./.venv-front/Scripts/python.exe -B code/demo_app.py
```

### 原生 Linux 核查与启动

```bash
export JEPA_CONFIG="$PWD/jepa-runtime.toml"
"$MODEL_PY" -B code/model_doctor.py
"$MODEL_PY" -B code/model_doctor.py --model-runtime &&
  "$MODEL_PY" -B code/demo_app.py
```

默认 doctor 区分可读取的研究包与经过严格发行校验的发布包；`--model-runtime` 才使用配置的 native/WSL 解释器检查上游源码树、四份运行 checkpoint SHA、完整 RGB runtime versions 与 `torch.cuda.is_available()`。它不加载权重做推理、不读取用户视频、不下载文件、不启动常驻 Worker。即使 doctor 成功，仍需实际新视频推理才能声称 GPU 路径通过。

前端地址是 `http://127.0.0.1:5002/`。公开样例须点击“用当前模式分析样例”才真实执行；样例本身不是预置推理结果或质量证明。完整 CLI、输入路径边界、可选 localhost 5004 热 Worker、`retain_models=false` 低显存取舍与故障排查见 [运行指南](RUNNING.md)。不要因资源错误把 CPU fallback 当成优化/GPU 验收成功。

## 7. 核查与许可依据

本指南使用固定提交的官方 README/许可、Python 官方发布页、PyTorch 官方 CUDA wheel 索引及 torchvision 官方模型定义。另见：

- [V-JEPA 固定提交 LICENSE（MIT）](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE) 与源码中的 [Apache-2.0 来源声明](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/utils/randaugment.py)。
- [I-JEPA 固定提交 LICENSE（CC BY-NC 4.0）](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE)。
- [torchvision 0.27.0 LICENSE（BSD-3-Clause 源码许可）](https://github.com/pytorch/vision/blob/v0.27.0/LICENSE)；官方 [预训练模型使用说明](https://docs.pytorch.org/vision/0.27/models.html) 要求另行核对预训练模型及其数据相关条款。

本轮文档核对不下载大权重、不另行运行 GPU 验收，也不新增生产依赖。许可说明不作全链商业授权保证；发行运行证据与冻结实验记录应分别报告。

## 8. 当前验收边界

完整融合与历史真 JEPA 已在隔离源码目录完成真实推理；新 Windows venv 已完成学习式 CPU 与工作台流程检查。GPU 检查复用明确声明的固定 WSL 模型环境与经 SHA 核验的官方资源，没有复用私人特征缓存。完整证据边界见 [发行验收说明](REPRODUCIBLE_RELEASE.md)，不宣称从零重装整个 CUDA/Python 环境、其他硬件或原生 Linux 独立部署均通过。

新增可配置 loader 只将 V checkpoint 暂存改为 CPU mmap，再严格装入既有模型，避免重复 CUDA FP32 分配导致 OOM；原 `vjepa_predictor.py` 和 `optimized_jepa_extractor.py` 未因本次适配修改。该加载调整不授权更改权重、特征或来源校验。详细发行范围见 [项目状态](PROJECT_STATUS.md) 的发行说明，不更改其中具体实验记录。
