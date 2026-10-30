# 部署与运维指南

本文命令均针对 Ubuntu 22.04 x86_64 GPU 服务器，不代表已在本地执行。Conda 环境名由 `AIGC_CONDA_ENV` 配置（当前示例为 `photomaker`，默认值为 `aigc-photomaker`），GPU 固定为物理 GPU 1：`CUDA_VISIBLE_DEVICES=1`。启动脚本不得再传 `--cuda-device`，因为 ComfyUI 0.42.1 会用该参数覆盖 `CUDA_VISIBLE_DEVICES`；保留环境映射后，进程内仍使用逻辑 `cuda:0`。

## 1. 前置条件与访问控制

- 一张 RTX 3090 24GB，NVIDIA 驱动支持锁定的 PyTorch 2.9.1+cu126；安装/启动脚本按 `总显存 - 已用显存` 检查 GPU 1，要求可用显存至少为 `AIGC_GPU_MIN_FREE_MIB`（默认 20480 MiB）。存在其他计算进程时仅警告，不直接失败；旧的 `AIGC_GPU_MAX_USED_MIB=1024` 已删除，遗留配置不再生效。
- 已安装 Conda、Python 3.11（制作离线包时）、Git、curl、tar、sha256sum、unzip、gcc/g++；缺少编译器时安装 `build-essential`。
- 准备足够磁盘存放应用、Conda 环境、wheel、ComfyUI 和模型。大模型不包含在应用 ZIP 中。
- 从 ZIP 的父目录解压：`unzip aigc-photomaker.zip && cd aigc-photomaker`。路径为单层 `aigc-photomaker`。
- 安装器在缺少 Key 时向 `.env` 写入随机 API Key，并将权限设为 600。不要提交、打印或截图该文件。新 shell 可执行 `set -a; source .env; set +a` 供冒烟和基准脚本读取。
- 默认 API 为 `0.0.0.0:8000`，ComfyUI 为 `127.0.0.1:8188`；可通过 `.env` 的 `AIGC_APP_PORT` 和 `AIGC_COMFY_PORT` 改为空闲端口，浏览器使用实际 API 端口。生产访问应限制受控网段并配置 TLS，防火墙仅向需要的来源放行实际 API 端口，例如 `sudo ufw allow from 受控网段 to any port 8186 proto tcp`；不要公开 ComfyUI，也不要把示例命令当作已配置防火墙的证据。

## 2. 镜像在线安装

```bash
export HF_ENDPOINT=https://hf-mirror.com
export GITHUB_PROXY=https://ghproxy.net/
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/install.sh
bash scripts/download_models.sh
bash scripts/start.sh
bash scripts/status.sh
```

可选 PyTorch 索引由 `PYTORCH_INDEX_URL` 控制，默认 `https://download.pytorch.org/whl/cu126`；额外 pip 索引由 `PIP_EXTRA_INDEX_URL` 控制。不要为绕过校验随意更换模型 revision 或 SHA256。

**配置优先级**：不可变仓库/commit/ Torch 版本来自 `config/versions.lock.env`；运行配置中，调用者显式导出的 `AIGC_*`、`HF_ENDPOINT`、`PIP_INDEX_URL`、`PIP_EXTRA_INDEX_URL`、`PYTORCH_INDEX_URL`、`GITHUB_PROXY` 覆盖 `.env`，`.env` 再提供默认值。也就是说 `.env` 不能覆盖锁定 commit，也不能悄悄覆盖当前 shell 的显式选择。

安装器固定 ComfyUI commit，复制随包交付的 InstantID/FLUX 人脸守卫与历史 PhotoMaker Plus 节点源码，创建/复用环境并安装 wheel，校验和导入模型，并把锁定的 antelopev2 归档安全解压到 `models/insightface/models/antelopev2/`。`feature/flux` 默认需要 FLUX.2 Distilled FP8、Qwen3 文本编码器、FLUX.2 VAE 和 antelopev2，新增下载约 12.8GB；Base FP8 仅作可选回滚，旧 SDXL、InstantID 和风格 LoRA 仅作为可选回滚项，`AIGC_INCLUDE_OPTIONAL_MODELS=1` 才会包含全部可选模型。模型目录为 `.runtime/ComfyUI/models/`，离线模式不会隐式联网补模型。

### 2.1 在线手工安装 Python 依赖

需要手工控制依赖安装时，在项目根目录执行：

```bash
conda create -n photomaker -y python=3.11 pip
conda activate photomaker
python -m pip install --upgrade pip setuptools wheel numpy==1.26.4 Cython==3.0.12 build
PIP_NO_BUILD_ISOLATION=1 python -m pip install -r requirements.txt
AIGC_SKIP_PYTHON_INSTALL=1 bash scripts/install.sh
```

新增/更新风格 LoRA 或 `vendor/` 自定义节点后，也要重新执行安装命令：它负责下载并校验新增模型、把更新后的节点复制到运行目录，然后再执行 `bash scripts/stop.sh && bash scripts/start.sh`。只上传源码而不重新安装，运行中的 ComfyUI 不会自动获得新节点。

该流程只替换第三方 Python 依赖安装，不替代 ComfyUI 源码与节点安装、模型下载、API Key 创建、GPU/依赖 preflight 或后续启动；项目自身始终以 editable 方式执行 `pip install -e . --no-deps`，保证 `python -m aigc_photomaker...` 可用。仍须执行 `bash scripts/start.sh`。`AIGC_SKIP_PYTHON_INSTALL=1` 与 `AIGC_OFFLINE=1` 不兼容，避免离线前缀解包清除手工安装的包。

## 3. 完全离线发布与安装

### 3.1 在联网构建机制作大包

构建机也必须是 Ubuntu 22.04 x86_64，具备 Python 3.11、Conda 和网络。这里使用镜像加速构建，但最终离线包须整体传输并校验：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export GITHUB_PROXY=https://ghproxy.net/
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export AIGC_OFFLINE=0
bash scripts/prepare_offline_bundle.sh
```

脚本生成 `offline/comfyui/ComfyUI.tar.gz`、`offline/conda/aigc-photomaker.tar.gz`（conda-pack）、`offline/wheelhouse/`、`offline/models/`、`offline/SOURCE_COMMITS` 和覆盖这些文件的 `offline/SHA256SUMS`。该目录是大包，模型和 PyTorch wheel 不塞进应用 ZIP。

### 3.2 分开上传应用与大包

在干净目标目录先解压应用，再上传完整 `offline/` 到项目根目录，最终必须是 `<项目>/offline/SHA256SUMS`，不要多套一层目录。大文件可用 scp/rsync/受控对象存储分卷，传输方式不限，但不能省略完整性检查：

```bash
unzip aigc-photomaker.zip
cd aigc-photomaker
# 在此处通过受控渠道上传 offline/，然后：
test -f offline/conda/aigc-photomaker.tar.gz
test -f offline/models/diffusion_models/flux-2-klein-base-4b-fp8.safetensors
test -f offline/models/diffusion_models/flux-2-klein-4b-fp8.safetensors
test -f offline/models/text_encoders/qwen_3_4b.safetensors
test -f offline/models/vae/flux2-vae.safetensors
test -f offline/models/instantid/antelopev2.zip
(cd offline && sha256sum -c SHA256SUMS)
```

### 3.3 网络禁用验证与启动

应在全新虚拟机/系统用户或干净重装目录中，由管理员禁用网卡/隔离网络后再执行；仅设置环境变量不等于网络禁用证据。记录隔离方式、时间和机器信息，且不要运行在线安装步骤：

```bash
export AIGC_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
bash scripts/install.sh
bash scripts/download_models.sh
bash scripts/start.sh
bash scripts/status.sh
set -a; source .env; set +a
bash scripts/smoke_test.sh
```

`install.sh` 先验证全部 SHA256，再从 `offline/conda/aigc-photomaker.tar.gz` 解到 Conda 的 `envs/${AIGC_CONDA_ENV:-aigc-photomaker}` 前缀并执行该前缀的 `conda-unpack`，之后 pip 统一使用 `--no-index --find-links=offline/wheelhouse`。不要改成 `conda create/install/run` 来完成离线解包。保存断网证明、hash 输出、安装/启动日志、状态及 `reports/` 冒烟报告；任何失败都如实记为失败。

## 4. 启动、状态、日志、停止和重启

```bash
bash scripts/start.sh                 # 校验模型/GPU，启动 ComfyUI，预热两种模式，再启动 Worker/API
bash scripts/status.sh                # api、worker、comfy；端口；物理 GPU 1；磁盘；readiness
bash scripts/stop.sh                  # 依次停止 API、Worker、ComfyUI
bash scripts/stop.sh && bash scripts/start.sh
```

如果默认端口已有其他服务，**不要停止或替换该服务**。在 `.env` 中修改已有配置项（不要重复添加），例如本次服务器使用：

```bash
AIGC_COMFY_PORT=8185
AIGC_APP_PORT=8186
```

`start.sh`、`status.sh`、预热、Worker、`smoke_test.sh` 和 `benchmark.sh` 都读取这组配置。启动前的绑定检查只负责防止误连；若配置端口被占用会明确失败，不会 kill 占用进程。启动成功日志会打印实际 ComfyUI、API 和浏览器地址。

PID 位于 `.runtime/app/pids/{api,worker,comfy}.pid`。日志位于：

- `.runtime/app/logs/api.log`
- `.runtime/app/logs/worker.log`
- `.runtime/app/logs/comfy.log`

查看示例：`tail -n 200 .runtime/app/logs/comfy.log`。单个日志达到 50MiB 时启动流程会轮转，最多保留 5 个编号旧文件。readiness 返回 503 时 `status.sh` 失败并保留响应体；不能只看进程存在就判定就绪。

## 5. systemd（用户服务）

仓库单元 `deploy/aigc-photomaker.service` 假定项目位于执行用户的 `~/aigc-photomaker`。路径不同则先复制单元并仅调整实际安装路径，审查后再启用：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/aigc-photomaker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start aigc-photomaker.service
systemctl --user status aigc-photomaker.service
systemctl --user restart aigc-photomaker.service
systemctl --user stop aigc-photomaker.service
journalctl --user -u aigc-photomaker.service -n 200 --no-pager
```

`Type=oneshot` 配合 `RemainAfterExit=yes`；启动脚本本身管理三个长期进程。需要登录会话外运行时，由管理员评估 `loginctl enable-linger 用户名` 的机器级影响，不要默认视为已启用。开机自动启动用 `systemctl --user enable aigc-photomaker.service`，并在重启后重新验收 GPU 占用和预热。

## 6. 性能验收口径与报告

先预热、确认无排队且仅一个活跃生成任务，再运行：

```bash
bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000 --style cyberpunk
bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000 --style cyberpunk
bash scripts/benchmark.sh --alternate-modes --runs 10 --base-seed 13000
```

可用 `--input`、`--report-dir`、`--api-url` 改变输入、报告目录和 API 地址。默认 JSON/Markdown 报告写入 `reports/`，文件名带 UTC 时间戳。Standard 门槛是至少 20 个成功样本、零失败、无 OOM、GPU 1 显存证据且 P95 ≤ 30 秒；Turbo 是至少 10 个成功样本、零失败、无 OOM 和真实显存证据，不强制 30 秒门槛。交替模式只证明交替执行，不替代 Standard 单模式测试。

## 7. 故障排查

| 现象 | 检查与处理 |
| --- | --- |
| 端口占用 | 不停止已有服务。用 `ss -ltnp` 查找空闲端口，在 `.env` 设置 `AIGC_COMFY_PORT` 与 `AIGC_APP_PORT`（本次可用 8185/8186）。`start.sh` 会检查所选地址，若仍被占用则明确失败，不会误连旧服务或 kill 对方。 |
| 判断风格 LoRA 是否实际执行 | 在项目根目录运行 `python scripts/inspect_latest_task.py`，查看最新任务的节点 6/9/13/14 和 `WORKFLOW_CHECKS`；也可用 `--task-id 任务ID`。脚本不打印 API Key。 |
| 30 秒超标 | 先确认已预热、无排队、单并发，再看报告各阶段和三个日志；排查其他 GPU 进程、慢磁盘、下载/上传网络、CPU 人脸检测和模式切换。不得用冷启动样本或交替报告冒充达标。 |
| GPU 显存不足或 OOM | 启动前检查会要求 `总显存 - 已用显存` 至少达到 `AIGC_GPU_MIN_FREE_MIB`（默认 20480 MiB），其他计算进程只触发警告。停止其他显存占用后重试，并确认 batch 1、1024×1024、只一个 Worker/ComfyUI；保留 OOM 日志。旧 `AIGC_GPU_MAX_USED_MIB=1024` 限制已移除，不能用该遗留变量绕过可用显存检查。不要降低分辨率、关闭身份模块或无限重试。 |
| `onnxruntime-gpu==1.20.1` 找不到 | 官方 PyPI 未发布该版本（返回 404），不是 Conda 或 Python 环境问题。改用官方存在且提供 CPython 3.11 wheel 的 `onnxruntime-gpu==1.20.2`，同步更新 `requirements.txt` 与 `requirements/ml.txt` 后重试。 |
| 缺脸/多脸 | 换清晰、正脸、单人照片；`NO_FACE_DETECTED`/`MULTIPLE_FACES_DETECTED` 是输入质量问题，不以随机图绕过 InstantID/antelopev2。输出对应错误为 `OUTPUT_NO_FACE_DETECTED` 和 `OUTPUT_MULTIPLE_FACES_DETECTED`。 |
| 模型缺失或 SHA 不匹配 | 运行 `bash scripts/download_models.sh`（离线时先验证大包），核对 `config/models.lock.json` 的目录、revision、大小和 hash；不要删除校验或混用 V1/V2。 |
| 节点/模型不匹配 | 核对固定 ComfyUI commit、完整自定义节点目录、FLUX.2 Distilled FP8、Qwen3 文本编码器、FLUX.2 VAE 与 antelopev2 解压目录；Standard/Turbo 都必须保留 ReferenceLatent 和输入/输出守卫。 |
| 启动中断或半启动 | 查看日志与 PID。启动脚本对失败/INT/TERM 清理本次已启动进程；确认无占用后再启动，不要手工复制不明 PID。 |
| Worker/API 不就绪 | 看 `.runtime/app/logs/worker.log`、`api.log` 和 readiness 响应，检查数据库/磁盘、ComfyUI 回环访问与预热；Key 不应出现在日志。 |
| 浏览器打不开/401/限流 | 检查防火墙、实际 `AIGC_APP_PORT` 监听和 API Key；401 不通过关闭鉴权解决，限流应等待并检查请求频率。 |

所有修复均保持物理 GPU 1、完整身份链和 1024×1024，并新增/更新测试和真实报告。
