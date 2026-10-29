# AIGC PhotoMaker

基于 ComfyUI 的单人照片风格化应用：浏览器上传一张单人脸照片，选择赛博朋克、动漫、水墨或油画，使用 Standard（质量优先）或 Turbo（速度优先）生成 **1024×1024 RGB PNG**。`feature/flux` 当前链路为 **FLUX.2 [klein] Base 4B FP8（Standard）/ Distilled 4B FP8（Turbo）+ 原生 ReferenceLatent 身份与构图参考**。输入与输出均执行单人脸守卫。旧 SDXL、InstantID 和风格 LoRA 仅保留为可选回滚材料，`main` 不被本分支修改。

> 当前交付只声明代码与静态结构。模型下载、安装、启动、API、浏览器、画质和性能均须在目标服务器验收；本仓库没有预填服务器成功结果。

## 运行边界

- 目标机：Ubuntu 22.04 x86_64、Conda、RTX 3090 24GB；Conda 环境名由 `AIGC_CONDA_ENV` 配置（当前示例为 `photomaker`，默认值为 `aigc-photomaker`）。
- GPU 固定为**物理 GPU 1**，所有 GPU 进程使用 `CUDA_VISIBLE_DEVICES=1`，进程内设备号是 `cuda:0`。
- 一个 GPU Worker 同时只执行一个生成任务。可以排队提交，但不承诺并发生成吞吐。
- **30 秒是待实测目标，不是既成结果**：仅指服务已预热、无排队、单并发时，从点击生成到结果完整下载并可显示的端到端时延；Standard 验收门槛为至少 20 个成功样本、零失败、无 OOM 且 P95 ≤ 30 秒。首次加载、冷启动和排队时间不属于这个口径。

## 最短镜像快速开始（服务器执行）

把应用 ZIP 放到目标目录后，从其父目录执行。ZIP 解出单层 `aigc-photomaker`，不要再进入同名嵌套目录。

```bash
unzip aigc-photomaker.zip
cd aigc-photomaker
export HF_ENDPOINT=https://hf-mirror.com
export GITHUB_PROXY=https://ghproxy.net/
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/install.sh
bash scripts/download_models.sh
bash scripts/start.sh
bash scripts/status.sh
set -a; source .env; set +a
bash scripts/smoke_test.sh
```

说明：

1. `install.sh` 创建/复用 Conda 环境，安装锁定依赖和 ComfyUI，生成私有 API Key，下载并校验模型，然后运行服务器测试。它需要编译器、Git、curl、tar、Conda 和空闲的物理 GPU 1。
2. `GITHUB_PROXY` 会前置到完整 GitHub URL；`HF_ENDPOINT`、`PIP_INDEX_URL`、`PIP_EXTRA_INDEX_URL`、`PYTORCH_INDEX_URL`、`GITHUB_PROXY` 及显式导出的 `AIGC_*` 优先于 `.env`，不可变版本锁优先于两者。
3. 启动会完成 Standard/Turbo 真实预热，预热素材是真实单人脸 `assets/warmup-portrait.png`，不是随机无脸图。
4. `smoke_test.sh` 从环境读取 `AIGC_API_KEY`。如在新 shell 中运行，先执行 `set -a; source .env; set +a`，不要把 Key 写进命令历史、截图或报告。
5. 默认浏览器地址为 `http://服务器IP:8000/`。若已有服务占用默认端口，不要停止对方；在 `.env` 设置空闲的 `AIGC_APP_PORT` 和 `AIGC_COMFY_PORT`（本次可用 8186 和 8185），浏览器改用实际 API 端口。ComfyUI 始终仅监听 `127.0.0.1`，不要对公网开放。防火墙只按实际访问来源放行实际 API 端口，并优先使用受控内网或反向代理 TLS。

完全离线安装、systemd、日志、排障和大包上传见 [DEPLOYMENT.md](DEPLOYMENT.md)。模型与容量依据见 [MODEL_GPU_EVALUATION.md](MODEL_GPU_EVALUATION.md)，工作流见 [COMFYUI_WORKFLOWS.md](COMFYUI_WORKFLOWS.md)，验收见 [ACCEPTANCE.md](ACCEPTANCE.md)。

## 常用运维与验收（服务器执行）

```bash
# 查看进程、端口、GPU、磁盘和 readiness
bash scripts/status.sh

# 分别测 Standard 与 Turbo；报告写入 reports/
bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000
bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000

# 交替模式只作切换稳定性证据，不能代替 Standard 单模式门槛
bash scripts/benchmark.sh --alternate-modes --runs 10

# 停止；重启即先停止再启动
bash scripts/stop.sh
bash scripts/start.sh
```

本地开发只允许静态检查，不在本地安装、下载或启动服务：

```bash
python scripts/static_check.py --all
python -m compileall -q scripts src tests
git diff --check
```

`static_check.py` 只检查文件、关键契约与引用，**不证明任何命令已运行，也不替代功能、浏览器或 GPU 验收**。阶段计划见 [ROADMAP.md](ROADMAP.md)，独立开发指令见 [PROMPTS.md](PROMPTS.md)，第三方许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目不声明已取得商业使用许可。
