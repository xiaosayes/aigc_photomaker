# 验收清单与证据模板

本文件是空白验收计划，不是验收结果。所有复选框初始均未勾选；“预期”不等于“通过”。执行人只能依据命令输出、真实报告、截图和日志填写“未执行 / 通过 / 失败 / 阻塞”，不得删除失败样本或虚构结果。API Key、人脸敏感数据和服务器秘密不得进入报告。

## 验收记录头

| 字段 | 填写 |
| --- | --- |
| 项目版本 / Git commit | 待填写 |
| 服务器、OS、驱动、GPU | 待填写 |
| Conda 环境 | `aigc-photomaker`（实测确认：待填写） |
| 物理 GPU / 映射 | `CUDA_VISIBLE_DEVICES=1`（实测确认：待填写） |
| 模型 manifest / 节点与 ComfyUI commit | 待填写 |
| 执行人、日期、时区 | 待填写 |
| 总结 | 未执行 |

## A. 静态检查（本地证据，不是运行证据）

- [ ] 十个根目录文档、脚本引用和本地 Markdown 链接由 `python scripts/static_check.py --docs` 检查，保存退出码。
- [ ] `python scripts/static_check.py --all` 同时检查 Web、Shell、文档；关键字通过不被记为功能通过。
- [ ] `python -m compileall -q scripts src tests` 的退出码已记录。
- [ ] 工作流及配置 JSON 可解析，Markdown 本地链接目标存在；保存实际使用的只读检查命令和结果。
- [ ] `git diff --check`、`git status --short` 输出已保存，确认无意外改动或秘密。

证据路径/退出码：待填写。限制说明：静态检查不执行安装、Conda、模型、API、浏览器或 GPU 命令。

## B. 服务器功能验收

### B1. 镜像安装与启动

- [ ] 从 ZIP 父目录解压，路径是单层 `aigc-photomaker`，记录 `unzip aigc-photomaker.zip` 和 `cd aigc-photomaker`。
- [ ] 记录 HF/GitHub/pip 镜像配置与显式环境覆盖优先级；不记录秘密值。
- [ ] `bash scripts/install.sh` 完成锁定源码/依赖安装、私有 Key 初始化、模型校验和测试；保存退出码及脱敏日志。
- [ ] `bash scripts/start.sh` 启动 ComfyUI、真实双模式预热、Worker、API；启动中断清理另行测试。
- [ ] `bash scripts/status.sh` 证明 api/worker/comfy 进程、实际配置的 API/ComfyUI 端口（默认 8000/8188）监听、物理 GPU 1、磁盘和 readiness；readiness 503 被正确视为失败。
- [ ] 日志 `.runtime/app/logs/{api,worker,comfy}.log` 无秘密；PID 文件与实际命令匹配。
- [ ] `bash scripts/stop.sh` 与重启正常，无残留进程/端口/孤儿任务。

### B2. 自动化与 API

在已激活服务器的依赖环境中执行，保存完整输出：

```bash
python -m pytest -q tests
bash scripts/smoke_test.sh
```

- [ ] 单元和 Mock ComfyUI 集成测试退出码已记录，但未冒充真实 GPU 证据。
- [ ] Standard 与 Turbo 均返回完整可解码的 1024×1024 RGB PNG。
- [ ] 输入无脸纯色图和双人脸图分别返回结构化错误；输出无脸/多脸由 `FluxSingleFaceOutputGuard` 拒绝且不作为成功结果交付；无效图、超限图、非法风格/模式/参数均验证。
- [ ] 缺 Key/错 Key 为 401，限流生效，结果接口需要鉴权，日志无 Key。
- [ ] 固定 seed 可复现；随机 seed 有记录；取消后任务可查询且状态终止。
- [ ] 默认 `reports/smoke-<UTC时间>.json` 和 `.md` 路径已填写；报告的版本、参数、GPU 证据完整。

### B3. 人工恢复演练

按 `bash scripts/smoke_test.sh` 输出中的 `RECOVERY_INSTRUCTIONS` 分开执行，演练会重启真实服务：

- [ ] Worker 终止并恢复后，原 `prompt_id` 不变、未重复提交，任务最终可查询。
- [ ] ComfyUI 终止并恢复后，已有 prompt 被查询/恢复，未盲目重复生成。
- [ ] 运行中取消、排队中取消、失败任务查询均保留状态与脱敏结构化错误。
- [ ] 启动中 INT/TERM/失败只清理本次进程；磁盘不足、保留期清理不清运行任务。

证据与异常：待填写。

## C. 真实性能（独立证据）

前置：已完成预热、无排队、单并发、1024×1024、完整身份链、物理 GPU 1。输入、风格、seed 序列、软件版本和起止时间均记录。

```bash
bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000 --style cyberpunk
bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000 --style cyberpunk
bash scripts/benchmark.sh --alternate-modes --runs 10 --base-seed 13000
```

- [ ] Standard：至少 20 个成功样本、零失败、无 OOM、真实 GPU 1 峰值显存，P50/P95/min/max 均来自报告。
- [ ] Standard 的 30 秒判定只针对热态端到端 P95 ≤ 30 秒；结果为：未执行。
- [ ] Turbo：至少 10 个成功样本、零失败、无 OOM、真实显存；P95 只报告，不套用 Standard 强制门槛。
- [ ] 交替模式报告只用于切换稳定性，未替代 Standard 单模式门槛。
- [ ] 默认 `reports/benchmark-<模式>-<UTC时间>.json` 与 `.md` 实际文件名已填写；失败样本和 OOM 日志保留。
- [ ] 16–22GB 仍标为工程估算，实测峰值单独填写，未把预算当结果。

| 指标 | Standard | Turbo | 交替模式 |
| --- | --- | --- | --- |
| 成功/失败数 | 待填写 | 待填写 | 待填写 |
| P50 / P95 秒 | 待填写 | 待填写 | 待填写 |
| GPU 1 峰值 MiB | 待填写 | 待填写 | 待填写 |
| 报告路径 | 待填写 | 待填写 | 待填写 |
| 判定 | 未执行 | 未执行 | 未执行 |

## D. 人工画质与身份验收

- [ ] 使用获授权的多张清晰单人照片，覆盖正脸/侧脸、不同肤色、年龄、眼镜和复杂背景；记录同意与数据保留方式。
- [ ] 四种风格 × Standard/Turbo 使用相同输入和固定 seed 对照，样本清单可追溯；记录各模式 FP8 权重、revision/SHA256、采样参数和修改前后配对结果，不以单一 seed 代替多 seed 验证。
- [ ] 至少两名评审独立查看原图/结果，盲评身份相似度、五官完整、手部/边缘伪影、风格符合、细节和 1024×1024 完整性。
- [ ] Turbo 与 Standard 的退化如实记录；时延快不自动等于画质通过。
- [ ] `feature/flux` 的 Standard/Turbo Distilled 与 `main` 的 SDXL + InstantID 分开做同输入同 seed 对照；后续权重、步数或 CFG 调整一次只改一个变量，且重新预热。
- [ ] 不以预热图一张好结果代表整体质量；失败、缺脸、多脸和敏感样本处理有记录。

| 样本集/评审 | 身份（ rubric 待确认 ） | 风格 | 伪影 | 结论/证据 |
| --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | 未执行 |

## E. 真实浏览器验收

- [ ] 在受控客户端访问 `http://服务器IP:实际AIGC_APP_PORT/`，记录浏览器/版本、设备、网络与截图；实际 ComfyUI 端口未暴露公网。
- [ ] 上传合法图片，选择四种风格、Standard/Turbo、随机/固定 seed，看到排队/进度、结果预览和下载。
- [ ] 下载图片完整解码为 1024×1024 RGB PNG；结果 URL 不暴露公共静态目录。
- [ ] 无 Key/错 Key、无效文件、无脸、多脸、取消、失败、超时均有可理解界面反馈。
- [ ] 刷新/新会话的 Key 行为符合 sessionStorage；无 localStorage，Key 不在 URL、截图、日志中。
- [ ] 桌面与移动视口可用，无外部 CDN 依赖；浏览器 Network/Console 证据已脱敏保存。
- [ ] 多人同时提交只验证排队与状态正确，不宣称多并发 GPU 性能。

浏览器证据路径与问题单：待填写。

## F. 完全离线验收（独立于镜像功能）

- [ ] 在联网 Ubuntu 22.04 x86_64 构建机运行 `bash scripts/prepare_offline_bundle.sh`，保存版本和大包清单。
- [ ] 应用 ZIP 与大型 `offline/` 分包上传，目录未嵌套；`offline/conda/aigc-photomaker.tar.gz`、wheelhouse、模型、ComfyUI 源码均存在。
- [ ] 安装前 `(cd offline && sha256sum -c SHA256SUMS)` 全部条目有原始输出；篡改/缺文件负例能失败。
- [ ] 在干净机器/虚拟机和管理员实际禁用网络后安装，记录隔离证据；不是只设置离线变量。
- [ ] `AIGC_OFFLINE=1 bash scripts/install.sh` 使用 conda-pack 解包、命名前缀和 `conda-unpack`，pip 全程 no-index，无隐式下载。
- [ ] 断网状态下模型导入、双模式预热、启动、状态、冒烟均重新执行并保存日志/报告。
- [ ] 停止、重启和一次真实浏览器生成在断网状态复验；离线结果与在线结果不互相借用。

离线证据、网络隔离方式、报告路径和判定：未执行 / 待填写。

## 最终签署

| 类别 | 判定 | 证据索引 | 未解决问题 |
| --- | --- | --- | --- |
| 静态检查 | 未执行 | 待填写 | 待填写 |
| 服务器功能 | 未执行 | 待填写 | 待填写 |
| 真实性能 | 未执行 | 待填写 | 待填写 |
| 人工画质 | 未执行 | 待填写 | 待填写 |
| 浏览器 | 未执行 | 待填写 | 待填写 |
| 完全离线 | 未执行 | 待填写 | 待填写 |

只有各类别具有独立、可复核证据，才能分别签署；任一类别未执行不得写成整体验收通过。
