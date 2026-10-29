# 人物风格化生成项目设计（已确认）

- 设计日期：2026-10-26
- 项目代号：`aigc-photomaker`
- 核心引擎：ComfyUI
- `feature/flux` 当前模型路线：FLUX.2 [klein] Base 4B FP8（Standard）/ Distilled 4B FP8（Turbo）+ 原生 ReferenceLatent；`main` 的 SDXL + InstantID 不被本分支修改
- 加速路线：FLUX.2 [klein] Base + Distilled FP8 Turbo
- 目标服务器：Ubuntu 22.04 x86_64、Conda、一张 RTX 3090 24GB
- GPU 绑定：物理卡 1（第二张卡），通过 `CUDA_VISIBLE_DEVICES=1` 隔离
- 设计状态：架构、模型方向、性能目标、安全与部署原则均已与需求方确认

## 1. 背景与目标

项目用于将一张人物照片转换为赛博朋克、动漫、水墨、油画等风格的新图片。系统不仅提供 ComfyUI 工作流，还包含可直接对接工作流的 Web 应用、API、持久化任务系统、Worker、部署脚本、测试、性能基准及中文文档。

当前项目用于实操与内部研究，模型选型不以商用授权为约束，但仍保留第三方许可证和模型来源记录。许可证清单不构成法律意见；若后续用于商业服务，必须重新审核所有模型、代码、字体、示例图片及人脸数据的授权。

### 1.1 核心成功标准

1. 用户可通过浏览器上传人物图片、选择风格及 Standard/Turbo 模式，并查看进度、预览和下载结果。
2. 程序通过 ComfyUI HTTP API 提交真实工作流，不在 FastAPI 进程中加载生图模型。
3. Standard/Turbo 工作流完整使用 FLUX.2 UNET、Qwen、VAE、原生 ReferenceLatent 和输入/输出单人脸守卫，不以残缺加载方式冒充完整实现。
4. 正常单并发、无排队、服务已预热时，1024×1024 图片从点击生成到看到结果的目标为 30 秒以内。
5. 一张 RTX 3090 24GB 支撑一个活跃生成任务，所有 GPU 进程只能看到物理卡 1。
6. GitHub 不可直连时仍可部署，支持镜像和离线依赖包两种方式。
7. 本地只开发和静态检查，不启动模型、ComfyUI、Conda 环境或 Web 服务；最终交付 ZIP 由用户在 GPU 服务器运行。

### 1.2 使用规模

- 小团队使用，每天约 50–300 张。
- 多人可以同时提交，但首期性能承诺只针对无排队的单并发场景。
- GPU Worker 同时只执行一个生成任务，避免多个任务争抢 24GB 显存。
- 以每张 30 秒估算，300 张约占用 2.5 小时 GPU 时间，当前日生成量不要求增加 GPU 数量。

## 2. 范围

### 2.1 首期交付范围

- 响应式 Web 页面，无 Node.js 构建依赖。
- FastAPI 后端与静态页面同源托管。
- API Key 鉴权、请求限流、上传校验和结构化错误。
- SQLite WAL 持久化任务、独立单并发 Worker、任务租约和异常恢复。
- ComfyUI Standard 与 Turbo 两套可编辑工作流。
- 工作流 UI JSON 与程序调用的 API JSON。
- 四种内置风格：赛博朋克、动漫、水墨、油画。
- 1024×1024 PNG 输出，随机种子或可复现的固定种子。
- 启动预热、健康检查、日志、结果保留期清理。
- Conda 安装、镜像下载、离线安装、启动、停止、状态、冒烟测试及基准测试脚本。
- 单元测试、Mock ComfyUI 集成测试及真实 GPU 验收脚本。
- 根目录中文设计、部署、模型评估、工作流、验收、阶段计划及阶段 Prompt 文档。

### 2.2 明确不在首期范围

- 用户注册、支付、多租户配额、管理后台和多机调度。
- 多 GPU 并行生成、跨服务器队列和 Kubernetes。
- 每人脸训练 LoRA、视频生成、批量出图、高清二次修复和放大。
- 在 InstantID 之外继续叠加其他身份模块或额外 ControlNet。
- 每次请求动态切换基础大模型。
- 完整内容审核平台或匿名公网开放服务。

## 3. 模型选型

### 3.1 推荐组合

| 组件 | 默认选择 | 作用 |
| --- | --- | --- |
| 基础生图模型 | `black-forest-labs/FLUX.2-klein-base-4b-fp8` 与 `FLUX.2-klein-4b-fp8` | Standard Base 与 Turbo Distilled 双权重路线 |
| 身份与构图参考 | FLUX.2 原生 ReferenceLatent | 将上传照片的身份和构图信息接入正负条件 |
| 文本与 VAE | Qwen3 4B 与 FLUX.2 VAE | 文本条件、参考图编码和结果解码 |
| 人脸检测与特征 | InsightFace `antelopev2` | 仅执行输入/输出单人脸守卫 |
| Turbo 加速 | `latent-consistency/lcm-lora-sdxl` | 将采样压缩到 8–12 步 |
| 工作流引擎 | ComfyUI，固定不可变提交 | 模型加载、图执行、采样、VAE 解码和图片输出 |
| InstantID 节点 | `cubiq/ComfyUI_InstantID`，固定不可变提交并保留 Apache-2.0 许可证 | 在 ComfyUI 中执行 InstantID 及单人脸守卫 |

模型文件版本、下载地址和 SHA256 由 `config/models.lock.json` 统一记录，ComfyUI 与节点提交由 `config/versions.lock.env` 固定。安装过程不追踪分支最新提交，也不在每次启动时自动升级。

### 3.2 为什么不是其他路线

- **PhotoMaker V1**：速度和许可边界较简单，但单图身份保持能力不及 V2，不再作为默认路线。
- **InstantID**：身份保持能力强，但额外 ControlNet 会增加计算与显存开销，关键点约束也可能限制强风格变化。
- **FLUX.1-dev + PuLID/InfiniteYou**：视觉上限较高，但在一张 3090 上同时满足 24GB 显存和 30 秒端到端目标的风险更大。
- **SDXL Lightning 直接替换基础模型**：速度快，但与身份模块组合后的画质和相似度需要额外验证，不作为默认质量路线。

### 3.3 RealVisXL V5.0 与 SDXL 的关系

RealVisXL V5.0 是基于 SDXL 架构继续训练和调优的写实向权重，不是新的 SDXL 官方架构。它与官方 SDXL Base 1.0 使用基本相同的 ComfyUI 节点、分辨率体系和计算流程，因此在相同参数下，显存和速度通常接近，主要区别是模型先验与画面风格。

- 写实人像、电影感和写实赛博朋克：优先 RealVisXL V5.0。
- 动漫、水墨、油画：RealVisXL 不一定始终更好，需要风格模板和固定种子实测。
- 人物相似度：主要由输入照片、PhotoMaker V2、身份权重、风格和采样参数共同决定，不能只看基础模型名称。
- PhotoMaker V2 官方示例重点验证过 RealVisXL V4.0，因此 V5.0 虽为默认候选，仍须通过 A/B 验收后定版。

项目保留 RealVisXL V4.0 和 `stabilityai/stable-diffusion-xl-base-1.0` 的配置化替换能力，但默认只下载和加载一个基础模型，避免额外磁盘、显存和切换时延。

### 3.4 风格策略

当前使用经过整理的提示词模板，并为每种风格串联一个固定 revision、大小和 SHA256 的 SDXL LoRA：

1. **赛博朋克**：霓虹、雨夜、未来都市、电影级布光，同时约束面部清晰和身份保持。
2. **动漫**：精致二维动画、清晰线稿、协调色彩，避免低质量五官和多余肢体。
3. **水墨**：宣纸、留白、墨色层次和东方美学，避免仅给照片叠加灰色滤镜。
4. **油画**：可见笔触、画布纹理、古典色彩和柔和布光，避免直接复制具体在世艺术家风格。

风格模板由后端受控配置注入，工作流中的身份触发词必须保留，不能因用户提示词覆盖而失效。

## 4. 系统架构

```text
浏览器 Web UI
  │ 上传、风格、Standard/Turbo、种子、进度、预览、下载
  ▼
FastAPI 应用
  ├── API Key、限流、输入校验、EXIF 清理、任务 API
  ├── SQLite WAL：任务、状态、参数、错误、时间指标、租约
  ├── data/inputs：规范化输入
  └── data/outputs：生成结果
  ▼
独立 Worker（一个活跃生成任务）
  │ 原子认领任务、填充 API 工作流、上传图片、提交 prompt
  │ 查询执行状态、下载结果、原子写回、记录指标和错误
  ▼
ComfyUI 127.0.0.1:8188
  ├── Standard：FLUX.2 Klein Base FP8 + ReferenceLatent
  └── Turbo：FLUX.2 Klein Distilled FP8 + ReferenceLatent，低步数采样
  ▼
物理 GPU 1（CUDA_VISIBLE_DEVICES=1，进程内显示为 cuda:0）
```

### 4.1 进程边界

- **Web/API 进程**：处理 HTTP 和数据库，不加载 PyTorch 生图模型。
- **Worker 进程**：负责编排和状态恢复，不直接执行扩散采样。
- **ComfyUI 进程**：独占当前 GPU 的一个生成任务，只监听回环地址。
- 三个进程使用同一 Conda 环境，但通过不同命令和 PID 文件管理，可分别查看日志和重启。

### 4.2 为什么首期不使用 Redis

当前只有单机和一个 GPU Worker。SQLite WAL、原子状态更新和租约已经能够覆盖持久化与崩溃恢复需求，减少离线安装和运维组件。若后续增加多 Worker 或多机调度，再将队列抽象替换为 Redis/RQ、Celery 或专门任务平台。

### 4.3 前端策略

使用 HTML、CSS 和原生 JavaScript，交付时无需 npm、Vite 或浏览器 CDN。页面通过 `fetch` 调用同源 API；API Key 仅保存在当前浏览器会话，不写入 URL、日志或结果文件。结果通过鉴权接口读取为 Blob 后展示，不暴露公共图片目录。

## 5. 核心数据流

1. 用户选择图片、风格和生成模式，可选填写固定种子。
2. 浏览器提交 multipart 请求，FastAPI 校验 API Key、请求频率和表单字段。
3. 后端检查真实文件格式、解码图片、限制像素量、处理 EXIF 方向、转 RGB 并移除元数据。
4. 后端以 UUID 和任务 ID 组织输入路径，创建 `queued` 任务并立即返回任务 ID。
5. Worker 原子认领任务，读取对应 Standard/Turbo API 工作流。
6. Worker 将规范化图片上传到 ComfyUI，填充图片名、提示词、种子、步数、CFG、采样器和输出前缀。
7. ComfyUI 返回 `prompt_id`；Worker 持久化该 ID 并监听执行状态。
8. 成功后，Worker 根据 history 中的输出信息调用 `/view`，校验图片并原子保存到结果目录。
9. 数据库记录完成状态、输出路径、各阶段耗时和错误信息，前端轮询到完成后获取图片。
10. 保留期清理任务删除过期输入、输出及对应任务记录；运行中的任务不得被清理。

## 6. API 与任务模型

### 6.1 主要接口

| 接口 | 用途 |
| --- | --- |
| `GET /api/v1/health/live` | API 进程存活检查，不暴露敏感配置 |
| `GET /api/v1/health/ready` | 数据库、模型、ComfyUI、Worker 和预热就绪检查 |
| `GET /api/v1/styles` | 返回可用风格、模式和默认参数 |
| `POST /api/v1/tasks` | 上传图片并创建生成任务，成功返回 202 和任务 ID |
| `GET /api/v1/tasks/{task_id}` | 返回状态、进度、警告、计时和结构化错误 |
| `GET /api/v1/tasks/{task_id}/result` | 鉴权读取完成图片 |
| `POST /api/v1/tasks/{task_id}/cancel` | 取消尚未开始或当前正在执行的单个任务 |

健康检查可不鉴权，但不得返回 API Key、服务器绝对路径、完整环境变量或模型敏感信息；任务和结果接口必须鉴权。

### 6.2 状态与异常

主状态采用 `queued`、`running`、`succeeded`、`failed`、`cancelled`。运行阶段另记录上传、提交、执行、结果写回等细粒度状态，避免用含义混乱的百分比驱动业务逻辑。

统一错误至少覆盖：

- `UNAUTHORIZED`、`RATE_LIMITED`
- `INVALID_IMAGE`、`IMAGE_TOO_LARGE`、`TOO_MANY_PIXELS`
- `NO_FACE_DETECTED`、`MULTIPLE_FACES_DETECTED`
- `INVALID_STYLE`、`INVALID_MODE`、`INVALID_PARAMETER`
- `MODEL_MISSING`、`WORKFLOW_INVALID`、`COMFYUI_UNAVAILABLE`
- `GPU_OOM`、`COMFYUI_TIMEOUT`、`RESULT_MISSING`
- `DISK_SPACE_LOW`、`INTERNAL_ERROR`

对外错误包含人类可读说明、是否可重试和任务 ID；详细堆栈仅写服务端日志。

### 6.3 恢复原则

- 数据库更新采用事务，任务认领使用原子条件更新。
- Worker 定期更新心跳和租约，防止异常退出后任务永久停留在运行态。
- 对已有 `prompt_id` 的恢复任务，先查询 ComfyUI history，能恢复结果则不重复生成。
- 无法确定原任务是否仍在执行时先检查队列并中断对应任务，不盲目重复提交。
- GPU OOM 和确定性工作流错误不做无限自动重试；网络错误仅在可确认未成功提交时限次重试。

## 7. ComfyUI 工作流设计

### 7.1 Standard

```text
LoadImage
  → ImageScale（1024×1024，center crop）
  → 输入单人脸守卫
  → VAEEncode
  → 正负 ReferenceLatent

UNETLoader（Standard Base FP8 / Turbo Distilled FP8）
  → CFGGuider
  → Flux2Scheduler + RandomNoise + KSamplerSelect
  → SamplerCustomAdvanced
  → VAEDecode
  → 输出单人脸守卫
  → SaveImage

负向提示词 → CLIPTextEncode → ReferenceLatent → CFGGuider
EmptyFlux2LatentImage 1024×1024 → SamplerCustomAdvanced
SamplerCustomAdvanced → VAEDecode → 输出单人脸守卫 → SaveImage
```

节点必须覆盖 UNET、Qwen、VAE、ReferenceLatent、输入/输出守卫、正负面条件、Flux2Scheduler、RandomNoise、CFGGuider、SamplerCustomAdvanced、解码和保存。应用只修改声明过的输入映射，不依赖节点界面坐标。

### 7.2 Turbo

Turbo 使用独立的 Distilled FP8 权重，配合 Flux2Scheduler、euler 和 4 步低 CFG。Standard/Turbo 是独立 JSON 和独立质量档位，不在 Standard 超时后静默替换，也不通过降低输出分辨率伪装加速。

### 7.3 参数基线

- Standard：1024×1024、batch 1、Base FP8、20 步、CFG 4.0、euler。
- Turbo：1024×1024、batch 1、Distilled FP8、4 步、CFG 1.0、euler。
- 人脸 Provider：分别测试 CPU 与 CUDA；默认选择通过画质、时延和显存验收的配置。
- 不启用 Refiner、Hires Fix、ControlNet、批量生成和 upscale。
- 种子来自安全随机数或用户输入，并随任务保存以支持复现。

所有默认值集中在配置和风格模板中，API 工作流与 UI 工作流保持相同节点语义。

## 8. GPU、显存与性能设计

### 8.1 GPU 结论

一张 RTX 3090 24GB 足以承载本设计的单并发 Standard/Turbo 服务，不需要第二张生成 GPU。工程显存预算按约 16–22GB 控制，最终数值必须由服务器峰值采样确认，不能将估算写成实测结论。

如果出现 OOM，优先检查其他进程、重复模型加载、人脸 CUDA 上下文、预览方法和 ComfyUI 模型管理参数。不能首先通过降低分辨率或偷偷减少身份模块来规避问题。

### 8.2 第二张卡的正确指定方式

服务器物理卡编号为 0、1，项目使用编号 1。所有 GPU 相关进程设置：

```bash
export CUDA_VISIBLE_DEVICES=1
```

此时进程内的可见设备被重新映射为逻辑 `cuda:0`。启动脚本不再传 `--cuda-device`：ComfyUI 0.42.1 会用该参数覆盖 `CUDA_VISIBLE_DEVICES`，从而把逻辑 0 错误变成物理 0。脚本同时用 `nvidia-smi --id=1` 做物理卡检查，并在日志中记录映射关系。

### 8.3 30 秒口径

30 秒从用户点击生成开始，到浏览器取得并显示结果为止，包含上传、校验、任务提交、人脸处理、身份编码、采样、VAE 解码、结果写回、下载和状态轮询。当前只考核没有排队任务的单并发场景。

服务在完整预热前保持未就绪，因此正常用户请求不承担模型冷加载。安装、首次模型下载和服务启动时间不属于单图生成时延。

设计预算如下，实际以验收为准：

| 阶段 | 目标预算 |
| --- | ---: |
| 上传、校验、规范化和落盘 | 0.2–1 秒 |
| Worker 认领和 ComfyUI 提交 | 小于 0.5 秒 |
| 人脸分析、身份编码和提示词处理 | 0.5–2 秒 |
| Standard 采样 | 约 20–22.5 秒的主要耗时预算 |
| VAE 解码、PNG 写回和显示 | 0.5–2 秒 |
| 状态轮询余量 | 小于 1 秒 |

### 8.4 性能验收

- 使用固定的授权测试图片和固定种子，先完成至少一次全链路预热。
- Standard 至少连续运行 20 次、Turbo 至少连续运行 10 次，报告 P50、P95、最小值、最大值、失败率和各阶段耗时；另做 Standard/Turbo 交替测试，覆盖 LoRA 缓存复用、模式切换开销和显存峰值。
- 使用 `nvidia-smi` 采样物理卡 1 的显存峰值，并检查是否误用其他 GPU。
- Standard 验收目标为预热后单并发 P95 不超过 30 秒且无 OOM。
- 画质与相似度必须同时进行人工 A/B，不以速度作为唯一验收指标。
- 若 Standard 超时，依次检查重复加载和其他 GPU 进程、调整采样器与步数、比较人脸 Provider，最后再使用 Turbo；不得隐藏参数变化。

## 9. 输入安全、隐私与运行保护

- 支持 JPEG、PNG、WebP，默认文件大小上限 10MB。
- 同时检查扩展名、文件签名和真实图片解码结果，并限制总像素数。
- 自动处理 EXIF 方向、转换 RGB、移除 EXIF，使用 UUID 文件名，不信任原始路径。
- 首期只接受单人脸图片；无人脸或多人脸返回可操作提示，不静默选错身份。人脸数量校验和身份特征提取复用同一次人脸分析结果，不为校验重复执行检测。
- 所有任务与结果接口使用 `X-API-Key`，配置按 IP 和 API Key 限流。
- ComfyUI 只监听 `127.0.0.1`，公网只暴露经过反向代理 TLS 保护的 Web/API。
- 默认结果保留 7 天，可配置；清理输入、输出和数据库记录，不清理仍在执行的任务。
- 日志包含任务 ID、状态、错误码和耗时，不记录 Base64 图片、API Key 或完整鉴权头。
- 页面提示用户仅上传已获授权的人物照片。人脸图片及其嵌入具有敏感性，不得用于识别真实身份或未经授权的画像分析。
- 首期面向受信用户，不开放匿名公网访问；面向公众上线前必须补充内容审核、滥用处置和更完整的访问控制。

## 10. 可观测性与运维

- 使用 JSON 结构化日志，包含 UTC 时间、级别、组件、任务 ID、事件和耗时。
- ComfyUI、API、Worker 分文件记录，启动脚本为每次运行保留日志并在状态命令中给出位置。
- `/health/ready` 检查数据库、磁盘、关键模型文件、ComfyUI、Worker 心跳和预热状态。项目内置来源及许可证明确的 `assets/warmup-portrait.png`，启动时分别完成 Standard 与 Turbo 全链路预热，避免用人造无脸图片导致预热失败。
- `status.sh` 显示 PID、端口、最近错误、物理卡 1 显存及磁盘空间，不打印秘密配置。
- 启动前检查模型清单和 SHA256，避免下载半截文件后进入生成阶段。
- 提供优雅停止：先停止接收新任务，再处理或标记当前任务，最后停止 ComfyUI。
- 提供可选 systemd 单元模板；无 root 权限时可用项目脚本和 PID 文件运行。

## 11. 测试策略

### 11.1 本地约束

本地只编辑、审查和执行不启动应用的静态检查，例如 Python 语法编译、JSON 解析、Shell 语法检查和 ZIP 内容核对。不安装 Conda 依赖，不导入生图模型，不启动 ComfyUI、Worker 或 Web 服务，也不声称本地测试已经覆盖 GPU 行为。

### 11.2 服务器测试

1. **单元测试**：图片校验、配置、风格模板、状态机、租约、错误映射、保留期清理。
2. **工作流测试**：JSON 格式、必需节点、连接、参数映射、模型文件名及两种模式差异。
3. **Mock 集成测试**：上传、提交、history、结果下载、超时、断连、执行错误和恢复。
4. **真实 GPU 冒烟测试**：Standard/Turbo 各生成一张，验证输出、日志、GPU 编号和人物身份。
5. **性能测试**：预热后的连续请求、P50/P95、显存峰值和 30 秒目标。
6. **质量 A/B**：RealVisXL V5.0、V4.0、SDXL Base 的固定图片、风格和种子对比；基础模型切换在独立测试窗口执行。
7. **恢复测试**：Worker 重启、ComfyUI 重启、超时、OOM、无效图片和磁盘不足。
8. **部署测试**：镜像路径与离线路径均从干净 Conda 环境验证，校验包哈希。

## 12. 镜像与离线交付设计

### 12.1 应用 ZIP

应用 ZIP 包含源码、静态前端、两套工作流、`vendor/ComfyUI_InstantID` 固定版本节点源码及其许可证、可选回滚的 `vendor/ComfyUI-PhotoMaker-Plus`、依赖锁文件、脚本、测试、示例配置和全部文档。ZIP 不包含 ComfyUI 完整源码、模型权重、PyTorch 大型 wheel、Conda 本身、生成结果、日志或 API Key；ComfyUI 由镜像或离线源码包安装。

### 12.2 镜像安装

- `HF_ENDPOINT`：支持 Hugging Face 镜像。
- `PIP_INDEX_URL`、`PIP_EXTRA_INDEX_URL`：支持 Python 包镜像和 PyTorch CUDA wheel 来源。
- GitHub 代理参数：用于 ComfyUI 和固定版本节点源码。
- 每个模型支持独立 URL 覆盖，防止某一镜像缺少文件。
- 所有下载使用临时文件、失败重试、断点或完整重下以及 SHA256 校验，校验成功后原子改名。

### 12.3 离线安装

`prepare_offline_bundle.sh` 在可联网的 Ubuntu x86_64 机器上准备与服务器 Python/CUDA 匹配的离线包，内容包括 ComfyUI 源码、自定义节点、Python wheelhouse 和模型。GPU 服务器上传应用 ZIP 与离线包后，以 `OFFLINE=1` 安装；离线模式禁止隐式访问公网。

模型与 wheelhouse 因体积较大独立于应用 ZIP。离线清单记录来源、不可变版本、文件大小和 SHA256，避免镜像更新导致依赖漂移。

## 13. Conda 与脚本

- Conda 环境名：`aigc-photomaker`。
- Python：3.11，兼顾 ComfyUI、InsightFace 和科学计算依赖兼容性。
- PyTorch、CUDA wheel、ComfyUI 和节点版本在锁文件中固定；不盲目跟随每日最新版本。
- 服务器安装脚本不修改系统 CUDA、不卸载现有驱动、不替换其他 Conda 环境。

```text
scripts/install.sh                 创建隔离环境并安装依赖
scripts/download_models.sh         下载、导入和校验模型
scripts/start.sh                   检查卡 1、启动三进程、预热并等待就绪
scripts/stop.sh                    优雅停止服务
scripts/status.sh                  输出服务与 GPU 状态
scripts/smoke_test.sh              执行真实 GPU 冒烟测试
scripts/benchmark.sh               执行性能与显存基准
scripts/prepare_offline_bundle.sh  在联网 Ubuntu 机器制作离线包
```

## 14. 根目录文档

| 文档 | 内容 |
| --- | --- |
| `README.md` | 项目功能、目录、快速开始和常用命令 |
| `DESIGN.md` | 本设计与决策依据 |
| `DEPLOYMENT.md` | Ubuntu、Conda、镜像、离线包、卡 1、启动和故障排查 |
| `MODEL_GPU_EVALUATION.md` | 模型对比、RealVisXL/SDXL 关系、显存、数量和性能评估 |
| `COMFYUI_WORKFLOWS.md` | 节点图、模型放置、导入、参数和程序 API 对接细节 |
| `ACCEPTANCE.md` | 功能、画质、30 秒、显存、恢复及离线验收清单 |
| `ROADMAP.md` | 后续实施方案、阶段目标、输入输出和退出条件 |
| `PROMPTS.md` | 每个后续阶段可直接交给编码代理执行的 Prompt |
| `THIRD_PARTY_NOTICES.md` | 模型、代码、示例素材来源和许可证提示 |

## 15. 后续演进边界

- **质量增强**：在独立测试阶段评估 InstantID、FLUX 身份保持、风格 LoRA、姿态控制和高清修复，不影响已验证的默认工作流。
- **多分辨率**：增加 832×1216 等 SDXL 适配比例，并重新测量显存与时延。
- **多用户**：增加用户体系、配额、对象存储和内容审核，而不是直接开放当前单 API Key 服务。
- **扩展并发**：一个活跃生成任务对应一张 GPU；增加 Worker、队列和调度器，按 GPU 隔离任务。
- **可复现发布**：每次模型或节点升级都生成新锁文件、质量报告和性能报告，支持回滚。

## 16. 设计验收结论

本设计将“效果与时延最优”解释为在 RTX 3090、24GB 显存、1024×1024 和无排队单并发条件下的综合最优，而非不计资源追求视觉上限。由于旧路线的四风格实测未达到要求，`feature/flux` 改为 FLUX.2 Base/Distilled + 原生 ReferenceLatent。Standard/Turbo 的最终默认参数以真实服务器质量、P95 时延和峰值显存验收结果为准，不预设 FLUX.2 必然更好。
