# 模型、GPU 与容量评估（feature/flux）

## 1. 当前路线

`feature/flux` 默认使用 **FLUX.2 [klein] Base 4B FP8（Standard）与 Distilled 4B FP8（Turbo）**，身份与构图参考由原生 ReferenceLatent 提供。官方 SDXL Base + InstantID 保留为 `main` 的历史路线，本分支不再加载其权重。

- Standard：20 步、CFG 4.0、euler；
- Turbo：4 步、CFG 1.0、euler；
- 共用 Qwen3 文本编码器和 FLUX.2 VAE；
- 第一版使用原生提示词风格，不使用 SDXL 风格 LoRA 或 SDXL LCM。

这些参数是新基线，不是画质或性能承诺。Base 与 Distilled 的比较必须分开进行，不能把不同权重下的同 seed 结果当成严格单变量对照。

## 2. 各组件职责

- **Base FP8**：Standard 的未蒸馏生成模型，偏质量和可调整性。
- **Distilled FP8**：Turbo 的低步数模型，偏交互速度。FP8 用于控制单张 3090 的显存和加载成本，兼容性/画质须实测。
- **Qwen3 4B 文本编码器**：编码风格提示词，不是 SDXL CLIP。
- **FLUX.2 VAE**：负责参考图编码和生成潜图解码。
- **ReferenceLatent**：将参考图的身份与构图信息接入正负条件，替代 InstantID 的独立身份模块。
- **antelopev2**：仅用于输入/输出单人脸检查，不负责 FLUX 身份注入。

## 3. 可复用与重写边界

复用：FastAPI、鉴权、上传规范化、SQLite、Worker、取消/恢复、保留期清理、四风格文本、前端、ComfyUI HTTP/WebSocket 和 antelopev2 守卫。

重写：UNET/Qwen/VAE 加载、ReferenceLatent、Flux2Scheduler、CFGGuider、RandomNoise、SamplerCustomAdvanced、1024 参考图处理、双权重切换、模型锁和性能基线。

不兼容：RealVisXL/SDXL checkpoint、InstantID IP-Adapter/ControlNet、四个 SDXL 风格 LoRA、SDXL LCM-LoRA。它们不能直接串联到 FLUX Transformer。

## 4. 与旧路线比较

| 路线 | 优势 | 代价/风险 | 当前决定 |
| --- | --- | --- | --- |
| FLUX.2 Base + Distilled | 原生参考图编辑、统一生成/编辑、两种权重适配双模式 | 双模型下载和缓存、Qwen 文本链、显存与时延待实测 | `feature/flux` 默认 |
| SDXL Base + InstantID | 现有主分支链成熟、身份控制明确 | 四风格曾出现写实倾向与身份/风格冲突 | `main` 保留，不合并且不覆盖 |
| RealVisXL + PhotoMaker V2 | 写实人像与既有部署经验 | 四风格区分度不足 | 可选回滚材料 |
| FLUX.2 专用风格 LoRA | 可能进一步增强风格 | 必须确认 FLUX.2 原生兼容和许可 | 后续单独批准，不混用 SDXL LoRA |
| FLUX.1/FLUX.3 其他版本 | 可作为后续模型研究 | 不同权重、文本编码、许可与生态 | 不在当前实现范围 |

## 5. 显存、时延与验收

新增模型下载约 16.5GB。权重文件大小不能直接等同于显存峰值：还要计入文本编码、VAE、参考图条件、采样激活、CUDA 上下文、双 UNet 缓存和碎片。16–22GB 只能作为工程估算，真实值以 GPU 1 监控为准。

- 保持物理 GPU 1、1024×1024、batch 1 和单活跃任务。
- Standard/Turbo 不并行执行；交替模式用于测量权重切换，不替代单模式基准。
- 记录首次加载、热态 P50/P95、峰值显存、OOM 和其他服务的影响；不得停止已有服务制造达标结果。
- Standard 至少 20 个成功样本，Turbo 至少 10 个，另做双权重交替测试。
- 四风格以同输入同 seed 配对，记录风格辨识度、身份、参考构图和输出人脸数量。未经报告证明，不宣称 FLUX.2 优于旧路线。

**单 GPU 数量结论仍为一张 RTX 3090 支撑一个活跃生成任务。** 只有持续排队和真实服务目标无法满足时才讨论扩容；扩容前必须分别评估权重加载、CPU offload、网络、磁盘和其他 GPU 服务，不能仅靠增加 Worker 或停止已有服务解决。
