# 模型、GPU 与容量评估

## 1. 当前路线与实测边界

当前默认采用 **官方 SDXL Base 1.0 + InstantID + antelopev2 + 每风格一个 SDXL LoRA**；Turbo 在风格 LoRA 后增加 LCM-LoRA。RealVisXL V5 + PhotoMaker V2 已因四风格实测区分度不足退出当前工作流，仅作为可选回滚材料。

1024×1024、batch 1、Standard 25 步/CFG 4.0、Turbo 8 步/CFG 1.8 是新路线的初始基线，不代表画质或性能已经验收。InstantID 是否优于 PhotoMaker V2 必须由同输入、同 seed 的真实 A/B 决定。

**单 GPU 结论仍为一个活跃生成任务。** Worker 串行执行，其他请求排队。30 秒是热态、无排队、单并发的目标，不是预填结果。

## 2. 各组件职责

- **官方 SDXL Base 1.0**：当前基础生成模型，包含 CLIP、UNet 和 VAE。四个风格 LoRA 主要按这一架构训练，选择它是为了避免 RealVisXL 的强写实先验限制动漫、水墨和油画。
- **InstantID IP-Adapter**：将 antelopev2 人脸 embedding 注入身份条件，与文本条件共同影响生成。
- **InstantID ControlNet**：使用人脸关键点约束姿态和面部结构。它增加模型文件、计算与显存，不应被描述为无成本替换。
- **antelopev2**：用于人脸检测、关键点和身份特征，不判断姓名。输入和输出还各有一次单人脸守卫。
- **四个风格 LoRA**：赛博朋克、动漫、水墨、油画每次只加载一个，初始强度为 1.0、2.0、0.8、2.0。强度是 A/B 起点，不是最优值。
- **LCM-LoRA**：Turbo 的低步数加速组件。Turbo 与 Standard 分开评价，不预设谁在所有风格上更好。

## 3. 为什么替换 RealVisXL + PhotoMaker V2

RealVisXL 偏真实皮肤、摄影光影和写实人像。PhotoMaker V2 的 ID Encoder、InsightFace 与 UNet LoRA 能保留身份，但实测中它与写实基础模型共同压制了跨媒介风格，出现“原脸略变、背景略变”的结果。

这并不证明 PhotoMaker V2 或 RealVisXL 本身质量差，而是说明它们与当前四风格目标不匹配。InstantID 用 attention 和关键点 ControlNet 提供另一条身份路径，官方 SDXL Base 也更贴近所选风格 LoRA 的训练基座。

## 4. 路线比较

| 路线 | 优势 | 代价/风险 | 当前决定 |
| --- | --- | --- | --- |
| SDXL Base + InstantID | 与风格 LoRA 基座一致，身份和关键点分路控制 | ControlNet/IP-Adapter 增加下载、显存和兼容性风险 | 当前默认，待真实 A/B |
| RealVisXL + PhotoMaker V2 | 写实人像和既有部署链成熟 | 四风格实测弱，身份与风格互相抑制 | 可选回滚，不进入当前图 |
| RealVisXL V4 | 可对照 PhotoMaker 官方示例 | 仍有写实先验 | 可选，不默认加载 |
| InstantID + 社区 checkpoint | 可能进一步提高风格 | 每个 checkpoint 的许可、兼容性和缓存都需重新验收 | 后续候选 |
| FLUX.1-dev + PuLID/InfiniteYou | 视觉上限有吸引力 | 不同架构，单张 3090 的显存和时延风险更高 | 不在本次替换范围 |

## 5. A/B 方法

1. 固定获授权的单人脸输入、seed、风格 LoRA、分辨率和输出守卫。
2. 先 Standard 后 Turbo，逐风格比较，不把不同 seed 的结果当作公平对照。
3. 记录成功/失败、输出人脸数量、检测误拒、端到端 P50/P95、GPU 1 峰值和 OOM。
4. 至少两名评审盲看风格与身份；不能只选一张好图。
5. 每次只改 checkpoint、InstantID 权重、风格强度、CFG 或步数中的一个变量。
6. 未经报告证明，不把 InstantID 的路线更换称为质量提升。

## 6. 显存与容量

**16–22GB 仍是工程估算，不是实测占用。** InstantID 新增约 4.2GB 的 IP-Adapter/ControlNet 权重，官方 SDXL Base 约 6.9GB，antelopev2 还需单独加载；文件大小不能直接等同于显存峰值，必须计入激活、VAE、CUDA 上下文、缓存和碎片。

- 保持物理 GPU 1、1024×1024、batch 1 和单活跃任务。
- 启动要求可用显存达到 `AIGC_GPU_MIN_FREE_MIB`，其他服务不得被停止来伪造空闲。
- 如果实际 OOM，保存日志与报告，不通过删除身份模块或降低分辨率掩盖。
- 扩容仍须依据持续排队和真实服务目标，不以单日估算替代实测。

模型文件以 `config/models.lock.json` 为唯一机器可读清单，节点与 Torch 版本以 `config/versions.lock.env` 和 `vendor/ComfyUI_InstantID/.aigc-commit` 为准。
