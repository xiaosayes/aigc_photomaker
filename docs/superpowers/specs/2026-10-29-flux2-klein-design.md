# FLUX.2 [klein] 4B 人脸风格化设计

> 分支约束：全部变更只在 `feature/flux`，不合并到 `main`。`main` 的 SDXL + InstantID 基线保持不变。

## 1. 目标与模式

将现有浏览器单人照片风格化应用迁移到 FLUX.2 [klein] 4B，保留四风格、输入/输出单人脸守卫、1024×1024 RGB PNG、任务队列和 Standard/Turbo。

- **Standard**：`flux-2-klein-base-4b-fp8.safetensors`，20 步、CFG 4.0、euler，质量路线。
- **Turbo**：`flux-2-klein-4b-fp8.safetensors`，4 步、CFG 1.0、euler，蒸馏低步数路线。
- 两模式共用 `qwen_3_4b.safetensors` 与 `flux2-vae.safetensors`。FP8 初始选择用于控制单张 3090 的显存和加载成本，画质/兼容性须实测。

## 2. 可复用与重写边界

复用：FastAPI、鉴权、限流、上传规范化、SQLite、Worker、取消/恢复、保留期清理、四风格文本、ComfyUI HTTP/WebSocket、antelopev2 人脸守卫、结果下载和前端。

重写：UNET/Qwen/VAE 加载、ReferenceLatent 身份链、RandomNoise/CFGGuider/Flux2Scheduler/SamplerCustomAdvanced 采样、1024 参考图处理、双 FP8 权重切换、模型锁、预热、诊断、测试和性能基线。

不使用：SDXL CheckpointLoaderSimple、InstantID IP-Adapter/ControlNet、四个 SDXL 风格 LoRA、SDXL LCM-LoRA、SDXL KSampler 参数。旧文件仅作回滚可选材料。

## 3. 锁定模型

| 组件 | 仓库 / revision | 文件 | 大小 | SHA256 |
| --- | --- | --- | ---: | --- |
| Standard UNet | `black-forest-labs/FLUX.2-klein-base-4b-fp8` / `103db268c10d4d3921101b46057671f9ac460da6` | `flux-2-klein-base-4b-fp8.safetensors` | 4089498488 | `44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840` |
| Turbo UNet | `black-forest-labs/FLUX.2-klein-4b-fp8` / `5b4408e59397a4a37ccb46afe426d8ed86379441` | `flux-2-klein-4b-fp8.safetensors` | 4070624520 | `97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6` |
| 文本编码器 | `Comfy-Org/Flux-2-Klein-4B` / `5b3abca5c0dc1ecd4eaf3340868d09063fc1c965` | `split_files/text_encoders/qwen_3_4b.safetensors` | 8044982048 | `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a` |
| VAE | 同上 | `split_files/vae/flux2-vae.safetensors` | 336211292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` |

继续复用 `MonsterMMORPG/tools` 的 antelopev2 归档，revision/hash 保持现有锁。新增下载约 16.5GB，旧模型不默认下载。

## 4. 工作流

```text
1 UNETLoader（按模式选择 Base/Distilled FP8）
2 CLIPLoader（qwen_3_4b，type=flux2）
3 VAELoader（flux2-vae）
4 InstantIDFaceAnalysis（仅用于人脸守卫）
5 LoadImage
18 ImageScale（lanczos，1024×1024，center crop）
16 FluxSingleFaceInputGuard（18 + 4）
19 VAEEncode（16 + 3）
6/7 CLIPTextEncode（当前风格正负提示词）
20/21 ReferenceLatent（6/7 + 19，建立身份/构图参考）
15 CFGGuider（1 + 20 + 21 + 当前模式 CFG）
13 Flux2Scheduler（当前模式步数，1024×1024）
17 RandomNoise（任务 seed）
10 KSamplerSelect（euler）
8 EmptyFlux2LatentImage（1024×1024，batch 1）
9 SamplerCustomAdvanced（17 + 15 + 10 + 13 + 8）
11 VAEDecode（9 + 3）
14 FluxSingleFaceOutputGuard（11 + 4）
12 SaveImage（14）
```

ReferenceLatent 替代 InstantID，负责利用输入照片的身份和构图信息。节点 4 不再加载 InstantID 身份模型，只复用 antelopev2 做输入/输出数量检查。

## 5. 风格策略

第一版使用 FLUX 原生提示词与参考图条件，不加载任何 SDXL LoRA。`StylePreset.lora_filename/lora_strength` 改为可选配置，默认空值；不得把 SDXL LoRA 串入 FLUX。后续若风格不足，只评估明确兼容 FLUX.2 [klein] 4B 的适配层，单独批准。

## 6. 应用与部署

保留当前 API/Worker/任务恢复接口和端口。安装器下载锁定的 FLUX 权重并继续安全解压 antelopev2。预热按 Standard 20 步、Turbo 4 步完整执行，并强制验证 1024×1024 RGB PNG 和单人脸。诊断脚本增加 `flux2_reference_chain_present`、`flux2_samplers_present` 和输出守卫检查。

模型切换时 Standard/Turbo 不得并行执行。需要记录双 UNet 缓存、加载时间、热态时延和峰值显存，不把官方延迟数据当作本机结果。

## 7. 验收

1. 静态：Python/JSON/Shell、四个工作流可重建、敏感路径检查。
2. 服务器：完整 pytest、模型 hash、节点导入、双模式预热、状态与 smoke。
3. 画质：同一输入和 seed 四风格配对，检查风格辨识度、身份、参考构图和额外人脸。
4. 性能：Standard 至少 20 个成功样本，Turbo 至少 10 个，另做双权重交替切换。记录冷/热加载、P50/P95、GPU 1 峰值和 OOM。
5. 任一风格或 Turbo 未经验证不得宣称优于 `main` 的 SDXL + InstantID。
