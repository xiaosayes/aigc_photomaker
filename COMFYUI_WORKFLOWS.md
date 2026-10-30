# ComfyUI 工作流与 API 映射（feature/flux）

## 1. 当前基线

`feature/flux` 使用 **FLUX.2 [klein] 4B Distilled FP8 + 原生 ReferenceLatent 身份/构图参考**。Standard 使用 8 步、CFG 1.0，Turbo 使用 4 步、CFG 1.0；不再加载 Base FP8、SDXL、InstantID IP-Adapter/ControlNet、SDXL 风格 LoRA 或 SDXL LCM-LoRA。

| 文件 | 用途 |
| --- | --- |
| `workflows/standard_api.json` | Standard API 图 |
| `workflows/turbo_api.json` | Turbo API 图 |
| `workflows/standard_ui.json`、`turbo_ui.json` | 可编辑 UI 图 |
| `src/aigc_photomaker/workflow.py` | 模板与任务参数注入 |
| `scripts/build_workflows.py` | 从代码重建四个 JSON |

## 2. Standard 与 Turbo

Standard：

```text
UNETLoader = flux-2-klein-4b-fp8.safetensors
Flux2Scheduler = 8 steps
CFGGuider = 1.0
KSamplerSelect = euler
```

Turbo：

```text
UNETLoader = flux-2-klein-4b-fp8.safetensors
Flux2Scheduler = 4 steps
CFGGuider = 1.0
KSamplerSelect = euler
```

两种模式共用：

```text
CLIPLoader = qwen_3_4b.safetensors, type=flux2
VAELoader = flux2-vae.safetensors
EmptyFlux2LatentImage = 1024×1024, batch 1
```

## 3. 完整节点链

```text
1 UNETLoader（按模式选择 FP8）
2 CLIPLoader（Qwen3 文本编码器）
3 VAELoader（FLUX.2 VAE）
4 InstantIDFaceAnalysis（仅复用 antelopev2 做人脸检测）
5 LoadImage
18 ImageScale（lanczos，1024×1024，center crop）
16 FluxSingleFaceInputGuard（18 + 4）
19 VAEEncode（16 + 3）
6/7 CLIPTextEncode（风格正负提示词）
20/21 ReferenceLatent（6/7 + 19）
15 CFGGuider（1 + 20 + 21 + 当前 CFG）
13 Flux2Scheduler（当前步数，1024×1024）
17 RandomNoise（任务 seed）
10 KSamplerSelect（euler）
8 EmptyFlux2LatentImage
9 SamplerCustomAdvanced（17 + 15 + 10 + 13 + 8）
11 VAEDecode（9 + 3）
14 FluxSingleFaceOutputGuard（11 + 4）
12 SaveImage（14）
```

ReferenceLatent 将上传照片的身份与构图信息接入正负条件，代替 InstantID。节点 4 仅用于人脸数量检查，不加载 InstantID 身份权重。`ImageScale` 强制参考图和输出链使用 1024×1024，center crop 可能裁去边缘，须纳入人物完整性验收。

## 4. 模型目录

安装后位于 `.runtime/ComfyUI/models/`：

| 相对路径 | 用途 |
| --- | --- |
| `diffusion_models/flux-2-klein-base-4b-fp8.safetensors` | 可选回滚 Base FP8，默认不加载 |
| `diffusion_models/flux-2-klein-4b-fp8.safetensors` | Turbo Distilled FP8 |
| `text_encoders/qwen_3_4b.safetensors` | Qwen3 文本编码器 |
| `vae/flux2-vae.safetensors` | FLUX.2 VAE |
| `instantid/antelopev2.zip` | 人脸模型归档与 hash 证据 |
| `insightface/models/antelopev2/` | 输入/输出守卫使用的 ONNX |

精确 revision、大小和 SHA256 以 `config/models.lock.json` 为准。旧 SDXL、InstantID 和风格 LoRA 均为可选回滚项，默认不下载、不加载。

## 5. 人脸策略与 API

输入无脸/多脸分别返回 `NO_FACE_DETECTED` / `MULTIPLE_FACES_DETECTED`。输出无脸/多脸分别返回 `OUTPUT_NO_FACE_DETECTED` / `OUTPUT_MULTIPLE_FACES_DETECTED`。艺术化输出可能发生检测误报或漏报，须保留真实样本，不得绕过守卫。

| 步骤 | API |
| --- | --- |
| 健康检查 | `GET /system_stats` |
| 上传 | `POST /upload/image` |
| 提交 | `POST /prompt` |
| 查询 | `GET /history/{prompt_id}` |
| 取消 | `POST /queue`、`POST /interrupt` |
| 下载 | `GET /view` |

Worker 修改节点 1 的模式权重、节点 2/3 的文本/VAE 文件、节点 4 的 provider、节点 5 的输入图、节点 6/7 的风格提示词、节点 17 的 seed、节点 13 的步数、节点 15 的 CFG 和节点 12 的输出前缀。

诊断最新任务：

```bash
python scripts/inspect_latest_task.py
```

应检查 `flux2_reference_chain_present`、`flux2_samplers_present` 和 `output_guard_present`，不能只确认模型文件存在。

## 6. 安装与验收

更新模型或 vendor 后须重新安装并受管重启。预热分别执行 Standard 20 步和 Turbo 4 步，强制验证 1024×1024 RGB PNG 和输出单人脸。

先 Standard 后 Turbo，使用同一输入和 seed 比较四风格辨识度、身份、构图、额外人脸和耗时。原 Base 路线实测 20 步采样约 44 秒、总执行约 51–69 秒且质量不佳，现已改为 Standard/Turbo 共用 Distilled，分别 8/4 步。未经新 A/B，不得宣称优化已达标。
