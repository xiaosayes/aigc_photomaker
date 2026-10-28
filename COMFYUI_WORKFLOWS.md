# ComfyUI 工作流与 API 映射

## 1. 当前基线与交付文件

当前默认路线为 **官方 SDXL Base 1.0 + InstantID + 当前风格 LoRA**。RealVisXL + PhotoMaker 仅作为磁盘上的历史回滚材料，不进入当前工作流。

| 文件 | 用途 |
| --- | --- |
| `workflows/standard_api.json` | Standard API 图 |
| `workflows/turbo_api.json` | Turbo API 图，增加 LCM-LoRA |
| `workflows/standard_ui.json`、`turbo_ui.json` | 对应可编辑 UI 布局 |
| `src/aigc_photomaker/workflow.py` | 节点定义及任务参数注入 |
| `scripts/build_workflows.py` | 从代码重建四个 JSON |

InstantID 节点源码位于 `vendor/ComfyUI_InstantID/`，安装时复制到 `.runtime/ComfyUI/custom_nodes/ComfyUI_InstantID/`。输出目录前缀沿用项目历史名称 `aigc-photomaker`，不代表仍使用 PhotoMaker。

## 2. 节点图

### Standard

```text
1 CheckpointLoaderSimple（官方 SDXL Base 1.0）
2 InstantIDModelLoader（ip-adapter.bin）
3 InstantIDFaceAnalysis（antelopev2）
4 ControlNetLoader（InstantID ControlNet）
5 LoadImage（单人脸输入）
16 InstantIDSingleFaceInputGuard（5 + 3，拒绝输入无脸/多脸）
13 LoraLoaderModelOnly（1 + 当前风格 LoRA）
15 ApplyInstantID（2 + 3 + 4 + 16 + 13 + 正负条件）
8 EmptyLatentImage（1024×1024，batch 1）
9 KSampler（model/positive/negative 均来自 15，latent 来自 8）
11 VAEDecode（9 + VAE 1）
14 InstantIDSingleFaceOutputGuard（11 + 3）
12 SaveImage（14）
```

节点 15 初始 `weight=0.8`、`start_at=0`、`end_at=1`，可通过 `AIGC_INSTANTID_WEIGHT` 调整。Standard 默认 25 步、CFG 4.0、`dpmpp_2m/karras`。InstantID 不使用 PhotoMaker 的 `img` 触发词，因此所有风格模板使用普通 `person`。

### Turbo

Turbo 保留完整 InstantID 和输入/输出守卫，仅把模型链改为：

```text
13 当前风格 LoRA → 10 LCM LoraLoaderModelOnly → 15 ApplyInstantID
```

节点 9 使用 8 步、CFG 1.8、`lcm/normal`。节点 10 的模型来自 13，节点 15 的模型来自 10，不能绕过风格 LoRA 或 InstantID。

## 3. 当前模型目录

安装后位于 `.runtime/ComfyUI/models/`：

| 相对路径 | 用途 |
| --- | --- |
| `checkpoints/sd_xl_base_1.0.safetensors` | 官方 SDXL Base 1.0 |
| `instantid/ip-adapter.bin` | InstantID 身份条件 |
| `controlnet/instantid/diffusion_pytorch_model.safetensors` | InstantID 关键点 ControlNet |
| `instantid/antelopev2.zip` | 锁定的人脸模型归档及 hash 证据 |
| `insightface/models/antelopev2/` | 安装脚本安全解压后的五个 ONNX |
| `loras/lcm-lora-sdxl.safetensors` | Turbo LCM-LoRA |
| `loras/lah-cyberpunk-sdxl.safetensors` | 赛博朋克，初始强度 1.0 |
| `loras/anime-sdxl.safetensors` | 动漫，初始强度 2.0 |
| `loras/chinese-ink-sdxl.safetensors` | 水墨，初始强度 0.8 |
| `loras/oil-painting-sdxl.safetensors` | 油画，初始强度 2.0 |

精确 revision、大小和 SHA256 以 `config/models.lock.json` 为准。旧 RealVisXL、PhotoMaker V2 和 buffalo_l 已改为可选回滚项，默认启动不校验也不加载。

## 4. 人脸策略

1. 节点 3 加载 antelopev2，并同时供 InstantID 与两个守卫使用。
2. 节点 16 在身份提取前检查全部输入人脸：无脸返回 `NO_FACE_DETECTED`，多脸返回 `MULTIPLE_FACES_DETECTED`。
3. 节点 15 使用人脸 embedding 与关键点 ControlNet，不会自行把多脸输入当作合法输入。
4. 节点 14 在 SaveImage 前重新检测：输出无脸返回 `OUTPUT_NO_FACE_DETECTED`，多脸返回 `OUTPUT_MULTIPLE_FACES_DETECTED`。
5. 检测与识别只用于人脸区域、关键点和身份特征，不判断姓名。动漫、水墨等输出可能出现检测误报/漏报，必须保留真实样本验收，不得绕过守卫。

## 5. API 提交与结果

| 步骤 | API | 用途 |
| --- | --- | --- |
| 健康检查 | `GET /system_stats` | ComfyUI 状态 |
| 上传 | `POST /upload/image` | 上传规范化输入 |
| 提交 | `POST /prompt` | 提交 `{"prompt": graph, "client_id": "..."}` |
| 查询 | `GET /history/{prompt_id}` | 状态、错误与 SaveImage 输出 |
| 取消 | `POST /queue`、`POST /interrupt` | 删除排队项并中断 |
| 下载 | `GET /view` | 按 filename/subfolder/type 下载并完整解码 |

history 的 `prompt` 数组结构为 `[number, prompt_id, prompt_graph, extra_data, outputs_to_execute]`，工作流图位于索引 2。Worker 修改节点 1 的 checkpoint、节点 3 的 provider、节点 5 的图像、节点 6/7 的正负提示、节点 13 的风格 LoRA、节点 15 的身份权重、节点 9 的采样参数和节点 12 的输出前缀。

手工诊断最新任务可运行：

```bash
python scripts/inspect_latest_task.py
```

`WORKFLOW_CHECKS` 会检查风格 LoRA 是否进入 InstantID/采样链以及输出守卫是否存在。

## 6. 安装、UI 与错误排查

- 更新模型或 vendor 节点后必须重新运行安装脚本，再受管重启；不能只刷新网页。
- UI 可导入 `workflows/standard_ui.json` 或 `turbo_ui.json`，但 API JSON 与 UI JSON 不可混用。ComfyUI 仅通过实际 `AIGC_COMFY_PORT` 的回环地址访问，不向公网开放。
- `InstantID... node type not found`：确认 `ComfyUI_InstantID` 已复制并查看 `comfy.log` 的导入错误。
- 模型不存在：运行 `bash scripts/download_models.sh`，同时会校验 hash 并安全解压 antelopev2；不得关闭校验。
- 输出无脸/多脸：保留失败样本，检查风格检测兼容性与提示词，不绕过节点 14。
- OOM：保持 1024×1024、batch 1、单活跃任务和物理 GPU 1。InstantID 比 PhotoMaker 增加 ControlNet/IP-Adapter，显存必须重新实测。
- 水印：InstantID 上游训练数据可能带水印，当前保持产品要求的 1024×1024，记录真实结果，不通过改分辨率掩盖问题。

## 7. A/B 验收

固定同一输入和 seed，分别比较四种风格的 Standard/Turbo。记录风格辨识度、身份相似度、输出人脸数量、检测误拒、端到端耗时和 GPU 1 峰值。初始强度不是最优承诺；一次只改一个变量。未经服务器真实 A/B，不得宣称 InstantID 已优于旧 PhotoMaker 路线。
