# 四风格 A+B 优化设计

> **后续路线替换（用户明确批准）**：RealVisXL + PhotoMaker 的多轮 A/B 未达到四风格目标，当前默认直接改为官方 SDXL Base 1.0 + InstantID。本文其余内容保留为旧路线的选型与调参记录；当前实现以源码、模型锁和 COMFYUI_WORKFLOWS.md 为准。

## 目标

在保留 RealVisXL V5、完整 PhotoMaker V2 身份链、1024×1024、物理 GPU 1 和单活跃任务的前提下，解决 Standard 下四风格区分度弱以及输出出现多张人脸的问题。Turbo 使用相同风格链并单独验收。

## 已确认的模型与初始参数

| 风格 | 仓库 / 文件 | 触发词 | 初始强度 |
| --- | --- | --- | --- |
| cyberpunk | `Muapi/lah-cyberpunk-sdxl` / `lah-cyberpunk-sdxl.safetensors` | `cyberpunk style` | 1.0 |
| anime | `ntc-ai/SDXL-LoRA-slider.anime` / `anime.safetensors` | `anime` | 2.0 |
| ink_wash | `ming-yang/sdxl_chinese_ink_lora` / `Chinese_Ink_Painting_Lora_SDXL.safetensors` | `Chinese Ink` | 0.8 |
| oil_painting | `ntc-ai/SDXL-LoRA-slider.oil-painting` / `oil painting.safetensors` | `oil painting` | 2.0 |

revision、大小与 SHA256 锁定在 `config/models.lock.json`。许可标记不等于商业许可清关，第三方文档保留复核要求。

第二轮实测修订：原 `issaccyj/lora-sdxl-cyberpunk` 在实际工作流已正确加载的前提下，多个 seed 仍只改变背景且没有赛博朋克元素，因此判定模型选型不适合当前人像链并替换为偏写实霓虹人像的 Muapi 候选。旧文件可保留用于回滚，但不再进入运行清单。

## 生成链

Standard：PhotoMaker UNet LoRA → 当前风格的 `LoraLoaderModelOnly` → KSampler。

Turbo：PhotoMaker UNet LoRA → 当前风格 LoRA → LCM-LoRA → LCM KSampler。

每次只加载当前风格，不叠加四套风格。风格 LoRA 不修改 CLIP 身份编码路径。`StylePreset` 增加 `lora_filename` 与 `lora_strength`，预热使用 cyberpunk 风格链以覆盖真实风格加载。

## 提示词与输出策略

提示词明确 `solo, one person, single subject`，并强化各风格的介质、色彩、材质、光影和互斥负向条件。所有负面提示词加入 `multiple people, two people, twins, duplicate person, extra face`。

ComfyUI 在 VAEDecode 与 SaveImage 之间增加 `PhotoMakerSingleFaceOutputGuard`：使用已加载的 InsightFace 对输出执行一次分析，只有恰好一张脸才允许保存。输出无脸/多脸分别映射为 `OUTPUT_NO_FACE_DETECTED` 与 `OUTPUT_MULTIPLE_FACES_DETECTED`，不自动重试，不交付失败图片。

## 验收与边界

先 Standard 后 Turbo，同一输入和 seed 配对比较修改前后结果。四风格分别评价风格辨识度、身份保持、输出人脸数量和耗时；增加多个 seed，不以单张好图宣布通过。动漫和水墨可能出现检测器漏检/误报，必须记录并调整，不能绕过守卫。初始 LoRA 强度不是最优承诺，A/B 后才冻结。保持现有 30 秒性能口径，新增风格后重新测量。
