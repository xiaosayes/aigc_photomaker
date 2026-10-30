# 第三方代码、模型与数据说明

本清单区分**代码许可证、模型权重/使用条款、图片/人脸数据条款**；一类授权不能自动覆盖另一类。本文不是法律意见，也不声明本项目或任何组合已取得商业使用清关。分发或商用前必须逐项打开权威来源，按交付时的 commit/revision 保存许可证原文并复核；不确定时不猜测许可证。

## 1. 代码

| 组件 | 本项目使用版本/位置 | 许可与交付说明 | 权威来源/发布前核验 |
| --- | --- | --- | --- |
| ComfyUI | `63be2f9b82ad0dac88a7ceaa28a9cf3154125a18`（v0.42.1） | **GPL-3.0**。在线/离线均提供对应完整源码与 commit 证据；修改、再分发和组合方式须由发布方履行 GPL 义务，不能只给二进制或模型目录。 | [上游仓库及 LICENSE](https://github.com/Comfy-Org/ComfyUI/tree/63be2f9b82ad0dac88a7ceaa28a9cf3154125a18)；核验 `offline/comfyui/ComfyUI.tar.gz`、`.aigc-commit` 和 `offline/SOURCE_COMMITS`。 |
| ComfyUI PhotoMaker Plus | `459f5c009e8bc3fe76bd63a7f302ddf3406c7062`；完整交付源码在 `vendor/ComfyUI-PhotoMaker-Plus/` | **GPL-3.0**。随包保留 LICENSE、README 和完整修改后源码（包括本项目修改，如有）；安装时复制到 ComfyUI custom_nodes。分发必须保留相应源码和许可，不以模型许可代替代码许可。 | [固定上游 commit](https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus/tree/459f5c009e8bc3fe76bd63a7f302ddf3406c7062) 与随包 `vendor/ComfyUI-PhotoMaker-Plus/LICENSE`；发布前比对 commit、修改和完整源码。 |
| cubiq ComfyUI InstantID（含本项目的通用/FLUX 人脸守卫） | `72495e806bc2ab9c41581e15ccaa1bcf83c477e8`；源码在 `vendor/ComfyUI_InstantID/` | **Apache-2.0**。上游已进入 maintenance-only，本项目另加输入/输出单人脸守卫；固定 commit 的 LICENSE、修改后源码和兼容性证据须一并保留。当前 FLUX 分支只复用人脸检测与守卫，不加载 InstantID 身份权重。 | [固定上游 commit](https://github.com/cubiq/ComfyUI_InstantID/tree/72495e806bc2ab9c41581e15ccaa1bcf83c477e8)与随包 `.aigc-commit`；核验 ComfyUI 0.42.1 导入和真实执行。 |
| PhotoMaker V2 官方代码/模型仓库 | `TencentARC/PhotoMaker-V2`，权重 revision 见 `config/models.lock.json` | 官方代码许可与 V2 权重/模型卡条款须分别核验。这里不把 PhotoMaker Plus 的 GPL-3.0 错写成 TencentARC 官方代码或权重的许可。 | [PhotoMaker V2 官方仓库](https://huggingface.co/TencentARC/PhotoMaker-V2)及其中 LICENSE/模型卡；在锁定 revision 保存原文并确认分发、商用和归属要求。 |
| InsightFace Python 代码 | `insightface==0.7.3` | InsightFace 上游对代码与预训练模型作不同声明：代码许可不能扩展为模型商用授权；其模型 zoo 常带非商业研究限制，必须逐模型确认。 | [InsightFace 上游](https://github.com/deepinsight/insightface)的 LICENSE、README/model zoo；核验 0.7.3 包内许可证及实际下载模型条款。 |
| Python 直接及传递依赖 | `requirements/app.txt`、`requirements/ml.txt`、ComfyUI requirements、PyTorch/vision、Conda/pip 包 | FastAPI、Pydantic、SQLAlchemy、HTTPX、Pillow、NumPy、ONNX/ONNX Runtime、pytest 等各保留其适用许可证、版权和 NOTICE；不能用一个总许可证覆盖全部。PyTorch CUDA 运行库及 Conda 包也需按实际包核验。 | 以锁定环境 `pip show`/`pip inspect`、包内 LICENSE/NOTICE 和上游项目为证据；发布前生成实际直接+传递依赖清单，核对离线 wheel/conda 包内通知是否保留。 |

本项目自身代码的许可不能由本文件推定；若需对外发布，应由权利人另行明确项目许可证，并同时遵守上述第三方条款。

## 2. 模型与权重

模型的准确文件、revision、SHA256 和大小以 `config/models.lock.json` 为准。以下来源链接用于核查，不表示链接最新版就是锁定版。

| 模型/数据组件 | 用途 | 许可状态与强制核验 |
| --- | --- | --- |
| [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) | 可选回滚的 SDXL 写实 checkpoint | RealVisXL 是模型权重，不因其 SDXL 架构而自动继承一个可随意商用的结论。必须在锁定 revision 核对模型卡 license、使用限制、分发与署名；若页面不明确，标记待核实并阻塞相应发布/商用，不自行命名许可。可选 V4 同样单独复核。 |
| [black-forest-labs/FLUX.2-klein-base-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) | 可选回滚 Base FP8，默认不加载 | 模型卡标记 **Apache-2.0**。按锁定 revision 保存 LICENSE.md 与模型卡，复核商标、使用政策、输入人脸和生成内容责任；官方硬件/时延描述不等于本项目验收值。 |
| [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) | `feature/flux` 的 Standard/Turbo Distilled FP8 | 模型卡标记 **Apache-2.0**。按锁定 revision 保存许可与模型卡；蒸馏路线与 Base 分别验收，不把官方亚秒级描述当作本机结果。 |
| [Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) | Qwen3 文本编码器与 FLUX.2 VAE | 仓库标记 **Apache-2.0**。固定 Qwen3/VAE 的仓库 revision 和 hash，核对上游组件许可、重新分发和 NOTICE，不以 ComfyUI 打包许可替代全部模型条款。 |
| [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 可选回滚的官方 SDXL Base 1.0 | 核对锁定 revision 的 SDXL 模型卡、Community/使用许可及可接受使用条款。代码许可与 SDXL 权重条款不可混同；未核验不得声称商业可用。 |
| [TencentARC/PhotoMaker-V2](https://huggingface.co/TencentARC/PhotoMaker-V2) | 可选回滚的身份编码和 UNet 身份 LoRA | 核对 V2 官方权重及模型卡；不得用 V1、ComfyUI 节点或 Plus 节点条款替代 V2 权重条款。 |
| [InstantX/InstantID](https://huggingface.co/InstantX/InstantID) | 当前 InstantID IP-Adapter 与 ControlNet | 仓库标记 **Apache-2.0**，但仍须按锁定 revision 保存模型卡/LICENSE，分别复核权重、示例数据和与 SDXL/风格 LoRA 组合后的条款。 |
| [MonsterMMORPG/tools](https://huggingface.co/MonsterMMORPG/tools) 中的 `antelopev2.zip` | 当前 InstantID 人脸检测、关键点和 embedding | 镜像不自动改变 InsightFace 原模型条款；必须追溯 antelopev2 原始许可、研究/非商业限制、再分发条件和人脸数据义务，不把可下载视为商业清关。 |
| [immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l) 中的 detection/recognition ONNX | 可选回滚的人脸检测与识别特征 | 这是重新分发来源，不自动改变原 InsightFace 模型权利。必须同时追溯 InsightFace buffalo_l 原模型条款、该镜像的说明和 ONNX 文件来源；尤其核验研究/非商业限制、再分发与人脸数据处理条件。不确定则不得作商业许可声明。 |
| [latent-consistency/lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl) | Turbo LCM-LoRA SDXL | LCM 代码、LCM-LoRA 权重和所结合的基础模型分别核验。以锁定 revision 模型卡/LICENSE 为权威，保留归属和 NOTICE（如适用）；不凭“LoRA 很小”推定无限制。 |
| [Muapi/lah-cyberpunk-sdxl](https://huggingface.co/Muapi/lah-cyberpunk-sdxl) | 赛博朋克风格 LoRA，触发词 `cyberpunk style` | 模型卡标记 **OpenRAIL++**；按锁定 revision 保存模型卡和适用条款，复核与 RealVisXL/PhotoMaker 组合、再分发、署名和使用限制，不把标记扩写为商业清关。此前偏城市景观的 `issaccyj/lora-sdxl-cyberpunk` 因多 seed 人像风格无效已退出当前运行链。 |
| [ntc-ai/SDXL-LoRA-slider.anime](https://huggingface.co/ntc-ai/SDXL-LoRA-slider.anime) | 动漫风格 LoRA，触发词 `anime` | 模型卡标记 **MIT**；保存对应 LICENSE/版权与模型卡，仍须复核生成内容、组合模型和输入肖像责任。 |
| [ming-yang/sdxl_chinese_ink_lora](https://huggingface.co/ming-yang/sdxl_chinese_ink_lora) | 中国水墨风格 LoRA，触发词 `Chinese Ink` | 模型卡标记 **CreativeML OpenRAIL-M**；核对锁定 revision 的使用限制、分发和归属，不能与 MIT 风格 LoRA 混同。 |
| [ntc-ai/SDXL-LoRA-slider.oil-painting](https://huggingface.co/ntc-ai/SDXL-LoRA-slider.oil-painting) | 油画风格 LoRA，触发词 `oil painting` | 模型卡标记 **MIT**；保存对应 LICENSE/版权与模型卡，仍须复核组合模型、输出和输入数据条款。 |

生成结果还可能涉及输入者、被摄者、风格模板和模型使用条款。模型可下载不等于可再分发，可研究不等于可商用，可商用也不等于免除隐私、肖像或内容责任。

## 3. 预热图片与数据 provenance

`assets/warmup-portrait.png` 是 scikit-image 的 **`astronaut.png`** 样本：NASA 宇航员 Eileen Collins 的真实单人脸照片，用于 Standard/Turbo 的真实身份链预热，不是无脸随机图。未在本任务重新下载。

- 固定上游来源：[scikit-image v0.25.2 astronaut.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png)。
- 项目内来源说明：`assets/README.md`；SHA256：`88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`。
- scikit-image 将 astronaut 描述为 NASA 图片/美国公共领域；scikit-image **代码**的 BSD-3-Clause 与图片**数据**状态是两件事。
- NASA 图片的一般公共领域说明不消除署名、NASA 标识、背书暗示、被摄者人格/肖像等限制。发布前须复核 [NASA 媒体使用指南](https://www.nasa.gov/nasa-brand-center/images-and-media/) 和 scikit-image 数据说明，不得把“通常公共领域”扩写为所有司法辖区、所有用途均无条件许可。

用户上传的人脸不是第三方素材许可所覆盖的数据。部署方须取得合法授权，限定用途和保留期，保护输入、输出、日志及备份，并尊重删除请求；不得把用户照片用于额外训练或公开展示，除非另有明确合法依据。

## 4. 发布核验清单

- [ ] 固定 ComfyUI、PhotoMaker Plus、InstantID commit，许可证原文和对应完整修改后源码均随发布交付。
- [ ] 每个模型按锁定 revision 保存模型卡、LICENSE/条款和 hash；RealVisXL、SDXL、PhotoMaker V2、InstantID、InsightFace/antelopev2/buffalo_l、LCM 无“待核实”项被误写为许可已确认。
- [ ] 实际 Python/Conda/CUDA 直接和传递依赖的版权、LICENSE、NOTICE 已检查并保留。
- [ ] astronaut 来源/hash 与 `assets/README.md` 一致，NASA/肖像限制已审查；用户人脸数据另有授权和治理。
- [ ] 发布包不包含 API Key、私人输入/输出或未审查日志；法务/权利人未完成商业审查前，明确写“本项目不声明商业使用清关”。
