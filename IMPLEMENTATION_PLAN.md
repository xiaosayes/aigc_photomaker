# AIGC PhotoMaker Implementation Plan

> **2026-10-28 当前路线修订**：默认工作流已直接替换为官方 SDXL Base 1.0 + InstantID + 风格 LoRA，使用 antelopev2、InstantID IP-Adapter/ControlNet 和输入/输出单人脸守卫。本文后续 PhotoMaker 任务与代码块属于历史实施记录，不得用来覆盖当前 `src/aigc_photomaker/workflow.py` 和 `config/models.lock.json`。当前精确实现以源码、工作流 JSON、COMFYUI_WORKFLOWS.md 和 DESIGN.md 为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个可通过浏览器上传人物照片、调用真实 ComfyUI Standard/Turbo 工作流并在一台 RTX 3090 服务器上部署验收的完整项目及 ZIP。

**Architecture:** FastAPI 负责鉴权、上传和任务 API，SQLite WAL 负责持久化，独立单并发 Worker 通过 ComfyUI HTTP/WebSocket API 编排生成。当前 ComfyUI 使用官方 SDXL Base 1.0 + InstantID + 风格 LoRA，Turbo 增加 LCM-LoRA；应用、Worker、ComfyUI 分进程运行，所有 GPU 进程只看到物理卡 1。

**Tech Stack:** Python 3.11、FastAPI、Pydantic v2、SQLAlchemy 2、SQLite、HTTPX、Pillow、WebSocket、ComfyUI v0.42.1、PyTorch 2.9.1+cu126、InsightFace、原生 JavaScript/HTML/CSS、Shell、pytest。

## Global Constraints

- 已批准设计是唯一范围依据：`DESIGN.md`；本文根目录存放是需求方“文档放在根目录”的明确要求。
- 本地只开发及静态检查；不启动 Conda 环境、模型、ComfyUI、Worker 或 Web 服务。
- 所有需要依赖和 GPU 的 pytest、集成测试及验收命令均在 Ubuntu 22.04 x86_64 GPU 服务器执行。
- Conda 环境名固定为 `aigc-photomaker`，Python 固定为 3.11。
- GPU 固定为物理卡 1：`CUDA_VISIBLE_DEVICES=1`，进程内逻辑设备为 `cuda:0`。
- ComfyUI 固定提交 `63be2f9b82ad0dac88a7ceaa28a9cf3154125a18`（v0.42.1）。
- InstantID 节点固定提交 `72495e806bc2ab9c41581e15ccaa1bcf83c477e8`，修改后的完整源码和 Apache-2.0 许可证随包交付；历史 PhotoMaker Plus 仅用于回滚。
- 当前默认模型为官方 SDXL Base 1.0、InstantID IP-Adapter/ControlNet、antelopev2、四个风格 LoRA 和 LCM-LoRA SDXL。
- 输出固定为 1024×1024 PNG、batch 1；Standard 从 25 步开始，Turbo 从 8 步开始。
- 正常口径为服务完成预热、无排队、单并发，从点击到结果显示的端到端时延；Standard 验收目标为 P95 ≤ 30 秒。
- 未取得服务器实测报告前，不得宣称 30 秒、显存或画质已经验收通过。
- GitHub 不可直连时必须可用镜像或离线包；离线模式不得隐式联网下载模型。
- 应用 ZIP 不包含模型权重、PyTorch wheel、`.env`、运行日志、生成结果或 `.git`。
- 所有动态文件路径必须限制在项目的数据目录内；API Key 使用常量时间比较，日志不得输出秘密。
- 每个任务先写测试，再写最小实现；行为测试运行步骤标为服务器执行，本地只做语法、JSON、Shell 和 Git 静态检查。
- 源码文本、HTML 属性、Shell 标志和文档关键字检查只属于静态检查，不计为功能测试；功能证据必须来自 API、安装启动、真实浏览器或 GPU 行为。

## File Structure

```text
pyproject.toml
requirements/app.txt
requirements/ml.txt
config/models.lock.json
config/versions.lock.env
config/styles.json
src/aigc_photomaker/
  __init__.py
  config.py                 环境配置与路径
  errors.py                 统一错误及错误码
  logging_config.py         JSON 日志
  model_manager.py          模型下载、离线导入、大小和 SHA256 校验
  styles.py                 风格模板类型与加载
  workflow.py               Standard/Turbo 图定义、参数注入、UI JSON 生成
  images.py                 图片真实解码、EXIF、像素与尺寸规范化
  security.py               API Key 和限流
  db.py                     SQLite engine、PRAGMA 和建表
  models.py                 Task ORM 和 TaskRecord
  repository.py             原子认领、fencing token、状态与取消
  comfy.py                  ComfyUI HTTP/WebSocket 客户端
  task_service.py           上传事务、读取结果和取消编排
  worker.py                 租约 Worker、恢复、心跳、清理
  warmup.py                 Standard/Turbo 启动预热
  retention.py              到期数据清理
  app.py                    FastAPI 路由、健康和静态站点
  smoke.py                  真实 GPU 冒烟测试
  benchmark.py              端到端性能、显存和报告
web/index.html
web/styles.css
web/app.js
workflows/standard_api.json
workflows/turbo_api.json
workflows/standard_ui.json
workflows/turbo_ui.json
vendor/ComfyUI-PhotoMaker-Plus/
assets/warmup-portrait.png
assets/README.md
scripts/build_workflows.py
scripts/static_check.py
scripts/lib.sh
scripts/install.sh
scripts/download_models.sh
scripts/start.sh
scripts/stop.sh
scripts/status.sh
scripts/smoke_test.sh
scripts/benchmark.sh
scripts/prepare_offline_bundle.sh
deploy/aigc-photomaker.service
tests/
README.md
DESIGN.md
IMPLEMENTATION_PLAN.md
DEPLOYMENT.md
MODEL_GPU_EVALUATION.md
COMFYUI_WORKFLOWS.md
ACCEPTANCE.md
ROADMAP.md
PROMPTS.md
THIRD_PARTY_NOTICES.md
```

---

### Task 1: 项目基座、配置、错误和日志

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `requirements/app.txt`
- Create: `src/aigc_photomaker/__init__.py`, `config.py`, `errors.py`, `logging_config.py`
- Test: `tests/test_config.py`, `tests/test_errors.py`

**Interfaces:**
- Produces: `Settings`, `AppError(status_code, code, message, retryable=False)`, `configure_logging(component: str) -> None`。
- `Settings` 使用 `AIGC_` 环境前缀，所有相对路径以项目根目录解析。

- [ ] **Step 1: 写失败的配置和错误测试**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from aigc_photomaker.config import Settings

def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_defaults_match_runtime_contract(monkeypatch):
    monkeypatch.setenv("AIGC_API_KEY", "a" * 32)
    settings = Settings(_env_file=None)
    assert settings.app_port == 8000
    assert settings.comfy_port == 8188
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.worker_lease_seconds == 60
    assert settings.standard_steps == 25
    assert settings.turbo_steps == 8
```

```python
# tests/test_errors.py
from aigc_photomaker.errors import AppError

def test_error_payload_does_not_include_stack():
    error = AppError(400, "INVALID_IMAGE", "图片无法解码", False)
    assert error.to_dict() == {
        "code": "INVALID_IMAGE",
        "message": "图片无法解码",
        "retryable": False,
    }
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_config.py tests/test_errors.py -v`
Expected: 因 `aigc_photomaker.config`、`errors` 尚不存在而 FAIL。本地不执行依赖测试。

- [ ] **Step 3: 写最小项目配置和实现**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "aigc-photomaker"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
```

```python
# src/aigc_photomaker/config.py
from functools import lru_cache
from pathlib import Path
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIGC_", env_file=".env", extra="ignore")
    api_key: SecretStr
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188
    comfy_dir: Path = Path(".runtime/ComfyUI")
    data_dir: Path = Path("data")
    runtime_dir: Path = Path(".runtime/app")
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_input_side: int = 2048
    worker_lease_seconds: int = 60
    execution_timeout_seconds: int = 120
    retention_days: int = 7
    generation_rate_per_minute: int = 20
    standard_steps: int = 25
    standard_cfg: float = 5.0
    standard_sampler: str = "dpmpp_2m"
    standard_scheduler: str = "karras"
    turbo_steps: int = 8
    turbo_cfg: float = 1.8
    checkpoint_filename: str = "RealVisXL_V5.0_fp16.safetensors"
    face_provider: str = "CPU"

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 24:
            raise ValueError("AIGC_API_KEY must contain at least 24 characters")
        return value

    @property
    def comfy_url(self) -> str:
        return f"http://{self.comfy_host}:{self.comfy_port}"

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`AppError` 调用 `super().__init__(f"{code}: {message}")`，使异常匹配和日志包含错误码；`to_dict()` 只返回测试中的三个键。`configure_logging()` 使用 `python-json-logger`，字段为 `timestamp`、`level`、`component`、`event`、`task_id`、`duration_ms`。`requirements/app.txt` 写入以下直接依赖版本，离线包再对全部传递依赖和 wheel 记录 SHA256：

```text
fastapi==0.115.12
uvicorn==0.34.2
python-multipart==0.0.20
pydantic==2.11.4
pydantic-settings==2.9.1
SQLAlchemy==2.0.40
httpx==0.28.1
Pillow==11.2.1
websocket-client==1.8.0
python-json-logger==3.3.0
pytest==8.3.5
```

```gitignore
.env
.runtime/
data/
offline/
reports/
__pycache__/
*.py[cod]
*.zip
.pytest_cache/
```

- [ ] **Step 4: 在服务器验证通过，本地做静态检查**

Run on server: `pytest tests/test_config.py tests/test_errors.py -v`
Expected: 2 个配置测试和 1 个错误测试 PASS。
Run locally: `python -m compileall -q src tests`
Expected: 无语法错误，不导入或启动应用。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml .gitignore requirements src tests
 git commit -m "feat: add application foundation"
```

---

### Task 2: 不可变模型锁、ComfyUI 节点和单人脸策略

**Files:**
- Create: `config/models.lock.json`, `config/versions.lock.env`, `requirements/ml.txt`
- Create: `src/aigc_photomaker/model_manager.py`
- Create: `vendor/ComfyUI-PhotoMaker-Plus/` 完整固定版本源码
- Create: `vendor/ComfyUI-PhotoMaker-Plus/face_policy.py`
- Modify: `vendor/ComfyUI-PhotoMaker-Plus/photomaker.py`, `insightface_package.py`
- Test: `tests/test_model_manifest.py`, `tests/test_face_policy.py`, `tests/test_model_manager.py`

**Interfaces:**
- Produces: `select_single_face(faces: list[dict]) -> dict`，无人脸抛 `ValueError("NO_FACE_DETECTED: upload one clear, visible face")`，多人脸抛 `ValueError("MULTIPLE_FACES_DETECTED: upload an image containing exactly one face")`。
- Produces: `ModelStatus(key: str, path: Path, ok: bool, error: str | None)`、`ModelManager(manifest_path: Path, models_root: Path, hf_endpoint: str)`，以及 `download(include_optional=False, offline_dir=None)`、`verify(full_hash: bool, include_optional=False) -> list[ModelStatus]`。
- 模型锁每项包含 `key`、`required`、`repo`、`revision`、`filename`、`relative_path`、`size`、`sha256`。

- [ ] **Step 1: 写失败的清单、单人脸和离线模型测试**

```python
# tests/test_face_policy.py
import importlib.util
from pathlib import Path
import pytest

path = Path("vendor/ComfyUI-PhotoMaker-Plus/face_policy.py")
spec = importlib.util.spec_from_file_location("face_policy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_single_face_is_selected():
    face = {"embedding": [0.1, 0.2]}
    assert module.select_single_face([face]) is face

@pytest.mark.parametrize("faces,code", [([], "NO_FACE_DETECTED"), ([{}, {}], "MULTIPLE_FACES_DETECTED")])
def test_invalid_face_count(faces, code):
    with pytest.raises(ValueError, match=code):
        module.select_single_face(faces)
```

```python
# tests/test_model_manifest.py
import json
from pathlib import Path

def test_manifest_has_verified_defaults():
    models = json.loads(Path("config/models.lock.json").read_text())["models"]
    defaults = {m["key"]: m for m in models if m["required"]}
    assert set(defaults) == {"realvisxl_v5", "photomaker_v2", "lcm_lora_sdxl", "face_detection", "face_recognition"}
    assert all(len(m["sha256"]) == 64 and m["size"] > 0 for m in defaults.values())
```

```python
# tests/test_model_manager.py
import hashlib
import json
from aigc_photomaker.model_manager import ModelManager

def test_offline_import_verifies_hash(tmp_path):
    payload = b"model"
    relative = "photomaker/x.bin"
    source = tmp_path / "offline" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models":[{
        "key":"x", "required":True, "repo":"local/test", "revision":"1",
        "filename":"x.bin", "relative_path":relative, "size":len(payload),
        "sha256":hashlib.sha256(payload).hexdigest()
    }]}))
    manager = ModelManager(manifest, tmp_path / "models", "http://network-must-not-be-used")
    manager.download(offline_dir=tmp_path / "offline")
    assert (tmp_path / "models" / relative).read_bytes() == payload
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_model_manifest.py tests/test_face_policy.py tests/test_model_manager.py -v`
Expected: 清单、`face_policy.py` 和模型管理模块不存在，FAIL。

- [ ] **Step 3: 固定源码并写锁文件**

```bash
git clone https://ghproxy.net/https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus.git vendor/ComfyUI-PhotoMaker-Plus
git -C vendor/ComfyUI-PhotoMaker-Plus checkout 459f5c009e8bc3fe76bd63a7f302ddf3406c7062
rm -rf vendor/ComfyUI-PhotoMaker-Plus/.git
```

```dotenv
# config/versions.lock.env
COMFYUI_REPO=https://github.com/Comfy-Org/ComfyUI.git
COMFYUI_VERSION=v0.42.1
COMFYUI_COMMIT=63be2f9b82ad0dac88a7ceaa28a9cf3154125a18
PHOTOMAKER_PLUS_COMMIT=459f5c009e8bc3fe76bd63a7f302ddf3406c7062
TORCH_VERSION=2.9.1
TORCHVISION_VERSION=0.24.1
PYTORCH_CUDA=cu126
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
```

`models.lock.json` 使用以下七项，URL 由 `HF_ENDPOINT + repo + /resolve/ + revision + / + filename` 组成：

```json
{"models":[
 {"key":"realvisxl_v5","required":true,"repo":"SG161222/RealVisXL_V5.0","revision":"ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc","filename":"RealVisXL_V5.0_fp16.safetensors","relative_path":"checkpoints/RealVisXL_V5.0_fp16.safetensors","size":6938065488,"sha256":"6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80"},
 {"key":"photomaker_v2","required":true,"repo":"TencentARC/PhotoMaker-V2","revision":"f5a1e5155dc02166253fa7e29d13519f5ba22eac","filename":"photomaker-v2.bin","relative_path":"photomaker/photomaker-v2.bin","size":1799993138,"sha256":"0eeac57d7e09b95cee670a18e0d939ff303dd6f76d317c950bf584c6c9bdb8f5"},
 {"key":"lcm_lora_sdxl","required":true,"repo":"latent-consistency/lcm-lora-sdxl","revision":"a18548dd4956b174ec5b0d78d340c8dae0a129cd","filename":"pytorch_lora_weights.safetensors","relative_path":"loras/lcm-lora-sdxl.safetensors","size":393855224,"sha256":"a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141"},
 {"key":"face_detection","required":true,"repo":"immich-app/buffalo_l","revision":"d09715916a0778919a770c343533641e250b8699","filename":"detection/model.onnx","relative_path":"insightface/models/buffalo_l/det_10g.onnx","size":16923827,"sha256":"5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"},
 {"key":"face_recognition","required":true,"repo":"immich-app/buffalo_l","revision":"d09715916a0778919a770c343533641e250b8699","filename":"recognition/model.onnx","relative_path":"insightface/models/buffalo_l/w600k_r50.onnx","size":174383860,"sha256":"4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"},
 {"key":"realvisxl_v4","required":false,"repo":"SG161222/RealVisXL_V4.0","revision":"26dfe44930964cd70d0a817b6d1cc945c130e38d","filename":"RealVisXL_V4.0.safetensors","relative_path":"checkpoints/RealVisXL_V4.0.safetensors","size":6938040706,"sha256":"912c9dc74f5855175c31a7993f863a043ac8dcc31732b324cd05d75cd7e16844"},
 {"key":"sdxl_base_1_0","required":false,"repo":"stabilityai/stable-diffusion-xl-base-1.0","revision":"462165984030d82259a11f4367a4eed129e94a7b","filename":"sd_xl_base_1.0.safetensors","relative_path":"checkpoints/sd_xl_base_1.0.safetensors","size":6938078334,"sha256":"31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"}
]}
```

- [ ] **Step 4: 写人脸策略并接入既有检测路径**

```python
# vendor/ComfyUI-PhotoMaker-Plus/face_policy.py
def select_single_face(faces):
    if not faces:
        raise ValueError("NO_FACE_DETECTED: upload one clear, visible face")
    if len(faces) != 1:
        raise ValueError("MULTIPLE_FACES_DETECTED: upload an image containing exactly one face")
    return faces[0]
```

在 `photomaker.py` 增加 `from .face_policy import select_single_face`，并在 V2 分支中将 `faces[0]["embedding"]` 改为 `select_single_face(faces)["embedding"]`，每张输入图片只调用一次 `analyze_faces`。在 `insightface_package.py` 创建 `FaceAnalysis` 时增加 `allowed_modules=["detection", "recognition"]`，保持 `name="buffalo_l"` 和可配置 Provider。`requirements/ml.txt` 固定 `numpy==1.26.4`、`Cython==3.0.12`、`onnx==1.17.0`、`insightface==0.7.3`、`onnxruntime-gpu==1.20.1`，不执行插件中会同时安装 CPU/GPU 两个 onnxruntime 分发包的原始 requirements。

- [ ] **Step 5: 写模型下载、校验和离线导入**

```python
# src/aigc_photomaker/model_manager.py 的纯函数与离线核心
import hashlib
import shutil

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def import_offline(self, model, offline_dir):
    source = offline_dir / model["relative_path"]
    target = self.models_root / model["relative_path"]
    self.verify_one(model, source, full_hash=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(source, temporary)
    self.verify_one(model, temporary, full_hash=True)
    temporary.replace(target)
```

`ModelManager` 在线路径构造 `${HF_ENDPOINT}/${repo}/resolve/${revision}/${filename}`，默认镜像为 `https://hf-mirror.com`；采用 8MB 分块、3 次重试、`.part`、大小/SHA256 双校验和 `os.replace`。`offline_dir` 存在时不创建任何网络 Client。quick verify 只查存在和大小，full verify 计算哈希；默认过滤 `required=true`，`include_optional=True` 才包含 V4/Base。

- [ ] **Step 6: 验证并提交**

Run on server: `pytest tests/test_model_manifest.py tests/test_face_policy.py tests/test_model_manager.py -v`
Expected: 全部 PASS。
Run locally: `python -m json.tool config/models.lock.json > /dev/null && git diff --check`
Expected: JSON 合法且无空白错误。

```bash
git add config requirements/ml.txt src/aigc_photomaker/model_manager.py vendor tests/test_model_manifest.py tests/test_face_policy.py tests/test_model_manager.py
 git commit -m "feat: lock models and enforce single-face input"
```

---

### Task 3: 风格模板和两套真实 ComfyUI 工作流

**Files:**
- Create: `config/styles.json`, `src/aigc_photomaker/styles.py`, `workflow.py`
- Create: `scripts/build_workflows.py`
- Create: `workflows/standard_api.json`, `turbo_api.json`, `standard_ui.json`, `turbo_ui.json`
- Test: `tests/test_styles.py`, `tests/test_workflow.py`

**Interfaces:**
- Produces: `StylePreset(key, label, description, positive, negative)`、`load_styles(path: Path) -> dict[str, StylePreset]`。
- Produces: `ComfyImageRef(name, subfolder="", type="input")`、`build_workflow(mode: str) -> dict`、`prepare_workflow(mode, image, style, seed, output_prefix, settings) -> dict`。

- [ ] **Step 1: 写失败的工作流语义测试**

```python
# tests/test_workflow.py
import json
from pathlib import Path
from aigc_photomaker.config import Settings
from aigc_photomaker.workflow import ComfyImageRef, build_workflow, prepare_workflow
from aigc_photomaker.styles import load_styles

def test_standard_contains_complete_identity_chain():
    graph = build_workflow("standard")
    assert graph["4"]["class_type"] == "PhotoMakerLoraLoaderPlus"
    assert graph["6"]["inputs"]["insightface_opt"] == ["3", 0]
    assert graph["9"]["inputs"]["model"] == ["4", 0]
    assert graph["9"]["inputs"]["steps"] == 25

def test_turbo_adds_lcm_without_changing_resolution():
    graph = build_workflow("turbo")
    assert graph["10"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["9"]["inputs"]["model"] == ["10", 0]
    assert graph["9"]["inputs"]["sampler_name"] == "lcm"
    assert graph["9"]["inputs"]["steps"] == 8
    assert graph["8"]["inputs"]["width"] == 1024
    assert graph["8"]["inputs"]["height"] == 1024

def test_parameter_injection_preserves_trigger():
    styles = load_styles(Path("config/styles.json"))
    settings = Settings(api_key="a" * 32, _env_file=None)
    graph = prepare_workflow("standard", ComfyImageRef("face.png"), styles["cyberpunk"], 42, "aigc/task", settings)
    assert graph["5"]["inputs"]["image"] == "face.png"
    assert graph["9"]["inputs"]["seed"] == 42
    assert graph["12"]["inputs"]["filename_prefix"] == "aigc/task"
    assert graph["6"]["inputs"]["text"].count(" img") == 1
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_styles.py tests/test_workflow.py -v`
Expected: 工作流和风格模块不存在，FAIL。

- [ ] **Step 3: 写明确的 API 图定义**

```python
# src/aigc_photomaker/workflow.py
from copy import deepcopy
from dataclasses import dataclass

@dataclass(frozen=True)
class ComfyImageRef:
    name: str
    subfolder: str = ""
    type: str = "input"

def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": class_type}}

def build_workflow(mode: str) -> dict:
    if mode not in {"standard", "turbo"}:
        raise ValueError("INVALID_MODE")
    graph = {
      "1": node("CheckpointLoaderSimple", ckpt_name="RealVisXL_V5.0_fp16.safetensors"),
      "2": node("PhotoMakerLoaderPlus", photomaker_model_name="photomaker-v2.bin"),
      "3": node("PhotoMakerInsightFaceLoader", provider="CPU"),
      "4": node("PhotoMakerLoraLoaderPlus", model=["1",0], photomaker=["2",0], lora_strength=1.0),
      "5": node("LoadImage", image="warmup-portrait.png"),
      "6": node("PhotoMakerEncodePlus", clip=["1",1], photomaker=["2",0], image=["5",0], insightface_opt=["3",0], trigger_word="img", text="portrait of a person img"),
      "7": node("CLIPTextEncode", clip=["1",1], text="low quality, deformed face, extra fingers, bad anatomy"),
      "8": node("EmptyLatentImage", width=1024, height=1024, batch_size=1),
      "9": node("KSampler", model=["4",0], positive=["6",0], negative=["7",0], latent_image=["8",0], seed=0, steps=25, cfg=5.0, sampler_name="dpmpp_2m", scheduler="karras", denoise=1.0),
      "11": node("VAEDecode", samples=["9",0], vae=["1",2]),
      "12": node("SaveImage", images=["11",0], filename_prefix="aigc-photomaker/output")
    }
    if mode == "turbo":
        graph["10"] = node("LoraLoaderModelOnly", model=["4",0], lora_name="lcm-lora-sdxl.safetensors", strength_model=1.0)
        graph["9"]["inputs"].update(model=["10",0], steps=8, cfg=1.8, sampler_name="lcm", scheduler="normal")
    return graph
```

`prepare_workflow()` 深拷贝图，设置节点 1 的 `ckpt_name`、节点 3 的 Provider、节点 5 的图片、节点 6/7 的正负提示词、节点 9 的 seed/steps/CFG/采样参数及节点 12 的输出前缀。子目录图片用 `subfolder/name`，拒绝非 `input` 类型引用。

`config/styles.json` 四个正向模板分别只出现一次 `person img`：赛博朋克使用 neon-lit rainy futuristic city/cinematic portrait，动漫使用 detailed 2D anime/clean line art，水墨使用 expressive Chinese ink wash/rice paper/negative space，油画使用 visible impasto brushwork/canvas texture/classical lighting。公共负向词包含 low quality、face asymmetry、deformed eyes、bad anatomy、extra fingers、watermark、text、logo，并加入各风格的反向约束。

- [ ] **Step 4: 生成 UI/API 两份 JSON 并验证**

`scripts/build_workflows.py` 将 API 图节点 ID 转为整数 UI 节点，链接转为 `[link_id, origin_id, origin_slot, target_id, target_slot, type]`，按依赖计算 `order`，并设置固定布局。标量控件顺序明确为：KSampler `[seed,"fixed",steps,cfg,sampler_name,scheduler,denoise]`，LoadImage `[image,"image"]`，其他节点按 API inputs 声明顺序。生成后比较四份提交文件，保证重复生成无差异。

Run locally: `python scripts/build_workflows.py --check`
Expected: 只读取和验证 JSON，PASS，不导入 ComfyUI。
Run on server: `pytest tests/test_styles.py tests/test_workflow.py -v`
Expected: 全部 PASS；另用 ComfyUI `/object_info` 核对所有 `class_type` 和输入名。

- [ ] **Step 5: 提交**

```bash
git add config/styles.json src/aigc_photomaker/styles.py src/aigc_photomaker/workflow.py scripts/build_workflows.py workflows tests/test_styles.py tests/test_workflow.py
 git commit -m "feat: add standard and turbo ComfyUI workflows"
```

---

### Task 4: 图片规范化、API Key 和生成限流

**Files:**
- Create: `src/aigc_photomaker/images.py`, `security.py`
- Test: `tests/test_images.py`, `tests/test_security.py`

**Interfaces:**
- Produces: `ImageMetadata(width, height, sha256, size_bytes)`。
- Produces: `normalize_image(source: Path, destination: Path, max_pixels: int, max_side: int) -> ImageMetadata`。
- Produces: `verify_api_key(provided: str, expected: SecretStr) -> bool`、`GenerationRateLimiter.allow(key: str, now: float) -> bool`、`GenerationRateLimiter.allow_or_raise(key: str, now: float) -> None`。

- [ ] **Step 1: 用 Pillow 生成测试夹具并写失败测试**

```python
# tests/test_images.py
from pathlib import Path
import pytest
from PIL import Image
from aigc_photomaker.errors import AppError
from aigc_photomaker.images import normalize_image

def test_exif_is_removed_and_rgb_png_is_written(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (3000, 1000), (20, 30, 40, 255)).save(source)
    target = tmp_path / "normalized.png"
    metadata = normalize_image(source, target, 20_000_000, 2048)
    with Image.open(target) as result:
        assert result.mode == "RGB"
        assert result.size == (2048, 683)
        assert not result.getexif()
    assert metadata.size_bytes == target.stat().st_size

def test_non_image_is_rejected(tmp_path):
    source = tmp_path / "fake.png"
    source.write_text("not an image")
    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)
```

安全测试覆盖正确/错误/空 API Key、常量时间比较调用，以及同一分钟第 21 次生成被拒绝、下一窗口恢复。

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_images.py tests/test_security.py -v`
Expected: 模块不存在，FAIL。

- [ ] **Step 3: 写图片和限流实现**

```python
# 图片实现的核心顺序
Image.MAX_IMAGE_PIXELS = max_pixels
with Image.open(source) as probe:
    probe.verify()
with Image.open(source) as opened:
    if getattr(opened, "is_animated", False):
        raise AppError(400, "INVALID_IMAGE", "不支持动图", False)
    image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.width < 128 or image.height < 128:
        raise AppError(400, "INVALID_IMAGE", "图片短边或长边小于 128 像素", False)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image.save(destination.with_suffix(".part"), format="PNG", compress_level=3)
destination.with_suffix(".part").replace(destination)
```

实际代码捕获 `UnidentifiedImageError`、`OSError`、`DecompressionBombError`，统一转成不暴露路径的 `AppError`。`destination.with_suffix(".part")` 必须在异常时删除；SHA256 用 1MB 分块计算。限流器以 60 秒固定窗口保存 `{key: (window_index, count)}`，使用 `threading.Lock`，并清理旧窗口；API Key 使用 `secrets.compare_digest`。

- [ ] **Step 4: 验证并提交**

Run on server: `pytest tests/test_images.py tests/test_security.py -v`
Expected: 全部 PASS，包括 20MP 炸弹边界、动图、损坏文件、尺寸缩小和限流窗口。
Run locally: `python -m compileall -q src tests`

```bash
git add src/aigc_photomaker/images.py src/aigc_photomaker/security.py tests/test_images.py tests/test_security.py
 git commit -m "feat: secure uploads and API access"
```

---

### Task 5: SQLite 任务模型、原子认领和崩溃恢复语义

**Files:**
- Create: `src/aigc_photomaker/db.py`, `models.py`, `repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Produces: `TaskRecord` dataclass，包含任务、路径、模式、种子、状态、prompt_id、claim_token、lease、cancel_requested、timings 和 error。
- Produces: `TaskRepository.create_schema/create/get/claim/heartbeat/mark_submitted/update_progress/succeed/fail/request_cancel/mark_cancelled`；测试辅助方法为 `force_lease_expiry(task_id)`。
- `create` 自动设置 `output_path="outputs/{task_id}.png"`。
- 所有运行态变更方法接收 `claim_token`，防止旧 Worker 在租约过期后覆盖新 Worker 的结果。

- [ ] **Step 1: 写失败的状态机和 fencing 测试**

```python
# tests/test_repository.py
from aigc_photomaker.repository import TaskRepository

def test_claim_is_atomic_and_recovers_expired_running_task(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    created = repo.create(task_id="t1", input_path="inputs/t1.png", style="cyberpunk", mode="standard", seed=42)
    first = repo.claim("worker-a", lease_seconds=1)
    assert first.status == "running"
    assert first.claim_token
    assert repo.claim("worker-b", lease_seconds=1) is None
    repo.force_lease_expiry("t1")
    recovered = repo.claim("worker-b", lease_seconds=60)
    assert recovered.task_id == "t1"
    assert recovered.claim_token != first.claim_token
    assert not repo.heartbeat("t1", first.claim_token, 60)

def test_cancel_queued_task(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    repo.create("t2", "inputs/t2.png", "anime", "turbo", 7)
    assert repo.request_cancel("t2").status == "cancelled"
    assert repo.claim("worker", 60) is None
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_repository.py -v`
Expected: Repository 不存在，FAIL。

- [ ] **Step 3: 写 SQLite 和 Repository 实现**

SQLite 连接设置 `check_same_thread=False`、`timeout=30`，每次连接执行：

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=30000;
PRAGMA foreign_keys=ON;
```

任务时间使用 UTC Unix 秒浮点数。认领采用唯一 token 和单个原子 UPDATE，避免依赖 SQLite 不支持的有效行锁：

```sql
UPDATE tasks
SET status='running', worker_id=:worker_id, claim_token=:token,
    lease_expires_at=:now + :lease, heartbeat_at=:now
WHERE id=(
  SELECT id FROM tasks
  WHERE status='queued' OR (status='running' AND lease_expires_at < :now)
  ORDER BY CASE status WHEN 'queued' THEN 0 ELSE 1 END, created_at
  LIMIT 1
)
```

UPDATE 后按 token SELECT；没有返回即表示没有可认领任务。`mark_submitted` 保存 prompt_id，`succeed` 保存相对输出路径和 timings，`fail` 保存结构化错误，`request_cancel` 对 queued 直接置 cancelled、对 running 只置 `cancel_requested=1`。输入输出路径只保存相对 `data/` 的路径。

- [ ] **Step 4: 验证并提交**

Run on server: `pytest tests/test_repository.py -v`
Expected: 原子认领、过期恢复、prompt_id 保留、旧 token 被拒绝、取消和终态不可变均 PASS。
Run locally: `python -m compileall -q src tests`

```bash
git add src/aigc_photomaker/db.py src/aigc_photomaker/models.py src/aigc_photomaker/repository.py tests/test_repository.py
 git commit -m "feat: add durable single-worker task store"
```

---

### Task 6: ComfyUI HTTP/WebSocket 客户端和错误映射

**Files:**
- Create: `src/aigc_photomaker/comfy.py`
- Test: `tests/test_comfy.py`

**Interfaces:**
- Consumes: `ComfyImageRef`、`prepare_workflow`、`AppError`。
- Produces: `ExecutionResult(prompt_id, images, history)`。
- Produces: `ComfyUIClient.system_stats/upload_image/submit/history/wait_for_result/download_image/cancel` 和静态方法 `map_execution_error(message: str) -> AppError`。

- [ ] **Step 1: 写失败的 HTTP 和错误映射测试**

```python
# tests/test_comfy.py
import httpx
import pytest
from aigc_photomaker.comfy import ComfyUIClient
from aigc_photomaker.errors import AppError

def test_submit_returns_prompt_id():
    def handler(request):
        assert request.url.path == "/prompt"
        return httpx.Response(200, json={"prompt_id":"p1"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    comfy = ComfyUIClient("http://127.0.0.1:8188", http_client=client)
    assert comfy.submit({"1":{}}, "worker") == "p1"

@pytest.mark.parametrize("message,code", [("CUDA out of memory", "GPU_OOM"), ("NO_FACE_DETECTED: none", "NO_FACE_DETECTED"), ("MULTIPLE_FACES_DETECTED: two", "MULTIPLE_FACES_DETECTED")])
def test_execution_errors_are_mapped(message, code):
    error = ComfyUIClient.map_execution_error(message)
    assert isinstance(error, AppError)
    assert error.code == code
    assert str(error).startswith(code)
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_comfy.py -v`
Expected: 客户端不存在，FAIL。

- [ ] **Step 3: 写客户端实现**

HTTP 接口约定如下：

```python
client.upload_image(path)       # POST /upload/image，multipart 字段 image，返回 ComfyImageRef
client.submit(graph, client_id) # POST /prompt，JSON {"prompt": graph, "client_id": client_id}
client.history(prompt_id)       # GET /history/{prompt_id}，返回该 prompt 的 dict
client.system_stats()           # GET /system_stats
client.download_image(ref, dst) # GET /view?filename=ref.name&subfolder=ref.subfolder&type=ref.type，流式写 .part 后 replace
client.cancel(prompt_id)        # POST /queue 删除排队 ID；运行中再 POST /interrupt
```

内部 HTTPX Client 使用 `trust_env=False`，避免服务器代理把回环 ComfyUI 请求发往外部。`wait_for_result(prompt_id, timeout_seconds, heartbeat, is_cancel_requested, on_progress)` 连接 `ws://127.0.0.1:8188/ws?client_id={worker_id}`，WebSocket 创建时同样绕过环境代理。WebSocket `progress` 事件调用 `on_progress(value, max)`；`execution_error` 转成 `AppError`；收到成功事件后读取 history 并提取所有 SaveImage 输出。WebSocket 断线时每 0.5 秒轮询 history，直到成功、错误、取消或 120 秒超时；每次循环执行 heartbeat。超时先 cancel，再抛 `COMFYUI_TIMEOUT`。history 中 `status.status_str == "error"` 时解析 `messages`，不直接信任任意输出文件名。

- [ ] **Step 4: 验证并提交**

Run on server: `pytest tests/test_comfy.py -v`
Expected: 上传、提交、history 成功/错误、WebSocket 进度、断线回退、下载原子写、取消和超时均 PASS。
Run locally: `python -m compileall -q src tests`

```bash
git add src/aigc_photomaker/comfy.py tests/test_comfy.py
 git commit -m "feat: integrate ComfyUI execution API"
```

---

### Task 7: Worker、任务处理、预热和保留期清理

**Files:**
- Create: `src/aigc_photomaker/worker.py`, `warmup.py`, `retention.py`
- Create: `assets/warmup-portrait.png`, `assets/README.md`
- Test: `tests/test_worker.py`, `tests/test_retention.py`

**Interfaces:**
- Consumes: Repository、ComfyUIClient、StylePreset、Settings。
- Produces: `TaskProcessor.process(task: TaskRecord) -> None`、`Worker.run_forever()`。
- Produces: `run_warmup(settings, modes=("standard","turbo")) -> dict` 和 `RetentionCleaner.cleanup(now: float) -> int`。

- [ ] **Step 1: 写失败的新任务、恢复和清理测试**

测试 FakeComfyClient 依次记录 `upload_image → submit → wait_for_result → download_image`。覆盖：新任务成功；已有 prompt_id 的任务不重复 submit；取消运行任务调用 cancel；ComfyUI 错误保存 code/retryable；旧 claim_token 不能完成任务；清理只删除过期终态。

```python
def test_resume_does_not_submit_twice(processor, resumed_task, fake_comfy):
    processor.process(resumed_task)
    assert fake_comfy.submit_calls == 0
    assert fake_comfy.wait_calls == [resumed_task.prompt_id]
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_worker.py tests/test_retention.py -v`
Expected: Worker 和清理模块不存在，FAIL。

- [ ] **Step 3: 写任务处理主流程**

```python
# TaskProcessor.process 的明确顺序
if task.cancel_requested:
    if task.prompt_id:
        comfy.cancel(task.prompt_id)
    repo.mark_cancelled(task.task_id, task.claim_token)
    return
image_ref = comfy.upload_image(data_dir / task.input_path) if not task.prompt_id else None
if not task.prompt_id:
    graph = prepare_workflow(task.mode, image_ref, styles[task.style], task.seed, f"aigc-photomaker/{task.task_id}", settings)
    prompt_id = comfy.submit(graph, worker_id)
    repo.mark_submitted(task.task_id, task.claim_token, prompt_id)
else:
    prompt_id = task.prompt_id
result = comfy.wait_for_result(prompt_id, settings.execution_timeout_seconds, heartbeat, cancel_requested, progress)
if not result.images:
    raise AppError(500, "RESULT_MISSING", "ComfyUI 未返回输出图片", True)
comfy.download_image(result.images[0], data_dir / task.output_path)
repo.succeed(task.task_id, task.claim_token, timings)
```

Worker 每 0.5 秒尝试认领，2 秒原子更新 `.runtime/app/worker-heartbeat.json`，启动时创建 schema。收到 SIGTERM 后停止认领新任务；当前任务按取消策略结束。所有文件写入先解析目标并验证仍位于 `data/` 内。

- [ ] **Step 4: 写启动预热和保留期实现**

先取得并记录有明确来源的单人脸预热素材，避免启动流程在 Task 10 时才缺少文件：

```bash
mkdir -p assets
curl -fsSL https://ghproxy.net/https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/astronaut.png -o assets/warmup-portrait.png
sha256sum assets/warmup-portrait.png
```

`assets/README.md` 记录 scikit-image 原始 URL、NASA 素材说明、数据许可说明和实际 SHA256。`run_warmup` 将该图片复制到 ComfyUI input，以固定 seed 0、1024×1024、Standard 25 步和 Turbo 8 步执行完整人脸/身份/风格 LoRA/VAE/输出单人脸守卫链路，下载并验证 PNG，最后原子写入 `.runtime/app/warmup.json`。预热不再用 1 步近似图规避输出质量检查。

RetentionCleaner 每小时运行一次，仅清理 `succeeded/failed/cancelled` 且 `finished_at < now - retention_days*86400` 的任务；删除输入、输出和数据库记录，所有路径执行 `resolve().relative_to(data_dir.resolve())` 检查。

- [ ] **Step 5: 验证并提交**

Run on server: `pytest tests/test_worker.py tests/test_retention.py -v`
Expected: 全部 PASS。
Run locally: `python -m compileall -q src tests`

```bash
git add src/aigc_photomaker/worker.py src/aigc_photomaker/warmup.py src/aigc_photomaker/retention.py assets tests/test_worker.py tests/test_retention.py
 git commit -m "feat: add resilient worker and warmup"
```

---

### Task 8: FastAPI 任务接口、健康检查和静态托管

**Files:**
- Create: `src/aigc_photomaker/task_service.py`, `app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: 所有前序接口。
- Produces: `create_app(settings: Settings | None = None) -> FastAPI`。
- Produces: `TaskService.create/get_status/get_result/cancel`。

- [ ] **Step 1: 写失败的 API 合约测试**

使用临时 SQLite、临时 data 目录和 FastAPI TestClient。测试未带 key 返回 401；非法风格返回 400；合法 JPEG 创建 202 且数据库为 queued；状态包含 seed/style/mode/timings；未完成结果返回 409；结果需鉴权；取消 queued 返回 cancelled；ready 在预热文件或 Worker 心跳缺失时返回 503。

```python
response = client.post(
    "/api/v1/tasks",
    headers={"X-API-Key":"a" * 32},
    files={"file":("face.png", png_bytes, "image/png")},
    data={"style":"cyberpunk","mode":"standard","seed":"42"},
)
assert response.status_code == 202
assert response.json()["status"] == "queued"
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_api.py -v`
Expected: API 模块不存在，FAIL。

- [ ] **Step 3: 写流式上传服务和路由**

`TaskService.create` 每次读取 1MB，累计超过 `max_upload_bytes` 立即返回 `IMAGE_TOO_LARGE`，不将整个上传读入内存。写入临时文件后调用 `normalize_image`，种子为空时使用 `secrets.randbits(53)`，随后创建任务；任一步失败都删除临时文件和已创建输入。

```python
def require_api_key(request: Request, settings: Settings = Depends(get_settings)):
    if not verify_api_key(request.headers.get("X-API-Key", ""), settings.api_key):
        raise AppError(401, "UNAUTHORIZED", "API Key 无效", False)

def client_key(request: Request) -> str:
    raw_key = request.headers.get("X-API-Key", "").encode()
    fingerprint = hashlib.sha256(raw_key).hexdigest()[:16]
    return f"{request.client.host}:{fingerprint}"

@app.post("/api/v1/tasks", status_code=202, dependencies=[Depends(require_api_key)])
async def create_task(request: Request, file: UploadFile = File(...), style: str = Form(...), mode: str = Form("standard"), seed: int | None = Form(None)):
    generation_limiter.allow_or_raise(client_key(request))
    return await task_service.create(file, style, mode, seed)
```

实现 `GET /styles`、`GET /tasks/{task_id}`、`GET /tasks/{task_id}/result`、`POST /tasks/{task_id}/cancel`、live/ready。AppError 由统一 exception handler 返回 `{"error":{"code":"INVALID_IMAGE","message":"图片无法解码","retryable":false},"task_id":"task-uuid"}` 这一结构；未知异常只返回 `INTERNAL_ERROR`。ready 只检查模型存在和文件大小，SHA256 全量校验留给安装/启动，避免每次健康检查读取 7GB。API 路由注册后再将 `web/` 以 `StaticFiles(html=True)` 挂载到根路径。

- [ ] **Step 4: 验证并提交**

Run on server: `pytest tests/test_api.py -v`
Expected: 全部 PASS，并覆盖超大上传、错误 seed、未知任务、结果鉴权和异常文件清理。
Run locally: `python -m compileall -q src tests`

```bash
git add src/aigc_photomaker/task_service.py src/aigc_photomaker/app.py tests/test_api.py
 git commit -m "feat: expose authenticated generation API"
```

---

### Task 9: 中文响应式 Web 操作界面

**Files:**
- Create: `web/index.html`, `web/styles.css`, `web/app.js`
- Create: `scripts/static_check.py` 的 Web 静态检查部分
- Test: 服务器真实浏览器验收，不以源码文本检查代替功能测试

**Interfaces:**
- Consumes: Task 8 API 合约。
- Produces: 浏览器端上传、预览、风格/模式/种子、生成、取消、轮询、结果和下载。

- [ ] **Step 1: 写静态资源检查器并确认静态检查先失败**

静态检查器验证 HTML 包含 `file-input`、`style-select`、`mode-standard`、`mode-turbo`、`seed-input`、`generate-button`、`cancel-button`、`progress`、`result-image`、`download-link`；JS 不使用 `localStorage`，API Key 只使用 `sessionStorage`；页面无外部 CDN 脚本和字体。这些结果只证明静态交付结构，不证明浏览器行为。

- [ ] **Step 2: 在本地执行静态检查，确认文件尚未就绪**

Run locally: `python scripts/static_check.py --web`
Expected: 因静态文件不存在而返回非零；这只是静态 RED，不是浏览器功能测试。

- [ ] **Step 3: 写页面和 API 调用**

页面包含清晰的人脸授权提示、单人脸/10MB 限制、Standard“质量优先”和 Turbo“速度优先”说明。CSS 使用本地变量和响应式网格，不加载外部资源。JS 使用以下接口封装：

```javascript
async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set('X-API-Key', sessionStorage.getItem('aigc.apiKey') || '');
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error?.message || `请求失败 (${response.status})`);
  }
  return response;
}
```

生成时禁用重复提交，以 FormData 上传；每 500ms 查询任务，按真实 stage/progress 更新；完成后鉴权下载 Blob、撤销旧 ObjectURL、显示图片和下载链接。错误只通过 `textContent` 写入页面；取消按钮调用 cancel；120 秒前端超时只提示查看任务，不假设后台已停止。

- [ ] **Step 4: 服务器浏览器验收并提交**

Run locally: `python scripts/static_check.py --web`
Expected: 静态结构检查 PASS。
Browser checklist on server: 手机宽度无横向滚动；拖放和文件选择均可；错误 key、无人脸、多人脸、取消、Standard、Turbo、固定 seed 和下载行为正确；刷新后 API Key 会话行为符合说明。这些浏览器结果才是本任务的功能验收证据。

```bash
git add web scripts/static_check.py
 git commit -m "feat: add responsive portrait generation UI"
```

---

### Task 10: Conda、镜像下载、离线包和进程脚本

**Files:**
- Create: `scripts/lib.sh`, `install.sh`, `download_models.sh`, `start.sh`, `stop.sh`, `status.sh`, `prepare_offline_bundle.sh`
- Create: `.env.example`, `deploy/aigc-photomaker.service`
- Test: `scripts/static_check.py` 的 Shell 静态检查部分；功能验证由服务器实际安装和启动承担

**Interfaces:**
- Consumes: Task 2 的 `ModelManager.download/verify` 和模型锁。
- Scripts 返回非零即表示未满足前置条件；`start.sh` 只有在 Standard/Turbo 预热和 API ready 后才返回成功。

- [ ] **Step 1: 写 Shell 静态检查并确认先失败**

静态检查验证所有脚本含 `set -Eeuo pipefail`，start 设置 `CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"`，ComfyUI 使用 `--listen 127.0.0.1`，且不得传会覆盖环境映射的 `--cuda-device`，offline 安装含 `--no-index`。Task 2 的模型行为测试覆盖正确 hash 跳过、错误 hash 拒绝、大小不符、`.part` 清理、离线复制和可选模型过滤。

- [ ] **Step 2: 在本地执行静态检查，确认脚本尚未就绪**

Run locally: `python scripts/static_check.py --shell`
Expected: 部署脚本不存在而返回非零；安装和启动功能不在本地运行。

- [ ] **Step 3: 写统一脚本库和目录布局**

```bash
# scripts/lib.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/.runtime"
APP_RUNTIME="$RUNTIME_DIR/app"
COMFY_DIR="${AIGC_COMFY_DIR:-$RUNTIME_DIR/ComfyUI}"
mkdir -p "$APP_RUNTIME/logs" "$APP_RUNTIME/pids" "$ROOT/data/inputs" "$ROOT/data/outputs"

# AIGC_OFFLINE=1 时所有 pip 安装使用以下参数
PIP_OFFLINE_ARGS=(--no-index --find-links="$ROOT/offline/wheelhouse")
```

每个入口脚本先加载 `lib.sh` 和 `config/versions.lock.env`，再加载不纳入 Git 的 `.env`。镜像模式使用普通 pip index；离线模式将 `PIP_OFFLINE_ARGS` 传给每一个 pip 命令，并设置 `PIP_NO_INDEX=1`，不得只让部分依赖离线。

- [ ] **Step 4: 写 Conda 安装与运行脚本**

`install.sh` 的关键命令固定为：

```bash
conda create -y -n aigc-photomaker python=3.11 pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aigc-photomaker
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
python -m pip install -r requirements/app.txt -r requirements/ml.txt
python -m pip install -r .runtime/ComfyUI/requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

安装脚本先检查 Ubuntu、x86_64、conda、gcc/g++、nvidia-smi 和物理卡 1；缺少编译器时明确输出 `sudo apt-get install -y build-essential`。ComfyUI 通过可配置 GitHub 代理 clone 后 checkout 固定提交，失败则使用代理 tarball；离线模式从 `offline/comfyui/ComfyUI.tar.gz` 解压并核对 `.aigc-commit`。随后将 vendor 节点复制到 `custom_nodes/ComfyUI-PhotoMaker-Plus`。安装生成 48 字节随机 API Key 到权限 600 的 `.env`，已有 `.env` 不覆盖。

`start.sh` 的顺序固定为：加载 `.env` → 校验模型 → 检查卡 1 空闲显存 → 复制预热图片 → 启动 ComfyUI → 等待 `/system_stats` → Standard/Turbo 预热 → 启动 Worker → 启动 Uvicorn → 等待 ready。GPU 和 ComfyUI 核心环境为：

```bash
export CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"
export HF_HUB_OFFLINE="${AIGC_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${AIGC_OFFLINE:-0}"
python .runtime/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --preview-method none ${AIGC_COMFY_EXTRA_ARGS:-}
```

`lib.sh` 提供 PID 有效性、项目路径校验、进程组优雅停止、HTTP 等待和 50MB/5 份日志轮转。status 不打印 `.env`；stop 先停 API/Worker，再处理 ComfyUI。`prepare_offline_bundle.sh` 只在联网 Ubuntu x86_64 + Python 3.11 运行，下载 wheelhouse、固定 ComfyUI 源码及模型，生成 `SHA256SUMS`。systemd 模板用 `Type=oneshot`、`RemainAfterExit=yes` 调用 start/stop 绝对路径。

- [ ] **Step 5: 验证并提交**

Run locally: `bash -n scripts/*.sh && python -m compileall -q src tests`
Expected: 静态检查 PASS，不运行安装脚本。
Run locally: `python scripts/static_check.py --shell && bash -n scripts/*.sh`
Expected: 静态检查 PASS。
Run on server: `bash scripts/install.sh`
Expected: 单元测试 PASS；镜像安装创建隔离环境。离线验收另在干净目录以 `AIGC_OFFLINE=1` 执行并禁止网络。

```bash
git add scripts .env.example deploy requirements config
 git commit -m "feat: add mirrored and offline server deployment"
```

---

### Task 11: 真实 GPU 冒烟、基准和恢复验收工具

**Files:**
- Create: `src/aigc_photomaker/smoke.py`, `benchmark.py`
- Create: `scripts/smoke_test.sh`, `benchmark.sh`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `run_smoke(api_url, api_key, input_path) -> SmokeReport`。
- Produces: `run_benchmark(api_url, api_key, input_path, mode, runs) -> BenchmarkReport`。
- 报告包含客户端总耗时、服务端阶段耗时、P50/P95、失败率、物理卡 1 显存峰值及是否满足样本数和 30 秒门槛。

- [ ] **Step 1: 写失败的统计与门槛测试**

```python
# tests/test_benchmark.py
from aigc_photomaker.benchmark import evaluate_standard, percentile

def test_percentile_and_standard_gate():
    assert percentile([1, 2, 3, 4], 95) == 4
    report = evaluate_standard([29.0] * 20, failures=0, peak_vram_mib=22000)
    assert report["passed"] is True
    assert evaluate_standard([31.0] * 20, 0, 22000)["passed"] is False
    assert evaluate_standard([20.0] * 19, 0, 22000)["passed"] is False
```

- [ ] **Step 2: 在服务器验证测试先失败**

Run on server: `pytest tests/test_benchmark.py -v`
Expected: benchmark 模块不存在，FAIL。

- [ ] **Step 3: 写真实端到端测量**

`benchmark.py` 用 `time.perf_counter()` 从 multipart 创建前开始，到结果图片完整下载结束；每 500ms 轮询，与 Web 一致。运行期间每 100ms 调用 `nvidia-smi --id=1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits`，保存最大值。报告写 `reports/benchmark-{mode}-{timestamp}.json` 和同名 Markdown，记录模型、seed、参数、版本提交和错误，不写 API Key。

Standard 必须至少 20 个成功样本、无失败、P95 ≤ 30 秒且无 OOM 才通过；Turbo 至少 10 个成功样本。另有 `--alternate-modes` 执行 Standard/Turbo 交替，报告 LoRA 切换时延和显存，不把交替测试混入单模式 P95。

- [ ] **Step 4: 写 GPU 冒烟与恢复场景**

`smoke.py` 对 Standard/Turbo 各生成一张并用 Pillow 验证 1024×1024 RGB PNG；用纯色图期望 `NO_FACE_DETECTED`；将同一授权人脸左右拼接，期望 `MULTIPLE_FACES_DETECTED`；检查 nvidia-smi 进程只出现在物理卡 1。恢复脚本输出明确命令，由验收者分别重启 Worker 和 ComfyUI，验证已有 prompt_id 不重复提交、取消生效和错误任务可查询。

- [ ] **Step 5: 验证并提交**

Run on server: `pytest tests/test_benchmark.py -v && bash scripts/smoke_test.sh && bash scripts/benchmark.sh --mode standard --runs 20 && bash scripts/benchmark.sh --mode turbo --runs 10 && bash scripts/benchmark.sh --alternate-modes`
Expected: 测试和冒烟 PASS；性能结果按真实报告判断，不为通过而修改测量数据。
Run locally: `python -m compileall -q src tests && bash -n scripts/*.sh`

```bash
git add src/aigc_photomaker/smoke.py src/aigc_photomaker/benchmark.py scripts/smoke_test.sh scripts/benchmark.sh tests/test_benchmark.py
 git commit -m "feat: add GPU smoke and performance acceptance"
```

---

### Task 12: 中文部署、模型、工作流、路线和阶段 Prompt 文档

**Files:**
- Create: `README.md`, `DEPLOYMENT.md`, `MODEL_GPU_EVALUATION.md`, `COMFYUI_WORKFLOWS.md`, `ACCEPTANCE.md`, `ROADMAP.md`, `PROMPTS.md`, `THIRD_PARTY_NOTICES.md`
- Test: `scripts/static_check.py` 的文档静态检查部分；文档命令由对应服务器验收覆盖

**Interfaces:**
- Produces: 需求方可独立阅读、复制命令和继续开发的中文文档；所有命令与脚本参数一致。

- [ ] **Step 1: 写文档静态检查并确认先失败**

静态检查验证十个根目录交付文档（含已有 `DESIGN.md` 和 `IMPLEMENTATION_PLAN.md`）均存在，README 包含物理卡 1、Conda 环境、ZIP 和 30 秒口径；DEPLOYMENT 同时包含镜像与离线步骤；PROMPTS 至少包含六个独立阶段；所有文档引用的脚本路径都真实存在；第三方说明包含 ComfyUI GPL-3.0、PhotoMaker Plus GPL-3.0、PhotoMaker V2/InsightFace 许可提示和模型来源。文档中的命令是否可执行由对应服务器验收证明，不由关键字检查推断。

- [ ] **Step 2: 在本地执行静态检查，确认文档尚未就绪**

Run locally: `python scripts/static_check.py --docs`
Expected: 要求的文档尚不齐全而返回非零。

- [ ] **Step 3: 写可操作的部署与模型文档**

README 的最短镜像路径必须完整列出：

```bash
unzip aigc-photomaker.zip
cd aigc-photomaker
export HF_ENDPOINT=https://hf-mirror.com
export GITHUB_PROXY=https://ghproxy.net/
bash scripts/install.sh
bash scripts/download_models.sh
bash scripts/start.sh
bash scripts/status.sh
bash scripts/smoke_test.sh
```

DEPLOYMENT 增加防火墙、API Key、浏览器地址、日志、停止/重启、离线包上传目录、SHA256、GPU 1 映射、OOM、缺脸、模型不匹配、端口占用和 30 秒排查。MODEL_GPU_EVALUATION 用通俗语言解释 RealVisXL/SDXL、PhotoMaker V2/InstantID/FLUX 对比、16–22GB 为工程预算、单 GPU 数量结论及扩容原则。

COMFYUI_WORKFLOWS 给出两套节点图、模型目录、UI 导入、API 提交示例、trigger `img`、LCM 参数和节点错误排查。ACCEPTANCE 提供逐项勾选和报告填写格式，区分“静态检查”“服务器功能”“真实性能”“人工画质”，不把预期写成结果。

- [ ] **Step 4: 写后续路线和每阶段完整 Prompt**

ROADMAP 固定六阶段：服务器基线与模型验收、Standard 画质调优、Turbo 时延调优、可靠性/安全加固、离线发布、后续多 GPU/多用户。每阶段写目标、前置条件、执行项、输出、退出条件和回滚方式。

PROMPTS 为每阶段提供可直接复制的独立指令，统一包含“阅读 DESIGN.md/IMPLEMENTATION_PLAN.md、不降低 1024 分辨率、不绕过身份模块、物理卡 1、先写测试、保存报告、不得虚构结果”，并分别给出该阶段具体命令和验收门槛。阶段 Prompt 不依赖对话记忆，明确允许修改的文件和不得修改的锁文件。

- [ ] **Step 5: 核对预热素材来源并提交文档**

核对 Task 7 已提交的 scikit-image `astronaut.png`、`assets/README.md` 和 SHA256，确认第三方说明引用同一来源，且预热使用真实单人脸而不是无脸随机图。

Run locally: `python scripts/static_check.py --docs && git diff --check`
Expected: 静态检查 PASS，并核对所有 Markdown 链接目标存在；不将此结果写成文档命令已执行。

```bash
git add README.md DEPLOYMENT.md MODEL_GPU_EVALUATION.md COMFYUI_WORKFLOWS.md ACCEPTANCE.md ROADMAP.md PROMPTS.md THIRD_PARTY_NOTICES.md scripts/static_check.py
 git commit -m "docs: add complete server and roadmap guides"
```

---

### Task 13: 全量审查、静态验证和 ZIP 交付

**Files:**
- Modify: 审查发现的项目内问题文件
- Create: `aigc-photomaker.zip`（不纳入 Git）
- Test: 全部测试文件和最终 ZIP 内容清单

**Interfaces:**
- Produces: 可上传服务器的应用 ZIP、静态验证结果、明确标注为待服务器执行的验收命令。

- [ ] **Step 1: 对照设计逐项审查**

逐项检查 DESIGN 的 16 节，尤其确认完整 V2 ID Encoder/UNet LoRA、单人脸一次检测、两种 UI/API 工作流、租约 fencing、结果鉴权、预热、镜像/离线、卡 1、根目录文档和六阶段 Prompt。调用代码审查技能，按严重程度修复问题；每个修复先添加可复现测试。

- [ ] **Step 2: 执行本地允许的全部静态验证**

```bash
python -m compileall -q src tests scripts
python -m json.tool config/models.lock.json > /dev/null
python -m json.tool config/styles.json > /dev/null
python scripts/build_workflows.py --check
python scripts/static_check.py --all
bash -n scripts/*.sh
git diff --check
git status --short
```

Expected: 全部返回 0；最后只有计划内未跟踪交付物。此步骤不等于 pytest、ComfyUI 或 GPU 验收。

- [ ] **Step 3: 在服务器执行完整测试与验收**

```bash
pytest -v
bash scripts/smoke_test.sh
bash scripts/benchmark.sh --mode standard --runs 20
bash scripts/benchmark.sh --mode turbo --runs 10
bash scripts/benchmark.sh --alternate-modes
```

Expected: 保存真实输出和报告。若未在服务器执行，最终说明必须明确列出这些待执行命令，不得标记“全部测试通过”。

- [ ] **Step 4: 打包并检查 ZIP**

```bash
git add -A
 git commit -m "chore: prepare server release"
git archive --format=zip --prefix=aigc-photomaker/ --output=aigc-photomaker.zip HEAD
unzip -l aigc-photomaker.zip
```

检查 ZIP 含 vendor 完整源码、许可证、四个工作流、脚本、测试、预热素材和全部根目录文档；不含 `.git`、`.env`、`.runtime`、`data`、`offline`、模型、日志、pyc 或其他 ZIP。使用 Python 标准库 `zipfile.testzip()` 检查 CRC，确认根前缀一致。

- [ ] **Step 5: 交付说明**

最终回复列出 ZIP 绝对路径、主要功能、默认模型、单 GPU 结论、静态检查证据、尚未本地执行的测试，以及服务器最短操作命令。性能和画质只引用真实服务器报告，没有报告时保持“目标/待验收”表述。

---

## Plan Self-Review Notes

- **Spec coverage:** 任务 2/3/10 覆盖模型和节点；任务 3/6 覆盖两套 ComfyUI 工作流及程序调用；任务 4/5/6/7/8 覆盖安全、持久化、恢复和 API；任务 9 覆盖 Web；任务 10 覆盖 Conda/镜像/离线/卡 1；任务 11 覆盖 Standard 20 次、Turbo 10 次及交替验收；任务 12 覆盖全部根目录文档和阶段 Prompt；任务 13 覆盖 ZIP 和诚实交付。
- **Type consistency:** `ComfyImageRef` 在 Task 3 定义并由 Task 6 复用；Repository 的 claim_token 在 Task 5 定义并在 Task 7 使用；Settings 的 Standard/Turbo 字段在 Task 1 定义并由 Task 3/7/11 使用；API 合约在 Task 8 定义并由 Task 9/11 使用。
- **Runtime honesty:** 每个 pytest 步骤均指定服务器执行，本地步骤只进行静态检查，符合需求方“不在本地运行”的更高优先级约束。
