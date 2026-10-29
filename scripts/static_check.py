#!/usr/bin/env python3
"""Static delivery checks for project artifacts.

These checks verify structure only; they are not browser or functional tests.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = (
    "README.md",
    "DESIGN.md",
    "IMPLEMENTATION_PLAN.md",
    "DEPLOYMENT.md",
    "MODEL_GPU_EVALUATION.md",
    "COMFYUI_WORKFLOWS.md",
    "ACCEPTANCE.md",
    "ROADMAP.md",
    "PROMPTS.md",
    "THIRD_PARTY_NOTICES.md",
)
DOC_REQUIRED_CONTENT = {
    "README.md": (
        "unzip aigc-photomaker.zip", "cd aigc-photomaker", "aigc-photomaker",
        "Conda", "AIGC_CONDA_ENV", "物理 GPU 1",
        "CUDA_VISIBLE_DEVICES=1", "30 秒", "HF_ENDPOINT", "GITHUB_PROXY",
        "PIP_INDEX_URL", "已预热", "单并发",
    ),
    "DEPLOYMENT.md": (
        "镜像", "完全离线", "offline/conda/aigc-photomaker.tar.gz",
        "conda-unpack", "SHA256SUMS", "CUDA_VISIBLE_DEVICES=1", "防火墙",
        "API Key", "systemd", "30 秒",
    ),
    "MODEL_GPU_EVALUATION.md": (
        "RealVisXL", "SDXL", "PhotoMaker V2", "InstantID", "FLUX",
        "16–22GB", "工程估算", "单 GPU", "扩容",
    ),
    "COMFYUI_WORKFLOWS.md": (
        "standard_api.json", "turbo_api.json", "standard_ui.json", "turbo_ui.json",
        "ReferenceLatent", "Flux2Scheduler", "CFGGuider", "FluxSingleFaceInputGuard",
        "FluxSingleFaceOutputGuard", "/prompt", "/upload/image", "/history/",
    ),
    "ACCEPTANCE.md": ("静态检查", "服务器功能", "真实性能", "人工画质", "浏览器", "离线"),
    "ROADMAP.md": (
        "服务器基线与模型验收", "Standard 画质调优", "Turbo 时延调优",
        "可靠性/安全加固", "离线发布", "多 GPU/多用户",
    ),
    "PROMPTS.md": (
        "DESIGN.md", "IMPLEMENTATION_PLAN.md", "1024×1024", "CUDA_VISIBLE_DEVICES=1",
        "不得虚构结果", "新设计", "先写测试",
    ),
    "THIRD_PARTY_NOTICES.md": (
        "ComfyUI", "GPL-3.0", "PhotoMaker Plus", "PhotoMaker V2", "InsightFace",
        "RealVisXL", "SDXL", "LCM", "astronaut", "NASA", "商业",
    ),
}
PROMPT_STAGE_SPECIFIC_CONTENT = {
    1: ("服务器基线", "bash scripts/install.sh"),
    2: ("Standard 画质调优", "config/styles.json"),
    3: ("Turbo 时延调优", "--alternate-modes"),
    4: ("可靠性/安全加固", "prompt_id"),
    5: ("离线发布", "prepare_offline_bundle.sh"),
    6: ("多 GPU/多用户", "容量"),
}
REQUIRED_WEB_IDS = {
    "file-input",
    "style-select",
    "mode-standard",
    "mode-turbo",
    "seed-input",
    "generate-button",
    "cancel-button",
    "progress",
    "result-image",
    "download-link",
}


class WebHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.external_resources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.add(element_id)

        candidates: list[str] = []
        if tag == "script":
            candidates.append(attributes.get("src") or "")
        elif tag == "link":
            candidates.append(attributes.get("href") or "")
        for value in candidates:
            if re.match(r"^(?:https?:)?//", value, re.IGNORECASE):
                self.external_resources.append(value)


def check_web() -> list[str]:
    web_dir = PROJECT_ROOT / "web"
    html_path = web_dir / "index.html"
    css_path = web_dir / "styles.css"
    js_path = web_dir / "app.js"
    errors: list[str] = []

    for path in (html_path, css_path, js_path):
        if not path.is_file():
            errors.append(f"缺少 Web 静态文件: {path.relative_to(PROJECT_ROOT)}")
    if errors:
        return errors

    html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    javascript = js_path.read_text(encoding="utf-8")
    parser = WebHTMLParser()
    parser.feed(html)

    missing_ids = sorted(REQUIRED_WEB_IDS - parser.ids)
    if missing_ids:
        errors.append("HTML 缺少必需元素 ID: " + ", ".join(missing_ids))
    if parser.external_resources:
        errors.append("页面包含外部脚本、样式或字体资源: " + ", ".join(parser.external_resources))
    if re.search(r"@import\s+(?:url\()?\s*['\"]?(?:https?:)?//", css, re.IGNORECASE):
        errors.append("CSS 不得通过 @import 加载外部资源")
    if re.search(r"url\(\s*['\"]?(?:https?:)?//", css, re.IGNORECASE):
        errors.append("CSS 不得加载外部字体或其他远程资源")
    if re.search(r"\blocalStorage\b", javascript):
        errors.append("JavaScript 不得使用 localStorage")
    if not re.search(
        r"sessionStorage\s*\.\s*getItem\s*\(\s*['\"]aigc\.apiKey['\"]\s*\)",
        javascript,
    ):
        errors.append("JavaScript 必须从 sessionStorage 的 aigc.apiKey 读取 API Key")
    return errors


def _shell_function(contents: str, name: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n(.*?)^\}}", contents)
    return match.group(1) if match else ""


def check_shell() -> list[str]:
    """Check the deployment scripts without executing server-side operations."""
    scripts_dir = PROJECT_ROOT / "scripts"
    entrypoints = (
        "install.sh",
        "download_models.sh",
        "start.sh",
        "stop.sh",
        "status.sh",
        "prepare_offline_bundle.sh",
        "smoke_test.sh",
        "benchmark.sh",
    )
    errors: list[str] = []
    contents: dict[str, str] = {}
    for name in ("lib.sh", *entrypoints):
        path = scripts_dir / name
        if not path.is_file():
            errors.append(f"缺少部署脚本: scripts/{name}")
            continue
        contents[name] = path.read_text(encoding="utf-8")
        if "set -Eeuo pipefail" not in contents[name]:
            errors.append(f"scripts/{name} 必须设置 set -Eeuo pipefail")
        if name != "lib.sh" and sys.platform != "win32" and not path.stat().st_mode & 0o111:
            errors.append(f"scripts/{name} 必须可执行")

    start = contents.get("start.sh", "")
    if 'CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"' not in start:
        errors.append("scripts/start.sh 必须默认使用物理 GPU 1")
    if "--listen 127.0.0.1" not in start:
        errors.append("scripts/start.sh 的 ComfyUI 命令缺少 --listen 127.0.0.1")
    if "--cuda-device" in start:
        errors.append(
            "scripts/start.sh 不得传 --cuda-device；ComfyUI 0.42.1 会覆盖 CUDA_VISIBLE_DEVICES"
        )
    install = contents.get("install.sh", "")
    library = contents.get("lib.sh", "")
    if "--no-index" not in install + library:
        errors.append("离线 pip 安装必须包含 --no-index")
    if 'PIP_OFFLINE_ARGS=(--no-index --find-links="$ROOT/offline/wheelhouse")' not in library:
        errors.append("scripts/lib.sh 必须定义统一的离线 pip 参数")

    prepare = contents.get("prepare_offline_bundle.sh", "")
    if "conda-pack" not in prepare or "offline/conda/aigc-photomaker.tar.gz" not in prepare:
        errors.append("离线包必须使用 conda-pack 提供 Conda 环境")
    if not re.search(r"\bfind\b[^\n]*\bconda\b", prepare):
        errors.append("SHA256SUMS 必须包含离线 Conda 环境")
    offline_install = _shell_function(install, "install_conda_offline")
    if not offline_install:
        errors.append("scripts/install.sh 必须定义 install_conda_offline")
    else:
        if "conda-unpack" not in offline_install or not any(
            marker in offline_install for marker in ("envs/$conda_env", "envs/aigc-photomaker")
        ):
            errors.append("离线 Conda 安装必须解包到命名前缀并运行 conda-unpack")
        for forbidden in ("conda create", "conda install", "conda run"):
            if forbidden in offline_install:
                errors.append(f"离线 Conda 安装不得执行 {forbidden}")

    for variable in (
        "AIGC_", "HF_ENDPOINT", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
        "PYTORCH_INDEX_URL", "GITHUB_PROXY",
    ):
        if variable not in library:
            errors.append(f"scripts/lib.sh 必须保留显式环境覆盖 {variable}")
    if "compgen -v AIGC_" not in library:
        errors.append("scripts/lib.sh 必须保留全部显式 AIGC_* 环境覆盖")
    if 'COMFYUI_COMMIT="$locked_comfyui_commit"' not in library:
        errors.append(".env 不得覆盖锁定的 ComfyUI commit")
    for name in ("smoke_test.sh", "benchmark.sh"):
        acceptance_script = contents.get(name, "")
        if "AIGC_API_KEY" not in acceptance_script:
            errors.append(f"scripts/{name} 必须从环境读取 AIGC_API_KEY")
        if "activate_environment" not in acceptance_script:
            errors.append(f"scripts/{name} 必须激活服务环境")
    if "--fail-with-body" not in contents.get("status.sh", ""):
        errors.append("status.sh 必须将 readiness HTTP 503 视为失败并保留响应体")
    for contract in ("trap cleanup_started EXIT", "trap 'on_signal INT' INT", "trap 'on_signal TERM' TERM"):
        if contract not in start:
            errors.append(f"scripts/start.sh 缺少中断清理契约: {contract}")
    if "trap - EXIT ERR INT TERM" not in start:
        errors.append("scripts/start.sh 必须在完全就绪后解除清理 trap")
    if 'git -C "$COMFY_DIR" fetch "$COMFY_CLONE_URL"' not in install:
        errors.append("已有 ComfyUI checkout 的 fetch 必须使用配置的 GitHub 代理")
    if 'current_commit" == "$COMFYUI_COMMIT' not in install:
        errors.append("精确 commit 的已有 ComfyUI 安装不得要求网络 fetch")
    return errors


def _check_prompt_sections(prompts: str) -> list[str]:
    errors: list[str] = []
    headings = list(re.finditer(r"(?m)^##\s+阶段\s+([1-6])：[^\n]*$", prompts))
    numbers = [int(match.group(1)) for match in headings]
    if len(headings) < 6:
        errors.append("PROMPTS.md 必须包含至少六个独立阶段 Prompt")
    if numbers != list(range(1, 7)):
        errors.append("PROMPTS.md 必须按阶段 1 到 6 的顺序各提供一个 Prompt")
    if len(headings) != 6:
        return errors

    sections: list[str] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(prompts)
        sections.append(prompts[heading.start():end])
    normalized_bodies = {
        re.sub(r"\s+", " ", re.sub(r"^##[^\n]*\n", "", section)).strip()
        for section in sections
    }
    if len(normalized_bodies) != 6:
        errors.append("PROMPTS.md 六个阶段正文不得重复")

    for number, section in zip(range(1, 7), sections, strict=True):
        prefix = f"PROMPTS.md 阶段 {number}"
        if not any(marker in section for marker in ("目标", "只针对", "只修复", "首要产出")):
            errors.append(f"{prefix} 缺少目标和上下文")
        for label, required in (
            ("DESIGN.md 阅读要求", "DESIGN.md"),
            ("IMPLEMENTATION_PLAN.md 阅读要求", "IMPLEMENTATION_PLAN.md"),
            ("允许修改的文件权限", "允许修改"),
            ("禁止事项", "不得修改"),
            ("输出证据", "输出证据："),
            ("退出标准", "退出标准："),
            ("先写测试要求", "先写测试"),
        ):
            if required not in section:
                errors.append(f"{prefix} 缺少{label}")
        if not re.search(
            r"(?m)^\s{2,}(?:bash\s+scripts/|python(?:\s+-m|\s+scripts/)|"
            r"export\s+|unzip\s+|cd\s+|set\s+-a|\(cd\s+)",
            section,
        ):
            errors.append(f"{prefix} 缺少具体命令")
        if "1024×1024" not in section or not re.search(
            r"(?:不|不得)降低 1024 分辨率", section
        ):
            errors.append(f"{prefix} 缺少1024 分辨率约束")
        if "完整 PhotoMaker V2 身份模块" not in section or not re.search(
            r"(?:不|不得)绕过身份模块", section
        ):
            errors.append(f"{prefix} 缺少完整身份模块约束")
        for label, required in (
            ("物理 GPU 1 约束", ("物理 GPU 1", "CUDA_VISIBLE_DEVICES=1")),
            ("诚实且不得虚构结果要求", ("诚实报告", "不得虚构结果")),
        ):
            if any(token not in section for token in required):
                errors.append(f"{prefix} 缺少{label}")
        for required in PROMPT_STAGE_SPECIFIC_CONTENT[number]:
            if required not in section:
                errors.append(f"{prefix} 缺少阶段专属内容: {required}")

    stage_six = sections[5]
    if any(
        required not in stage_six
        for required in ("新设计", "明确批准", "批准前不得", "代码")
    ):
        errors.append("PROMPTS.md 阶段 6 必须先获得新设计明确批准再实现")
    return errors


def check_docs() -> list[str]:
    """Check documentation structure and references, never command execution."""
    errors: list[str] = []
    documents: dict[str, str] = {}
    for name in REQUIRED_DOCS:
        path = PROJECT_ROOT / name
        if not path.is_file():
            errors.append(f"缺少根目录文档: {name}")
            continue
        documents[name] = path.read_text(encoding="utf-8")

    for name, contents in documents.items():
        for required in DOC_REQUIRED_CONTENT.get(name, ()):
            if required not in contents:
                errors.append(f"{name} 缺少必需内容: {required}")
        for match in re.finditer(r"(?<![\w./-])(scripts/[A-Za-z0-9_.-]+\.(?:sh|py))", contents):
            reference = match.group(1)
            if not (PROJECT_ROOT / reference).is_file():
                errors.append(f"{name} 引用了不存在的脚本: {reference}")
        for match in re.finditer(r"(?<!!)\[[^]\n]*\]\(([^)\n]+)\)", contents):
            target = match.group(1).strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "/")) or re.match(
                r"^[A-Za-z][A-Za-z0-9+.-]*:", target
            ):
                continue
            local_target = target.split("#", 1)[0].split("?", 1)[0]
            if local_target and not (PROJECT_ROOT / local_target).exists():
                errors.append(f"{name} 的本地 Markdown 链接不存在: {target}")

    errors.extend(_check_prompt_sections(documents.get("PROMPTS.md", "")))
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="检查 Web 静态交付结构")
    parser.add_argument("--shell", action="store_true", help="静态检查部署 Shell 脚本")
    parser.add_argument("--docs", action="store_true", help="静态检查中文文档和本地引用")
    parser.add_argument("--all", action="store_true", help="执行全部静态检查")
    args = parser.parse_args(argv)
    if not (args.web or args.shell or args.docs or args.all):
        parser.error("请至少指定一种检查模式（如 --all、--docs、--web 或 --shell）")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors: list[str] = []
    if args.web or args.all:
        errors.extend(check_web())
    if args.shell or args.all:
        errors.extend(check_shell())
    if args.docs or args.all:
        errors.extend(check_docs())
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    checks = []
    if args.web or args.all:
        checks.append("Web 静态结构")
    if args.shell or args.all:
        checks.append("Shell 部署结构")
    if args.docs or args.all:
        checks.append("中文文档结构")
    print(f"PASS: {'、'.join(checks)}检查通过（不代表功能验收）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
