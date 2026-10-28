import importlib.util
import re
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "delivery_static_check", Path(__file__).parents[1] / "scripts" / "static_check.py"
)
static_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_check)


def write_core_scripts(root):
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    names = (
        "install.sh", "download_models.sh", "start.sh", "stop.sh",
        "status.sh", "prepare_offline_bundle.sh", "smoke_test.sh", "benchmark.sh",
    )
    for name in ("lib.sh", *names):
        path = scripts / name
        path.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n", encoding="utf-8")
        path.chmod(0o755)
    (scripts / "lib.sh").write_text(
        'set -Eeuo pipefail\nPIP_OFFLINE_ARGS=(--no-index --find-links="$ROOT/offline/wheelhouse")\n'
        'compgen -v AIGC_\nHF_ENDPOINT= PIP_INDEX_URL= PIP_EXTRA_INDEX_URL=\n'
        'PYTORCH_INDEX_URL= GITHUB_PROXY=\nCOMFYUI_COMMIT="$locked_comfyui_commit"\n',
        encoding="utf-8",
    )
    (scripts / "start.sh").write_text(
        'set -Eeuo pipefail\nexport CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"\n'
        'python main.py --listen 127.0.0.1\n'
        "trap cleanup_started EXIT\ntrap 'on_signal INT' INT\ntrap 'on_signal TERM' TERM\n"
        'trap - EXIT ERR INT TERM\n',
        encoding="utf-8",
    )
    (scripts / "install.sh").write_text(
        'set -Eeuo pipefail\ninstall_conda_offline() {\n'
        'prefix="$CONDA_BASE/envs/aigc-photomaker"\n"$prefix/bin/conda-unpack"\n}\n'
        'git -C "$COMFY_DIR" fetch "$COMFY_CLONE_URL"\n'
        '[[ "$current_commit" == "$COMFYUI_COMMIT" ]]\n',
        encoding="utf-8",
    )
    (scripts / "prepare_offline_bundle.sh").write_text(
        'set -Eeuo pipefail\nconda-pack\noffline/conda/aigc-photomaker.tar.gz\nfind conda \n',
        encoding="utf-8",
    )
    (scripts / "status.sh").write_text(
        'set -Eeuo pipefail\ncurl --fail-with-body\n', encoding="utf-8"
    )
    for name in ("smoke_test.sh", "benchmark.sh"):
        (scripts / name).write_text(
            'set -Eeuo pipefail\nAIGC_API_KEY=\nactivate_environment\n', encoding="utf-8"
        )


PROMPT_STAGE_FIXTURES = {
    1: ("服务器基线", "bash scripts/install.sh"),
    2: ("Standard 画质调优", "bash scripts/build_workflows.py\nconfig/styles.json"),
    3: ("Turbo 时延调优", "bash scripts/benchmark.sh --alternate-modes"),
    4: ("可靠性/安全加固", "bash scripts/smoke_test.sh\nprompt_id"),
    5: ("离线发布", "bash scripts/prepare_offline_bundle.sh"),
    6: ("多 GPU/多用户", "bash scripts/benchmark.sh\n容量"),
}


def prompt_fixture(index):
    topic, specific = PROMPT_STAGE_FIXTURES[index]
    approval = ""
    if index == 6:
        approval = "必须先编写新设计并获得明确批准；批准前不得写实现代码。\n"
    return f"""## 阶段 {index}：{topic} Prompt
目标是完成{topic}，先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md。
{approval}全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块、物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不降低 1024 分辨率、不绕过身份模块。先写测试，使用测试并保存诚实报告，不得虚构结果。
允许修改：阶段文件。不得修改：锁文件。
服务器命令：
  {specific}
输出证据：保存报告。退出标准：满足本阶段门槛。
"""


def write_core_docs(root):
    for name in static_check.REQUIRED_DOCS:
        required = static_check.DOC_REQUIRED_CONTENT.get(name, ())
        (root / name).write_text("\n".join(required), encoding="utf-8")
    (root / "PROMPTS.md").write_text(
        (root / "PROMPTS.md").read_text(encoding="utf-8")
        + "\n"
        + "\n".join(prompt_fixture(index) for index in range(1, 7)),
        encoding="utf-8",
    )
    prompts = (root / "PROMPTS.md").read_text(encoding="utf-8")
    for reference in re.findall(r"scripts/[A-Za-z0-9_.-]+\.(?:sh|py)", prompts):
        path = root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_docs_check_reports_missing_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    errors = static_check.check_docs()
    assert "缺少根目录文档: README.md" in errors
    assert "PROMPTS.md 必须包含至少六个独立阶段 Prompt" in errors


def test_readme_requires_conda_environment_and_physical_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    cases = (
        ("Conda", "", "Conda"),
        ("AIGC_CONDA_ENV", "AIGC_OTHER_ENV", "AIGC_CONDA_ENV"),
        ("物理 GPU 1", "其他 GPU", "物理 GPU 1"),
        ("CUDA_VISIBLE_DEVICES=1", "CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"),
    )
    for original, replacement, required in cases:
        write_core_docs(tmp_path)
        path = tmp_path / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(original, replacement),
            encoding="utf-8",
        )
        assert f"README.md 缺少必需内容: {required}" in static_check.check_docs()


def test_prompt_check_rejects_duplicated_stage_content(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    duplicated = "\n".join(
        prompt_fixture(1).replace("## 阶段 1：", f"## 阶段 {index}：", 1)
        for index in range(1, 7)
    )
    (tmp_path / "PROMPTS.md").write_text(duplicated, encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    errors = static_check.check_docs()
    assert "PROMPTS.md 阶段 2 缺少阶段专属内容: Standard 画质调优" in errors
    assert "PROMPTS.md 六个阶段正文不得重复" in errors


def test_prompt_check_requires_commands_in_each_stage(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    path = tmp_path / "PROMPTS.md"

    def remove_commands(match):
        section = match.group(0)
        return re.sub(r"(?m)^服务器命令：\n(?:^  .*\n?)+", "", section)

    path.write_text(
        re.sub(r"(?ms)^## 阶段 2：.*?(?=^## 阶段 )", remove_commands, path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "PROMPTS.md 阶段 2 缺少具体命令" in static_check.check_docs()


def test_prompt_check_requires_exit_criteria_in_each_stage(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    path = tmp_path / "PROMPTS.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "输出证据：保存报告。退出标准：满足本阶段门槛。",
            "输出证据：保存报告。",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "PROMPTS.md 阶段 1 缺少退出标准" in static_check.check_docs()


def test_prompt_check_requires_stage_six_design_approval_gate(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    path = tmp_path / "PROMPTS.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "必须先编写新设计并获得明确批准；批准前不得写实现代码。",
            "可以直接编写实现代码。",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "PROMPTS.md 阶段 6 必须先获得新设计明确批准再实现" in static_check.check_docs()


def test_docs_check_accepts_core_contract(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert static_check.check_docs() == []


def test_docs_check_rejects_missing_script_and_local_link(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    (tmp_path / "DEPLOYMENT.md").write_text(
        (tmp_path / "DEPLOYMENT.md").read_text(encoding="utf-8")
        + "\nscripts/missing.sh\n[缺失](missing.md)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    errors = static_check.check_docs()
    assert "DEPLOYMENT.md 引用了不存在的脚本: scripts/missing.sh" in errors
    assert "DEPLOYMENT.md 的本地 Markdown 链接不存在: missing.md" in errors


def test_all_mode_includes_docs_without_weakening_web_shell(tmp_path, monkeypatch):
    write_core_docs(tmp_path)
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    args = static_check.parse_args(["--all"])
    assert args.all and not args.docs
    assert "缺少部署脚本: scripts/start.sh" in static_check.check_shell()
    assert static_check.check_docs() == []


def test_shell_check_reports_missing_scripts(tmp_path, monkeypatch):
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "缺少部署脚本: scripts/start.sh" in static_check.check_shell()


def test_shell_check_accepts_core_contract(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert static_check.check_shell() == []


def test_shell_check_rejects_networked_offline_conda(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    path = tmp_path / "scripts" / "install.sh"
    path.write_text(path.read_text(encoding="utf-8").replace(
        'prefix="$CONDA_BASE/envs/aigc-photomaker"',
        'conda create -y -n aigc-photomaker\nprefix="$CONDA_BASE/envs/aigc-photomaker"',
    ), encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "离线 Conda 安装不得执行 conda create" in static_check.check_shell()


def test_shell_check_requires_interruption_cleanup(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    path = tmp_path / "scripts" / "start.sh"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "trap cleanup_started EXIT\n", ""
    ), encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert any("trap cleanup_started EXIT" in error for error in static_check.check_shell())


def test_shell_check_rejects_comfy_cuda_device_override(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    path = tmp_path / "scripts" / "start.sh"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "--listen 127.0.0.1", "--listen 127.0.0.1 --cuda-device 0"
    ), encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert any("--cuda-device" in error for error in static_check.check_shell())


def test_shell_check_requires_exported_override_precedence(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    path = tmp_path / "scripts" / "lib.sh"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "compgen -v AIGC_\n", ""
    ), encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "scripts/lib.sh 必须保留全部显式 AIGC_* 环境覆盖" in static_check.check_shell()


def test_shell_check_requires_readiness_http_failure(tmp_path, monkeypatch):
    write_core_scripts(tmp_path)
    path = tmp_path / "scripts" / "status.sh"
    path.write_text("set -Eeuo pipefail\n", encoding="utf-8")
    monkeypatch.setattr(static_check, "PROJECT_ROOT", tmp_path)
    assert "status.sh 必须将 readiness HTTP 503 视为失败并保留响应体" in static_check.check_shell()
