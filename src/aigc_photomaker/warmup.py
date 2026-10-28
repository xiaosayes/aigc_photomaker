import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image

from .comfy import ComfyUIClient
from .config import PROJECT_ROOT, Settings, get_settings
from .errors import AppError
from .logging_config import configure_logging
from .styles import StylePreset
from .workflow import prepare_workflow


WARMUP_STYLE = StylePreset(
    key="warmup",
    label="Warmup",
    description="Fixed cyberpunk portrait used for full startup validation",
    positive="solo, one person, single subject, cyberpunk style cinematic portrait of a person, neon cyan and magenta rim lighting, detailed facial features",
    negative="low quality, deformed face, extra fingers, bad anatomy, multiple people, two people, duplicate person, extra face",
    lora_filename="lah-cyberpunk-sdxl.safetensors",
    lora_strength=1.0,
)


def _verify_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (1024, 1024) or image.mode != "RGB":
                raise AppError(
                    502,
                    "RESULT_MISSING",
                    "Warmup must return a 1024x1024 RGB PNG",
                    True,
                )
            image.verify()
    except AppError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise AppError(502, "RESULT_MISSING", "Warmup returned an invalid PNG", True) from exc


def _write_status(runtime_dir: Path, report: dict) -> None:
    destination = runtime_dir / "warmup.json"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=runtime_dir,
            prefix=f"{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(report, temporary, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def run_warmup(settings: Settings, modes=("standard", "turbo")) -> dict:
    runtime_dir = settings.resolve(settings.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # A rerun must invalidate prior readiness before any operation that can fail.
    (runtime_dir / "warmup.json").unlink(missing_ok=True)
    comfy_input = settings.resolve(settings.comfy_dir) / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)
    portrait = PROJECT_ROOT / "assets" / "warmup-portrait.png"
    shutil.copyfile(portrait, comfy_input / "warmup-portrait.png")

    client = ComfyUIClient(settings.comfy_url)
    mode_reports = {}
    try:
        for mode in modes:
            image_ref = client.upload_image(comfy_input / "warmup-portrait.png")
            graph = prepare_workflow(
                mode,
                image_ref,
                WARMUP_STYLE,
                0,
                f"aigc-photomaker/warmup-{mode}",
                settings,
            )
            # Validate a deliverable single-face image, not a one-step approximation.
            graph["9"]["inputs"].update(seed=0)
            graph["8"]["inputs"].update(width=1024, height=1024, batch_size=1)
            prompt_id = client.submit(graph, "aigc-photomaker-warmup")
            result = client.wait_for_result(prompt_id, settings.execution_timeout_seconds)
            if not result.images:
                raise AppError(500, "RESULT_MISSING", "ComfyUI 未返回输出图片", True)
            output = runtime_dir / f"warmup-{mode}.png"
            client.download_image(result.images[0], output)
            _verify_png(output)
            mode_reports[mode] = {
                "seed": 0,
                "steps": graph["9"]["inputs"]["steps"],
                "width": 1024,
                "height": 1024,
            }
    finally:
        client.close()

    report = {
        "status": "succeeded",
        "completed_at": time.time(),
        "modes": mode_reports,
    }
    _write_status(runtime_dir, report)
    return report


if __name__ == "__main__":
    configure_logging("warmup")
    run_warmup(get_settings())
