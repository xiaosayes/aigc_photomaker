import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx


DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class ModelStatus:
    key: str
    path: Path
    ok: bool
    error: str | None


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelManager:
    def __init__(
        self,
        manifest_path: Path,
        models_root: Path,
        hf_endpoint: str = DEFAULT_HF_ENDPOINT,
    ):
        self.manifest_path = manifest_path
        self.models_root = models_root
        self.hf_endpoint = hf_endpoint.rstrip("/")
        self.models = json.loads(manifest_path.read_text(encoding="utf-8"))["models"]

    def _selected_models(self, include_optional):
        return [model for model in self.models if include_optional or model["required"]]

    def verify_one(self, model, path, full_hash):
        if not path.is_file():
            return ModelStatus(model["key"], path, False, "missing file")
        actual_size = path.stat().st_size
        if actual_size != model["size"]:
            return ModelStatus(
                model["key"],
                path,
                False,
                f"size mismatch: expected {model['size']}, got {actual_size}",
            )
        if full_hash:
            actual_hash = sha256_file(path)
            if actual_hash != model["sha256"]:
                return ModelStatus(
                    model["key"],
                    path,
                    False,
                    f"sha256 mismatch: expected {model['sha256']}, got {actual_hash}",
                )
        return ModelStatus(model["key"], path, True, None)

    def verify(self, full_hash: bool, include_optional=False) -> list[ModelStatus]:
        return [
            self.verify_one(
                model,
                self.models_root / model["relative_path"],
                full_hash=full_hash,
            )
            for model in self._selected_models(include_optional)
        ]

    def import_offline(self, model, offline_dir):
        source = offline_dir / model["relative_path"]
        target = self.models_root / model["relative_path"]
        source_status = self.verify_one(model, source, full_hash=True)
        if not source_status.ok:
            raise ValueError(f"{model['key']}: {source_status.error}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(source, temporary)
        temporary_status = self.verify_one(model, temporary, full_hash=True)
        if not temporary_status.ok:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"{model['key']}: {temporary_status.error}")
        os.replace(temporary, target)

    def _download_online(self, client, model):
        target = self.models_root / model["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        url = (
            f"{self.hf_endpoint}/{model['repo']}/resolve/"
            f"{model['revision']}/{quote(model['filename'], safe='/')}"
        )

        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            stream.write(chunk)
                status = self.verify_one(model, temporary, full_hash=True)
                if not status.ok:
                    raise ValueError(f"{model['key']}: {status.error}")
                os.replace(temporary, target)
                return
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise

    def download(self, include_optional=False, offline_dir=None):
        models = self._selected_models(include_optional)
        if offline_dir is not None:
            for model in models:
                target = self.models_root / model["relative_path"]
                if self.verify_one(model, target, full_hash=True).ok:
                    continue
                self.import_offline(model, offline_dir)
        else:
            pending = [
                model
                for model in models
                if not self.verify_one(
                    model,
                    self.models_root / model["relative_path"],
                    full_hash=True,
                ).ok
            ]
            if pending:
                with httpx.Client(follow_redirects=True) as client:
                    for model in pending:
                        self._download_online(client, model)
        return self.verify(full_hash=True, include_optional=include_optional)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or verify locked PhotoMaker models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--manifest", type=Path, default=Path("config/models.lock.json"))
        subparser.add_argument(
            "--models-root", type=Path, default=Path(".runtime/ComfyUI/models")
        )
        subparser.add_argument("--include-optional", action="store_true")
        if command == "download":
            subparser.add_argument("--offline-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if os.environ.get("AIGC_OFFLINE") == "1" and args.command == "download":
        if args.offline_dir is None:
            print("FAIL: AIGC_OFFLINE=1 requires --offline-dir", file=sys.stderr)
            return 2
    manager = ModelManager(
        args.manifest,
        args.models_root,
        hf_endpoint=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
    )
    if args.command == "download":
        statuses = manager.download(
            include_optional=args.include_optional,
            offline_dir=args.offline_dir,
        )
    else:
        statuses = manager.verify(
            full_hash=True,
            include_optional=args.include_optional,
        )
    failed = False
    for status in statuses:
        if status.ok:
            print(f"OK: {status.key}")
        else:
            failed = True
            print(f"FAIL: {status.key}: {status.error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
