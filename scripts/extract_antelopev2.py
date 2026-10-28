#!/usr/bin/env python3
"""Safely extract the locked antelopev2 archive without path traversal."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_FILES = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
}


def _member_name(member):
    name = member.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or member.is_dir()
        or (member.external_attr >> 16) & 0o170000 == 0o120000
    ):
        raise ValueError("Unsafe antelopev2 archive member")
    parts = list(path.parts)
    if parts and parts[0] == "antelopev2":
        parts = parts[1:]
    if len(parts) != 1 or parts[0] not in EXPECTED_FILES:
        return None
    return parts[0]


def extract_antelopev2(archive, destination):
    archive = Path(archive)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".antelopev2-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if member.is_dir():
                    continue
                name = _member_name(member)
                if name is None:
                    continue
                with source.open(member) as input_stream, (temporary / name).open("wb") as output:
                    shutil.copyfileobj(input_stream, output)
        if {path.name for path in temporary.iterdir()} != EXPECTED_FILES:
            raise ValueError("Antelopev2 archive is missing required ONNX files")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    extract_antelopev2(args.archive, args.destination)
    print(f"Extracted antelopev2 models to {args.destination}")


if __name__ == "__main__":
    main()
