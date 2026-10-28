# Server-pending: safe antelopev2 archive extraction tests.
import importlib.util
import zipfile
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "scripts" / "extract_antelopev2.py"
spec = importlib.util.spec_from_file_location("extract_antelopev2", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

EXPECTED = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
}


def make_archive(path, prefix="antelopev2/"):
    with zipfile.ZipFile(path, "w") as archive:
        for name in EXPECTED:
            archive.writestr(prefix + name, name.encode())


def test_extracts_required_models_from_nested_archive(tmp_path):
    archive = tmp_path / "antelopev2.zip"
    make_archive(archive)
    destination = tmp_path / "models" / "antelopev2"

    module.extract_antelopev2(archive, destination)

    assert {path.name for path in destination.iterdir()} == EXPECTED
    assert (destination / "glintr100.onnx").read_bytes() == b"glintr100.onnx"


def test_second_extraction_replaces_stale_destination_contents(tmp_path):
    archive = tmp_path / "antelopev2.zip"
    make_archive(archive)
    destination = tmp_path / "models" / "antelopev2"
    module.extract_antelopev2(archive, destination)
    (destination / "stale.onnx").write_bytes(b"stale")

    module.extract_antelopev2(archive, destination)

    assert {path.name for path in destination.iterdir()} == EXPECTED
    assert not (destination / "stale.onnx").exists()


def test_missing_required_file_preserves_existing_destination(tmp_path):
    archive = tmp_path / "antelopev2.zip"
    make_archive(archive)
    destination = tmp_path / "models" / "antelopev2"
    destination.mkdir(parents=True)
    (destination / "keep.onnx").write_bytes(b"keep")

    with zipfile.ZipFile(archive, "w") as output:
        for name in EXPECTED - {"1k3d68.onnx"}:
            output.writestr(f"antelopev2/{name}", name.encode())

    with pytest.raises(ValueError, match="missing required"):
        module.extract_antelopev2(archive, destination)
    assert (destination / "keep.onnx").read_bytes() == b"keep"


def test_rejects_traversal_and_does_not_replace_existing_models(tmp_path):
    archive = tmp_path / "antelopev2.zip"
    make_archive(archive)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("../escape.onnx", b"bad")
    destination = tmp_path / "models" / "antelopev2"
    destination.mkdir(parents=True)
    (destination / "keep.onnx").write_bytes(b"keep")

    with pytest.raises(ValueError, match="Unsafe antelopev2 archive"):
        module.extract_antelopev2(archive, destination)

    assert (destination / "keep.onnx").read_bytes() == b"keep"
    assert not (tmp_path / "escape.onnx").exists()
