import hashlib
import json

from aigc_photomaker.model_manager import main


def make_model(tmp_path, *, required=True):
    payload = b"locked-model"
    relative_path = "checkpoints/model.bin" if required else "optional.bin"
    model = {
        "key": "required" if required else "optional",
        "required": required,
        "repo": "local/test",
        "revision": "1",
        "filename": "model.bin",
        "relative_path": relative_path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": [model]}), encoding="utf-8")
    source = tmp_path / "offline" / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return manifest, model


def test_cli_offline_download_and_verify(tmp_path, monkeypatch):
    manifest, model = make_model(tmp_path)
    monkeypatch.setenv("AIGC_OFFLINE", "1")
    arguments = [
        "--manifest", str(manifest),
        "--models-root", str(tmp_path / "models"),
        "--offline-dir", str(tmp_path / "offline"),
    ]
    assert main(["download", *arguments]) == 0
    assert (tmp_path / "models" / model["relative_path"]).read_bytes() == b"locked-model"
    assert main(["verify", *arguments[:-2]]) == 0


def test_cli_offline_download_requires_source(tmp_path, monkeypatch, capsys):
    manifest, _ = make_model(tmp_path)
    monkeypatch.setenv("AIGC_OFFLINE", "1")
    result = main([
        "download", "--manifest", str(manifest),
        "--models-root", str(tmp_path / "models"),
    ])
    assert result == 2
    assert "requires --offline-dir" in capsys.readouterr().err


def test_cli_verify_rejects_bad_hash(tmp_path, capsys):
    manifest, model = make_model(tmp_path)
    target = tmp_path / "models" / model["relative_path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"locked-model")
    locked = json.loads(manifest.read_text(encoding="utf-8"))
    locked["models"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(locked), encoding="utf-8")

    assert main([
        "verify", "--manifest", str(manifest),
        "--models-root", str(tmp_path / "models"),
    ]) == 1
    assert "sha256 mismatch" in capsys.readouterr().err


def test_cli_include_optional(tmp_path, capsys):
    manifest, _ = make_model(tmp_path, required=False)
    assert main([
        "verify", "--manifest", str(manifest),
        "--models-root", str(tmp_path / "models"),
    ]) == 0
    assert main([
        "verify", "--manifest", str(manifest),
        "--models-root", str(tmp_path / "models"), "--include-optional",
    ]) == 1
    assert "optional" in capsys.readouterr().err
