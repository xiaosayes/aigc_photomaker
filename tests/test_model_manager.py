import hashlib
import json
from contextlib import contextmanager

import pytest

from aigc_photomaker.model_manager import ModelManager


def write_manifest(path, models):
    path.write_text(json.dumps({"models": models}))


def model_entry(relative="photomaker/x.bin", payload=b"model", **overrides):
    model = {
        "key": "x",
        "required": True,
        "repo": "local/test",
        "revision": "1",
        "filename": "x.bin",
        "relative_path": relative,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    model.update(overrides)
    return model


def test_offline_import_verifies_hash(tmp_path):
    payload = b"model"
    relative = "photomaker/x.bin"
    source = tmp_path / "offline" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    manifest = tmp_path / "models.json"
    write_manifest(manifest, [model_entry(relative=relative, payload=payload)])
    manager = ModelManager(manifest, tmp_path / "models", "http://network-must-not-be-used")
    manager.download(offline_dir=tmp_path / "offline")
    assert (tmp_path / "models" / relative).read_bytes() == payload


def test_offline_import_never_creates_network_client(tmp_path, monkeypatch):
    source = tmp_path / "offline" / "photomaker/x.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model")
    manifest = tmp_path / "models.json"
    write_manifest(manifest, [model_entry()])
    manager = ModelManager(manifest, tmp_path / "models", "http://network-must-not-be-used")

    def fail_client(*args, **kwargs):
        raise AssertionError("network client created in offline mode")

    monkeypatch.setattr("aigc_photomaker.model_manager.httpx.Client", fail_client)
    manager.download(offline_dir=tmp_path / "offline")


def test_offline_import_rejects_invalid_hash(tmp_path):
    source = tmp_path / "offline" / "photomaker/x.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"tampered")
    manifest = tmp_path / "models.json"
    write_manifest(manifest, [model_entry(payload=b"tampered", sha256="0" * 64)])
    manager = ModelManager(manifest, tmp_path / "models", "http://network-must-not-be-used")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        manager.download(offline_dir=tmp_path / "offline")
    assert not (tmp_path / "models" / "photomaker/x.bin").exists()


def test_quick_verify_checks_size_and_full_verify_checks_hash(tmp_path):
    target = tmp_path / "models" / "photomaker/x.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model")
    manifest = tmp_path / "models.json"
    write_manifest(manifest, [model_entry(sha256="0" * 64)])
    manager = ModelManager(manifest, tmp_path / "models", "http://unused")

    assert manager.verify(full_hash=False)[0].ok
    status = manager.verify(full_hash=True)[0]
    assert not status.ok
    assert "sha256 mismatch" in status.error


def test_online_download_url_encodes_spaces_in_upstream_filename(tmp_path):
    payload = b"model"
    manifest = tmp_path / "models.json"
    model = model_entry(filename="oil painting.safetensors", payload=payload)
    write_manifest(manifest, [model])
    manager = ModelManager(manifest, tmp_path / "models", "https://example.test")

    class Response:
        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            yield payload

    class Client:
        def __init__(self):
            self.urls = []

        @contextmanager
        def stream(self, method, url):
            self.urls.append(url)
            yield Response()

    client = Client()
    manager._download_online(client, model)

    assert len(client.urls) == 1
    assert "oil%20painting.safetensors" in client.urls[0]
    assert "oil painting.safetensors" not in client.urls[0]


def test_optional_models_are_excluded_by_default(tmp_path):
    manifest = tmp_path / "models.json"
    write_manifest(manifest, [
        model_entry(),
        model_entry(key="optional", required=False, relative_path="optional.bin"),
    ])
    manager = ModelManager(manifest, tmp_path / "models", "http://unused")

    assert [status.key for status in manager.verify(full_hash=False)] == ["x"]
    assert [status.key for status in manager.verify(full_hash=False, include_optional=True)] == ["x", "optional"]
