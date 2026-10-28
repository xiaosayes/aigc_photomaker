# Server-pending: run these dependency/behavior tests on the GPU server.
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from aigc_photomaker.errors import AppError
from aigc_photomaker.images import normalize_image


def test_exif_is_removed_and_rgb_png_is_written(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (3000, 1000), (20, 30, 40, 255)).save(source)
    target = tmp_path / "normalized.png"

    metadata = normalize_image(source, target, 20_000_000, 2048)

    with Image.open(target) as result:
        assert result.format == "PNG"
        assert result.mode == "RGB"
        assert result.size == (2048, 683)
        assert not result.getexif()
    assert (metadata.width, metadata.height) == (2048, 683)
    assert metadata.size_bytes == target.stat().st_size
    assert metadata.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not target.with_suffix(".part").exists()


def test_non_image_is_rejected(tmp_path: Path):
    source = tmp_path / "fake.png"
    source.write_text("not an image")

    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)

    assert not (tmp_path / "out.png").exists()
    assert not (tmp_path / "out.part").exists()


def test_real_pixel_count_over_limit_is_rejected_before_output(tmp_path: Path):
    source = tmp_path / "oversized.png"
    Image.new("RGB", (5000, 4000), "white").save(source)
    target = tmp_path / "out.png"

    with pytest.raises(AppError, match="INVALID_IMAGE") as raised:
        normalize_image(source, target, 19_999_999, 2048)

    assert raised.value.status_code == 400
    assert str(19_999_999) not in str(raised.value)
    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_pixel_limit_boundary_is_accepted(tmp_path: Path):
    source = tmp_path / "boundary.png"
    Image.new("RGB", (4000, 2500), "white").save(source)

    metadata = normalize_image(source, tmp_path / "out.png", 10_000_000, 2048)

    assert (metadata.width, metadata.height) == (2048, 1280)


def test_animated_image_is_rejected(tmp_path: Path):
    source = tmp_path / "animated.webp"
    frames = [Image.new("RGB", (256, 256), color) for color in ("red", "blue")]
    frames[0].save(source, save_all=True, append_images=frames[1:], duration=100)
    target = tmp_path / "out.png"

    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, target, 20_000_000, 2048)

    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_corrupt_image_is_rejected_and_part_file_is_removed(tmp_path: Path):
    source = tmp_path / "corrupt.png"
    Image.new("RGB", (256, 256), "white").save(source)
    source.write_bytes(source.read_bytes()[:-32])
    target = tmp_path / "out.png"

    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, target, 20_000_000, 2048)

    assert not target.exists()
    assert not target.with_suffix(".part").exists()


@pytest.mark.parametrize("size", [(127, 256), (256, 127)])
def test_dimensions_below_128_are_rejected(tmp_path: Path, size: tuple[int, int]):
    source = tmp_path / "small.jpg"
    Image.new("RGB", size, "white").save(source)

    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)


def test_jpeg_exif_orientation_is_applied(tmp_path: Path):
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (300, 200), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)

    metadata = normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)

    assert (metadata.width, metadata.height) == (200, 300)
    with Image.open(tmp_path / "out.png") as result:
        assert not result.getexif()


def test_webp_is_supported_without_upscaling(tmp_path: Path):
    source = tmp_path / "source.webp"
    Image.new("RGB", (256, 192), "white").save(source)

    metadata = normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)

    assert (metadata.width, metadata.height) == (256, 192)


def test_save_failure_removes_partial_file_and_hides_os_error(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (256, 256), "white").save(source)
    target = tmp_path / "out.png"

    def fail_save(self, fp, *args, **kwargs):
        Path(fp).write_bytes(b"partial")
        raise OSError("sensitive filesystem detail")

    monkeypatch.setattr(Image.Image, "save", fail_save)

    with pytest.raises(AppError, match="INVALID_IMAGE") as raised:
        normalize_image(source, target, 20_000_000, 2048)

    assert "sensitive filesystem detail" not in str(raised.value)
    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_cleanup_failure_does_not_mask_sanitized_image_error(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (256, 256), "white").save(source)
    target = tmp_path / "out.png"

    def fail_save(self, fp, *args, **kwargs):
        raise OSError("sensitive save detail")

    def fail_unlink(self, *args, **kwargs):
        raise OSError(f"sensitive cleanup detail: {self}")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(AppError, match="INVALID_IMAGE") as raised:
        normalize_image(source, target, 20_000_000, 2048)

    assert raised.value.status_code == 400
    assert raised.value.code == "INVALID_IMAGE"
    assert "sensitive save detail" not in str(raised.value)
    assert "sensitive cleanup detail" not in str(raised.value)
    assert str(target.with_suffix(".part")) not in str(raised.value)


def test_gif_is_rejected(tmp_path: Path):
    source = tmp_path / "source.gif"
    Image.new("RGB", (256, 256), "white").save(source)

    with pytest.raises(AppError, match="INVALID_IMAGE"):
        normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)
