import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from aigc_photomaker.errors import AppError, disk_space_error

_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    sha256: str
    size_bytes: int


def _invalid_image(message: str = "图片无法解码") -> AppError:
    return AppError(400, "INVALID_IMAGE", message, False)


def _validate_image_header(image: Image.Image, max_pixels: int) -> None:
    if image.format not in _SUPPORTED_FORMATS:
        raise _invalid_image("仅支持 JPEG、PNG 或 WebP 图片")
    if image.width * image.height > max_pixels:
        raise _invalid_image("图片像素数超过限制")


def _file_metadata(destination: Path, width: int, height: int) -> ImageMetadata:
    digest = hashlib.sha256()
    size_bytes = 0
    with destination.open("rb") as output:
        while chunk := output.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return ImageMetadata(width, height, digest.hexdigest(), size_bytes)


def normalize_image(
    source: Path,
    destination: Path,
    max_pixels: int,
    max_side: int,
) -> ImageMetadata:
    temporary = destination.with_suffix(".part")
    try:
        Image.MAX_IMAGE_PIXELS = max_pixels
        with Image.open(source) as probe:
            _validate_image_header(probe, max_pixels)
            probe.verify()

        with Image.open(source) as opened:
            _validate_image_header(opened, max_pixels)
            if getattr(opened, "is_animated", False):
                raise _invalid_image("不支持动图")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.width < 128 or image.height < 128:
                raise _invalid_image("图片短边或长边小于 128 像素")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(temporary, format="PNG", compress_level=3)
            width, height = image.size

        temporary.replace(destination)
        return _file_metadata(destination, width, height)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        disk_error = disk_space_error(exc)
        if disk_error is not None:
            raise disk_error from exc
        raise _invalid_image() from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
