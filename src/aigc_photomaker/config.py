from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIGC_", env_file=".env", extra="ignore")

    api_key: SecretStr
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188
    comfy_dir: Path = Path(".runtime/ComfyUI")
    data_dir: Path = Path("data")
    runtime_dir: Path = Path(".runtime/app")
    max_upload_bytes: int = 10 * 1024 * 1024
    multipart_overhead_bytes: int = 64 * 1024
    min_free_disk_bytes: int = 1024 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_input_side: int = 2048
    worker_lease_seconds: int = 60
    execution_timeout_seconds: int = 120
    retention_days: int = 7
    generation_rate_per_minute: int = 20
    standard_steps: int = 8
    standard_cfg: float = 1.0
    standard_sampler: str = "euler"
    turbo_steps: int = 4
    turbo_cfg: float = 1.0
    flux_standard_unet_filename: str = "flux-2-klein-4b-fp8.safetensors"
    flux_turbo_unet_filename: str = "flux-2-klein-4b-fp8.safetensors"
    flux_text_encoder_filename: str = "qwen_3_4b.safetensors"
    flux_vae_filename: str = "flux2-vae.safetensors"
    face_provider: str = "CPU"

    @field_validator("api_key")
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 24:
            raise ValueError("AIGC_API_KEY must contain at least 24 characters")
        return value

    @property
    def comfy_url(self) -> str:
        return f"http://{self.comfy_host}:{self.comfy_port}"

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
