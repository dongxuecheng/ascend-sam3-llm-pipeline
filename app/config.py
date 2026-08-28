"""Configuration shared by the API, workers and container entrypoint."""

import os
from pathlib import Path

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, FiniteFloat


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipeline_host: str = "0.0.0.0"
    pipeline_port: int = Field(default=18080, ge=1, le=65535)
    pipeline_data_dir: Path = Path("data/events")
    pipeline_api_key: str = ""
    cors_origins: str = ""
    sam3_url: AnyHttpUrl = "http://127.0.0.1:18000/predict/file"
    llm_base_url: AnyHttpUrl = "http://127.0.0.1:8080/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    sam3_concurrency: int = Field(default=4, ge=1, le=32)
    llm_concurrency: int = Field(default=2, ge=1, le=32)
    sam3_queue_size: int = Field(default=15, ge=1, le=1000)
    llm_queue_size: int = Field(default=15, ge=1, le=1000)
    sam3_timeout_seconds: FiniteFloat = Field(default=15.0, gt=0)
    llm_timeout_seconds: FiniteFloat = Field(default=60.0, gt=0)
    shutdown_timeout_seconds: FiniteFloat = Field(default=30.0, gt=0)
    max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=16_000_000, ge=1)
    llm_max_tokens: int = Field(default=128, ge=16, le=512)

    @classmethod
    def from_env(cls) -> "Settings":
        # Do not reuse generic HOST/PORT/HOME environment variables.
        return cls(**{
            name: os.environ[name.upper()]
            for name in cls.model_fields
            if name.upper() in os.environ
        })
