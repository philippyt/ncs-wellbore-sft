import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_RELEASE_DIR = Path("../wellbore-completion/output/release_v2")


def resolve_release_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env_dir = os.environ.get("WELLBORE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_RELEASE_DIR


class PackConfig(BaseModel):
    mode: Literal["routed", "full"] = "routed"
    threshold: float = 3.0
    max_pages_per_category: int = 10
    always_include_first_n: int = 2
    max_pages: int = 32
    max_chars: int = 140_000
    drop_ocr_loops: bool = True
    drop_figure_only_descriptions: bool = True


class RewardWeights(BaseModel):
    format_valid: float = Field(default=0.10, ge=0.0)
    header: float = Field(default=0.25, ge=0.0)
    casing: float = Field(default=0.25, ge=0.0)
    lot: float = Field(default=0.10, ge=0.0)
    formation_tops: float = Field(default=0.25, ge=0.0)
    dst: float = Field(default=0.05, ge=0.0)
    bloat_penalty_max: float = Field(default=0.15, ge=0.0)


class InferenceConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 4096
    concurrency: int = 4
    timeout_seconds: float = 180.0
    max_retries: int = 3
    mode: Literal["open_book", "closed_book"] = "open_book"
    use_json_mode: bool = True
