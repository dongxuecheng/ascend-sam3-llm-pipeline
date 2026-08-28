from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SAM3_CLASSES = ("fire", "smoke")
SAM3_THRESHOLD = 0.3
PROMPT_VERSION = "fire-smoke-v1"


@dataclass(frozen=True, slots=True)
class PreparedImage:
    original: bytes
    inference: bytes
    original_extension: str
    inference_mime: str
    width: int
    height: int
    orientation_normalized: bool = False


@dataclass(frozen=True, slots=True)
class Frame:
    image: PreparedImage
    machine_id: str
    stream_id: str
    stream_name: str | None
    captured_at: datetime | None
    received_at: datetime
    received_monotonic: float


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Candidate:
    frame: Frame
    detections: tuple[Detection, ...]
    sam3_elapsed_ms: float


class LLMVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    result: Literal["fire", "smoke", "fire_smoke", "none", "uncertain"]
    reason: str = Field(default="", max_length=512)

    @property
    def confirmed(self) -> bool:
        return self.result in {"fire", "smoke", "fire_smoke"}


@dataclass(frozen=True, slots=True)
class LLMReply:
    verdict: LLMVerdict
    raw_reply: str
    model: str
