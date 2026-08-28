"""Save each confirmed frame as an independent evidence directory."""

import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.domain import Candidate, LLMReply, SAM3_THRESHOLD


@dataclass(frozen=True, slots=True)
class SavedEvent:
    directory: Path
    original: Path
    annotated: Path
    metadata: Path


class EvidenceStore:
    def __init__(self, root: Path, *, settings: Settings | None = None):
        self.root = root.resolve()
        self.settings = settings or Settings()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Fail startup if the mounted directory is not writable.
        with tempfile.TemporaryFile(dir=self.root):
            pass

    def save(self, candidate: Candidate, reply: LLMReply, llm_elapsed_ms: float) -> SavedEvent:
        if not reply.verdict.confirmed:
            raise ValueError("Only confirmed frames may be saved")
        frame = candidate.frame
        confirmed_at = datetime.now(timezone.utc)
        parent = (self.root / confirmed_at.strftime("%Y-%m-%d") /
                  f"machine_{frame.machine_id}" / f"stream_{frame.stream_id}")
        # Defense in depth: identifiers are also validated by the HTTP endpoint.
        if not parent.resolve().is_relative_to(self.root):
            raise ValueError("Evidence directory must stay inside the data directory")
        parent.mkdir(parents=True, exist_ok=True)
        name = confirmed_at.strftime("%H%M%S_%f") + "_" + uuid4().hex
        target = parent / name
        staging = parent / (".tmp-" + name)
        staging.mkdir()
        original_name = "original" + frame.image.original_extension
        try:
            self._write(staging / original_name, frame.image.original)
            self._write(staging / "annotated.jpg", self._annotate(candidate))
            metadata = {
                "machine_id": frame.machine_id,
                "stream_id": frame.stream_id,
                "stream_name": frame.stream_name,
                "captured_at": frame.captured_at.isoformat() if frame.captured_at else None,
                "received_at": frame.received_at.isoformat(),
                "confirmed_at": confirmed_at.isoformat(),
                "image": {
                    "width": frame.image.width, "height": frame.image.height,
                    "orientation_normalized": frame.image.orientation_normalized,
                    "box_coordinate_space": "orientation-normalized image pixels, xyxy",
                    "original": original_name, "annotated": "annotated.jpg",
                },
                "sam3": {
                    "class_names": list(self.settings.sam3_classes),
                    "threshold": SAM3_THRESHOLD,
                    "detections": [asdict(detection) for detection in candidate.detections],
                    "elapsed_ms": round(candidate.sam3_elapsed_ms, 2),
                },
                "llm": {
                    **reply.verdict.model_dump(), "raw_reply": reply.raw_reply,
                    "model": reply.model, "prompt_version": self.settings.llm_prompt_version,
                    "system_prompt": self.settings.llm_system_prompt,
                    "user_prompt": self.settings.llm_user_prompt,
                    "elapsed_ms": round(llm_elapsed_ms, 2),
                },
                "alarm": {"status": "not_configured"},
            }
            self._write(staging / "metadata.json", (
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"))
            # Readers only see a final directory when all three files are complete.
            staging.rename(target)
        except Exception:
            if staging.resolve().is_relative_to(self.root):
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return SavedEvent(target, target / original_name, target / "annotated.jpg", target / "metadata.json")

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        with path.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _annotate(candidate: Candidate) -> bytes:
        with Image.open(io.BytesIO(candidate.frame.image.inference)) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font_size = max(12, min(40, min(canvas.size) // 35))
        font = ImageFont.load_default(size=font_size)
        line_width = max(2, min(canvas.size) // 250)
        for detection in candidate.detections:
            x1, y1, x2, y2 = detection.box
            x1, x2 = min(canvas.width - 1, x1), min(canvas.width - 1, x2)
            y1, y2 = min(canvas.height - 1, y1), min(canvas.height - 1, y2)
            color = {"fire": (255, 48, 48), "smoke": (255, 165, 0)}.get(
                detection.label, (64, 180, 255),
            )
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = f"{detection.label} {detection.score:.2f}"
            bounds = draw.textbbox((0, 0), label, font=font)
            text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            left = max(0, min(x1, canvas.width - text_width - 6))
            top = max(0, y1 - text_height - 8)
            draw.rectangle((left, top, left + text_width + 6, top + text_height + 6), fill=color)
            draw.text((left + 3 - bounds[0], top + 3 - bounds[1]), label, font=font, fill="black")
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
