"""Crash-safe evidence storage, retention and capacity protection."""

import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.domain import Candidate, LLMReply, SAM3_THRESHOLD
from app.time_utils import as_beijing, as_utc, utc_now


_DAY_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_EVENT_PATTERN = re.compile(r"\d{2}-\d{2}-\d{2}\.\d{6}(?:-\d{2})?")
_LEGACY_EVENT_PATTERN = re.compile(r"\d{6}_\d{6}_[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class SavedEvent:
    directory: Path
    original: Path
    annotated: Path
    metadata: Path


@dataclass(frozen=True, slots=True)
class StorageStatus:
    ready: bool
    writable: bool
    total_bytes: int
    used_bytes: int
    free_bytes: int
    used_percent: float
    free_inodes_percent: float | None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "status": "ok" if self.ready else "unavailable",
            "writable": self.writable,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "used_percent": round(self.used_percent, 2),
            "free_inodes_percent": (
                round(self.free_inodes_percent, 2)
                if self.free_inodes_percent is not None else None
            ),
            "detail": self.detail or None,
        }


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    expired_events_removed: int
    pressure_events_removed: int
    temporary_paths_removed: int
    bytes_released: int
    status: StorageStatus

    @property
    def removed(self) -> int:
        return (
            self.expired_events_removed
            + self.pressure_events_removed
            + self.temporary_paths_removed
        )


class EvidenceStore:
    def __init__(self, root: Path, *, settings: Settings | None = None):
        self.root = root.resolve()
        self.settings = settings or Settings(evidence_min_free_bytes=0)
        self._maintenance_lock = threading.Lock()

    def initialize(self) -> MaintenanceReport:
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=self.root):
            pass
        return self.maintain()

    def status(self, *, check_write: bool = True) -> StorageStatus:
        writable = self.root.is_dir()
        detail = ""
        if writable and check_write:
            probe = self.root / (".storage-probe-" + uuid4().hex)
            try:
                with probe.open("xb") as output:
                    output.write(b"ok")
                    output.flush()
                    os.fsync(output.fileno())
            except OSError as exc:
                writable = False
                detail = f"write_failed:{type(exc).__name__}"
            finally:
                probe.unlink(missing_ok=True)

        usage = shutil.disk_usage(self.root)
        used_percent = (usage.used / usage.total * 100) if usage.total else 100.0
        free_inodes_percent = self._free_inode_percent()
        reasons = []
        if not writable:
            reasons.append(detail or "not_writable")
        if self.settings.evidence_min_free_bytes >= usage.total and usage.total:
            reasons.append("minimum_free_bytes_exceeds_filesystem_capacity")
        elif usage.free < self.settings.evidence_min_free_bytes:
            reasons.append("minimum_free_bytes_not_met")
        if used_percent >= self.settings.evidence_max_usage_percent:
            reasons.append("maximum_usage_reached")
        if (free_inodes_percent is not None
                and free_inodes_percent < self.settings.evidence_min_free_inodes_percent):
            reasons.append("minimum_free_inodes_not_met")
        return StorageStatus(
            ready=not reasons,
            writable=writable,
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            used_percent=used_percent,
            free_inodes_percent=free_inodes_percent,
            detail=",".join(reasons),
        )

    def maintain(self) -> MaintenanceReport:
        with self._maintenance_lock:
            before = shutil.disk_usage(self.root).free
            expired = pressure = temporary = 0
            now = time.time()
            today = as_beijing(utc_now()).date()
            cutoff = today - timedelta(days=self.settings.evidence_retention_days - 1)
            finalized, temporary_paths = self._owned_paths()

            for path in temporary_paths:
                if now - self._safe_mtime(path) >= self.settings.evidence_tmp_max_age_seconds:
                    if self._remove_owned(path):
                        temporary += 1

            remaining = []
            for event_day, path in finalized:
                if event_day < cutoff:
                    if self._remove_owned(path):
                        expired += 1
                else:
                    remaining.append((event_day, path))

            status = self.status(check_write=False)
            capacity_is_valid = not (
                self.settings.evidence_min_free_bytes >= status.total_bytes
                and status.total_bytes
            )
            if capacity_is_valid and self._under_pressure(status):
                grace_before = now - self.settings.evidence_cleanup_grace_seconds
                for _, path in sorted(remaining, key=lambda item: (item[0], item[1].name)):
                    if self._safe_mtime(path) > grace_before:
                        continue
                    if self._remove_owned(path):
                        pressure += 1
                    status = self.status(check_write=False)
                    if self._at_cleanup_target(status):
                        break

            status = self.status(check_write=True)
            after = shutil.disk_usage(self.root).free
            return MaintenanceReport(
                expired, pressure, temporary, max(0, after - before), status,
            )

    def save(self, candidate: Candidate, reply: LLMReply, llm_elapsed_ms: float) -> SavedEvent:
        if not reply.verdict.confirmed:
            raise ValueError("Only confirmed frames may be saved")
        frame = candidate.frame
        if (not _IDENTIFIER_PATTERN.fullmatch(frame.machine_id)
                or not _IDENTIFIER_PATTERN.fullmatch(frame.stream_id)):
            raise ValueError("Machine and stream identifiers are not safe path names")
        confirmed_at = utc_now()
        confirmed_beijing = as_beijing(confirmed_at)
        parent = (self.root / confirmed_beijing.strftime("%Y-%m-%d") /
                  frame.machine_id / frame.stream_id)
        if not parent.resolve().is_relative_to(self.root):
            raise ValueError("Evidence directory must stay inside the data directory")
        parent.mkdir(parents=True, exist_ok=True)
        name = confirmed_beijing.strftime("%H-%M-%S.%f")
        for sequence in range(100):
            event_name = name if sequence == 0 else f"{name}-{sequence:02d}"
            target = parent / event_name
            staging = parent / (".tmp-" + event_name)
            if target.exists():
                continue
            try:
                staging.mkdir()
            except FileExistsError:
                continue
            break
        else:
            raise FileExistsError("Could not allocate a unique evidence directory")
        original_name = "original" + frame.image.original_extension
        try:
            self._write(staging / original_name, frame.image.original)
            self._write(staging / "annotated.jpg", self._annotate(candidate))
            metadata = self._metadata(candidate, reply, llm_elapsed_ms, confirmed_at, original_name)
            self._write(staging / "metadata.json", (
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"))
            staging.rename(target)
        except Exception:
            if staging.resolve().is_relative_to(self.root):
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return SavedEvent(target, target / original_name, target / "annotated.jpg", target / "metadata.json")

    def update_alarm(self, event: SavedEvent, status: dict) -> None:
        target = event.metadata
        if (target.name != "metadata.json" or target.is_symlink()
                or not target.resolve().is_relative_to(self.root)):
            raise ValueError("Alarm metadata must stay inside the data directory")
        metadata = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Evidence metadata must be a JSON object")
        metadata["alarm"] = status
        temporary = target.with_name(".tmp-alarm-" + uuid4().hex + ".json")
        try:
            self._write(temporary, (
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _metadata(
        self, candidate: Candidate, reply: LLMReply, llm_elapsed_ms: float,
        confirmed_at: datetime, original_name: str,
    ) -> dict:
        frame = candidate.frame
        received_utc = as_utc(frame.received_at)
        confirmed_utc = as_utc(confirmed_at)
        captured_utc = as_utc(frame.captured_at) if frame.captured_at else None
        skew = (
            (received_utc - captured_utc).total_seconds()
            if captured_utc is not None else None
        )
        return {
            "machine_id": frame.machine_id,
            "stream_id": frame.stream_id,
            "stream_name": frame.stream_name,
            "captured_at": frame.captured_at.isoformat() if frame.captured_at else None,
            "received_at": received_utc.isoformat(),
            "received_at_beijing": as_beijing(received_utc).isoformat(),
            "confirmed_at": confirmed_utc.isoformat(),
            "confirmed_at_beijing": as_beijing(confirmed_utc).isoformat(),
            "server_timezone": "Asia/Shanghai",
            "capture_clock_skew_seconds": round(skew, 3) if skew is not None else None,
            "capture_clock_skew_warning": (
                abs(skew) > self.settings.max_capture_clock_skew_seconds
                if skew is not None else False
            ),
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

    def _owned_paths(self) -> tuple[list[tuple[date, Path]], list[Path]]:
        finalized: list[tuple[date, Path]] = []
        temporary: list[Path] = []
        try:
            day_entries = list(os.scandir(self.root))
        except OSError:
            return finalized, temporary
        for day_entry in day_entries:
            if not day_entry.is_dir(follow_symlinks=False) or not _DAY_PATTERN.fullmatch(day_entry.name):
                continue
            try:
                event_day = date.fromisoformat(day_entry.name)
            except ValueError:
                continue
            for machine in self._identifier_directories(
                Path(day_entry.path), legacy_prefix="machine_",
            ):
                for stream in self._identifier_directories(
                    machine, legacy_prefix="stream_",
                ):
                    for entry in self._entries(stream):
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        path = Path(entry.path)
                        if entry.name.startswith(".tmp-"):
                            temporary.append(path)
                        elif (
                            (_EVENT_PATTERN.fullmatch(entry.name)
                             or _LEGACY_EVENT_PATTERN.fullmatch(entry.name))
                            and self._is_final_event(path)
                        ):
                            finalized.append((event_day, path))
                        for child in self._entries(path):
                            if (child.is_file(follow_symlinks=False)
                                    and child.name.startswith(".tmp-alarm-")
                                    and child.name.endswith(".json")):
                                temporary.append(Path(child.path))
        return finalized, temporary

    @staticmethod
    def _entries(path: Path) -> list[os.DirEntry]:
        try:
            with os.scandir(path) as entries:
                return list(entries)
        except OSError:
            return []

    def _identifier_directories(
        self, path: Path, *, legacy_prefix: str,
    ) -> list[Path]:
        def owned(name: str) -> bool:
            return bool(
                _IDENTIFIER_PATTERN.fullmatch(name)
                or (name.startswith(legacy_prefix)
                    and _IDENTIFIER_PATTERN.fullmatch(name[len(legacy_prefix):]))
            )

        return [
            Path(entry.path) for entry in self._entries(path)
            if owned(entry.name) and entry.is_dir(follow_symlinks=False)
        ]

    @staticmethod
    def _is_final_event(path: Path) -> bool:
        metadata = path / "metadata.json"
        annotated = path / "annotated.jpg"
        originals = list(path.glob("original.*"))
        return (
            metadata.is_file() and not metadata.is_symlink()
            and annotated.is_file() and not annotated.is_symlink()
            and len(originals) == 1 and originals[0].is_file() and not originals[0].is_symlink()
        )

    def _remove_owned(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(self.root) or resolved == self.root:
                return False
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            self._remove_empty_parents(path.parent)
            return True
        except OSError:
            return False

    def _remove_empty_parents(self, path: Path) -> None:
        while path != self.root and path.resolve().is_relative_to(self.root):
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return time.time()

    def _under_pressure(self, status: StorageStatus) -> bool:
        return (
            status.used_percent >= self.settings.evidence_max_usage_percent
            or status.free_bytes < self.settings.evidence_min_free_bytes
            or (
                status.free_inodes_percent is not None
                and status.free_inodes_percent < self.settings.evidence_min_free_inodes_percent
            )
        )

    def _at_cleanup_target(self, status: StorageStatus) -> bool:
        return (
            status.used_percent <= self.settings.evidence_target_usage_percent
            and status.free_bytes >= self.settings.evidence_min_free_bytes
            and (
                status.free_inodes_percent is None
                or status.free_inodes_percent >= self.settings.evidence_min_free_inodes_percent
            )
        )

    def _free_inode_percent(self) -> float | None:
        statvfs = getattr(os, "statvfs", None)
        if statvfs is None:
            return None
        try:
            values = statvfs(self.root)
            return (values.f_favail / values.f_files * 100) if values.f_files else 100.0
        except OSError:
            return 0.0

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
