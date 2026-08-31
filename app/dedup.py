"""Per-stream LLM admission and durable alarm deduplication."""

import asyncio
import logging
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


logger = logging.getLogger(__name__)
StreamKey = tuple[str, str]
_HAZARD_BITS = {"fire": 1, "smoke": 2, "fire_smoke": 3}


@dataclass(frozen=True, slots=True)
class StreamAdmission:
    admitted: bool
    reason: str = ""
    remaining_seconds: float = 0.0


class LLMStreamGate:
    """Allow at most one queued/running LLM candidate per stream."""

    def __init__(self, cooldown_seconds: float, *, clock: Callable[[], float] = time.monotonic):
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._active: set[StreamKey] = set()
        self._last_admitted: dict[StreamKey, float] = {}
        self._operations_since_prune = 0

    def admit(self, machine_id: str, stream_id: str) -> StreamAdmission:
        key = machine_id, stream_id
        if key in self._active:
            return StreamAdmission(False, "stream_inflight")
        now = self._clock()
        self._operations_since_prune += 1
        if self._operations_since_prune >= 256:
            self.prune(now=now)
        elapsed = now - self._last_admitted.get(key, float("-inf"))
        if elapsed < self.cooldown_seconds:
            return StreamAdmission(False, "cooldown", self.cooldown_seconds - max(0.0, elapsed))
        self._active.add(key)
        self._last_admitted[key] = now
        return StreamAdmission(True)

    def release(self, machine_id: str, stream_id: str) -> None:
        self._active.discard((machine_id, stream_id))

    def prune(self, *, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        stale = [
            key for key, admitted_at in self._last_admitted.items()
            if key not in self._active and now - admitted_at >= self.cooldown_seconds
        ]
        for key in stale:
            self._last_admitted.pop(key, None)
        self._operations_since_prune = 0
        return len(stale)


@dataclass(frozen=True, slots=True)
class AlarmReservation:
    key: StreamKey
    hazards: int


@dataclass(frozen=True, slots=True)
class AlarmDecision:
    reservation: AlarmReservation | None
    reason: str = ""
    remaining_seconds: float = 0.0


class AlarmDeduplicator:
    """Persist successful alarms and serialize alarm delivery per stream."""

    def __init__(
        self, data_root: Path, cooldown_seconds: float, *,
        retention_days: int = 90, future_tolerance_seconds: float = 300,
    ):
        self.root = data_root.resolve()
        self.cooldown_seconds = cooldown_seconds
        self.retention_days = retention_days
        self.future_tolerance_seconds = future_tolerance_seconds
        self.state_directory = self.root / ".state"
        self.database = self.state_directory / "alarm-dedup.sqlite3"
        self._lock = asyncio.Lock()
        self._inflight: set[StreamKey] = set()
        self.recovered_corruption = False

    def initialize(self) -> None:
        if self.state_directory.is_symlink():
            raise ValueError("Alarm state directory must not be a symbolic link")
        self.state_directory.mkdir(parents=True, exist_ok=True)
        if not self.state_directory.resolve().is_relative_to(self.root):
            raise ValueError("Alarm state directory must stay inside the data directory")
        if self.database.is_symlink():
            raise ValueError("Alarm state database must not be a symbolic link")
        try:
            self._initialize_database()
        except sqlite3.DatabaseError:
            if not self.database.exists():
                raise
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine = self.state_directory / (
                f"alarm-dedup.corrupt-{timestamp}-{uuid4().hex}.sqlite3"
            )
            self.database.replace(quarantine)
            self.recovered_corruption = True
            logger.critical("alarm_state_corrupt quarantined=%s", quarantine)
            self._initialize_database()
        self._cleanup_quarantines()

    async def reserve(self, machine_id: str, stream_id: str, result: str) -> AlarmDecision:
        key = machine_id, stream_id
        hazards = _HAZARD_BITS[result]
        async with self._lock:
            if key in self._inflight:
                return AlarmDecision(None, "stream_inflight")
            try:
                row = await asyncio.to_thread(self._read, key)
            except Exception as exc:
                row = None
                logger.exception(
                    "alarm_state_read_failed machine=%s stream=%s error_type=%s; alarm allowed",
                    machine_id, stream_id, type(exc).__name__,
                )
            now = time.time()
            merged = hazards
            if row is not None:
                sent_at, previous_hazards = row
                if sent_at > now + self.future_tolerance_seconds:
                    logger.error(
                        "alarm_state_future_timestamp machine=%s stream=%s sent_at=%s now=%s; state ignored",
                        machine_id, stream_id, sent_at, now,
                    )
                    try:
                        await asyncio.to_thread(self._delete, key)
                    except Exception as exc:
                        logger.exception(
                            "alarm_state_future_delete_failed machine=%s stream=%s error_type=%s",
                            machine_id, stream_id, type(exc).__name__,
                        )
                else:
                    elapsed = max(0.0, now - sent_at)
                    if elapsed < self.cooldown_seconds:
                        if hazards & ~previous_hazards == 0:
                            return AlarmDecision(None, "cooldown", self.cooldown_seconds - elapsed)
                        merged = previous_hazards | hazards
            self._inflight.add(key)
            return AlarmDecision(AlarmReservation(key, merged))

    async def finish(self, reservation: AlarmReservation, *, sent: bool) -> None:
        async with self._lock:
            try:
                if sent:
                    await asyncio.to_thread(
                        self._write, reservation.key, time.time(), reservation.hazards,
                    )
            finally:
                self._inflight.discard(reservation.key)

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alarm_state (
                    machine_id TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    sent_at REAL NOT NULL,
                    hazards INTEGER NOT NULL,
                    PRIMARY KEY (machine_id, stream_id)
                )
                """
            )
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or check[0] != "ok":
                raise sqlite3.DatabaseError("Alarm state integrity check failed")
            cutoff = time.time() - self.retention_days * 86400
            connection.execute("DELETE FROM alarm_state WHERE sent_at < ?", (cutoff,))
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=5)

    def _read(self, key: StreamKey) -> tuple[float, int] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT sent_at, hazards FROM alarm_state WHERE machine_id = ? AND stream_id = ?",
                key,
            ).fetchone()
        return (float(row[0]), int(row[1])) if row is not None else None

    def _write(self, key: StreamKey, sent_at: float, hazards: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO alarm_state (machine_id, stream_id, sent_at, hazards)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(machine_id,stream_id) DO UPDATE SET
                    sent_at = excluded.sent_at,
                    hazards = excluded.hazards
                """,
                (*key, sent_at, hazards),
            )
            cutoff = time.time() - self.retention_days * 86400
            connection.execute("DELETE FROM alarm_state WHERE sent_at < ?", (cutoff,))
            connection.commit()

    def _cleanup_quarantines(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        files = sorted(
            (
                path for path in self.state_directory.glob("alarm-dedup.corrupt-*.sqlite3")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for index, path in enumerate(files):
            if index >= 3 or path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def _delete(self, key: StreamKey) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM alarm_state WHERE machine_id = ? AND stream_id = ?", key,
            )
            connection.commit()
