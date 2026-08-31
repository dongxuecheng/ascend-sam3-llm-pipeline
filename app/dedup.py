"""Per-stream LLM admission and durable alarm deduplication."""

import asyncio
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


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

    def admit(self, machine_id: str, stream_id: str) -> StreamAdmission:
        key = machine_id, stream_id
        if key in self._active:
            return StreamAdmission(False, "stream_inflight")
        now = self._clock()
        elapsed = now - self._last_admitted.get(key, float("-inf"))
        if elapsed < self.cooldown_seconds:
            return StreamAdmission(False, "cooldown", self.cooldown_seconds - max(0.0, elapsed))
        # This check-and-set has no await and is atomic on the single asyncio event loop.
        self._active.add(key)
        self._last_admitted[key] = now
        return StreamAdmission(True)

    def release(self, machine_id: str, stream_id: str) -> None:
        self._active.discard((machine_id, stream_id))


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

    def __init__(self, data_root: Path, cooldown_seconds: float):
        self.root = data_root.resolve()
        self.cooldown_seconds = cooldown_seconds
        self.state_directory = self.root / ".state"
        self.database = self.state_directory / "alarm-dedup.sqlite3"
        self._lock = asyncio.Lock()
        self._inflight: set[StreamKey] = set()

    def initialize(self) -> None:
        if self.state_directory.is_symlink():
            raise ValueError("Alarm state directory must not be a symbolic link")
        self.state_directory.mkdir(parents=True, exist_ok=True)
        if not self.state_directory.resolve().is_relative_to(self.root):
            raise ValueError("Alarm state directory must stay inside the data directory")
        if self.database.is_symlink():
            raise ValueError("Alarm state database must not be a symbolic link")
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
            connection.commit()

    async def reserve(self, machine_id: str, stream_id: str, result: str) -> AlarmDecision:
        key = machine_id, stream_id
        hazards = _HAZARD_BITS[result]
        async with self._lock:
            if key in self._inflight:
                return AlarmDecision(None, "stream_inflight")
            row = await asyncio.to_thread(self._read, key)
            now = time.time()
            merged = hazards
            if row is not None:
                sent_at, previous_hazards = row
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
                ON CONFLICT(machine_id, stream_id) DO UPDATE SET
                    sent_at = excluded.sent_at,
                    hazards = excluded.hazards
                """,
                (*key, sent_at, hazards),
            )
            connection.commit()
