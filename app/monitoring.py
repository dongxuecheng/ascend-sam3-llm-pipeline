"""Cached component health, storage maintenance and periodic status logging."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from app import __version__, alarm
from app.clients import LLMClient, Sam3Client
from app.config import Settings
from app.pipeline import Pipeline
from app.storage import EvidenceStore, StorageStatus
from app.time_utils import as_beijing, utc_now


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProbeState:
    status: str = "unknown"
    checked_at: datetime | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int = 0
    error_type: str | None = None
    http_status: int | None = None

    def success(self) -> None:
        now = utc_now()
        self.status = "ok"
        self.checked_at = now
        self.last_success_at = now
        self.consecutive_failures = 0
        self.error_type = None
        self.http_status = None

    def failure(self, exc: Exception) -> None:
        self.status = "down"
        self.checked_at = utc_now()
        self.consecutive_failures += 1
        self.error_type = type(exc).__name__
        self.http_status = (
            exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        )

    def disabled(self) -> None:
        self.status = "not_checked"
        self.checked_at = utc_now()

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "checked_at": self._local(self.checked_at),
            "last_success_at": self._local(self.last_success_at),
            "consecutive_failures": self.consecutive_failures,
            "error_type": self.error_type,
            "http_status": self.http_status,
        }

    @staticmethod
    def _local(value: datetime | None) -> str | None:
        return as_beijing(value).isoformat() if value is not None else None


class RuntimeMonitor:
    def __init__(
        self, settings: Settings, pipeline: Pipeline, sam3: Sam3Client,
        llm: LLMClient, store: EvidenceStore,
    ):
        self.settings = settings
        self.pipeline = pipeline
        self.sam3 = sam3
        self.llm = llm
        self.store = store
        self.sam3_state = ProbeState()
        self.llm_state = ProbeState()
        self.storage_status = store.status(check_write=False)
        self.tasks: list[asyncio.Task] = []
        self.stopping = False

    async def start(self) -> None:
        self.storage_status = await asyncio.to_thread(self.store.status, check_write=True)
        if self.settings.upstream_health_probes_enabled:
            await self.probe_upstreams()
        else:
            self.sam3_state.disabled()
            self.llm_state.disabled()
        self.tasks = [
            asyncio.create_task(self._probe_loop(), name="health-probes"),
            asyncio.create_task(self._maintenance_loop(), name="evidence-maintenance"),
            asyncio.create_task(self._status_log_loop(), name="status-log"),
        ]

    async def stop(self) -> None:
        self.stopping = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    @property
    def live(self) -> bool:
        return (
            not self.stopping
            and self.pipeline.healthy
            and bool(self.tasks)
            and all(not task.done() for task in self.tasks)
        )

    @property
    def ready(self) -> bool:
        upstreams_ready = (
            not self.settings.upstream_health_probes_enabled
            or (self.sam3_state.status == "ok" and self.llm_state.status == "ok")
        )
        alarm_ready = (
            not self.settings.alarm_required_for_readiness or alarm.is_configured()
        )
        return self.live and self.storage_status.ready and upstreams_ready and alarm_ready

    async def probe_upstreams(self) -> None:
        await asyncio.gather(
            self._probe(self.sam3.probe, self.sam3_state, "sam3"),
            self._probe(self.llm.probe, self.llm_state, "llm"),
        )

    async def refresh_storage(self) -> None:
        self.storage_status = await asyncio.to_thread(self.store.status, check_write=True)

    def snapshot(self) -> dict:
        upstreams: str | dict
        if not self.settings.upstream_health_probes_enabled:
            upstreams = "not_checked"
        else:
            upstreams = {
                "sam3": self.sam3_state.as_dict(),
                "llm": self.llm_state.as_dict(),
            }
        now = utc_now()
        return {
            "status": "ok" if self.ready else "unavailable",
            "version": __version__,
            "server_time_utc": now.isoformat(),
            "server_time_beijing": as_beijing(now).isoformat(),
            "live": self.live,
            "ready": self.ready,
            "upstreams": upstreams,
            "storage": self.storage_status.as_dict(),
            "alarm": {
                "configured": alarm.is_configured(),
                "required_for_readiness": self.settings.alarm_required_for_readiness,
                "state_recovered_after_corruption": (
                    self.pipeline.alarm_deduplicator.recovered_corruption
                ),
            },
            "queues": {
                "sam3": self.pipeline.sam3_queue.qsize(),
                "llm": self.pipeline.llm_queue.qsize(),
            },
            "counts": dict(self.pipeline.counts),
        }

    async def _probe(self, operation, state: ProbeState, name: str) -> None:
        try:
            await operation()
        except Exception as exc:
            state.failure(exc)
            logger.warning(
                "upstream_unhealthy component=%s error_type=%s http_status=%s failures=%s",
                name, type(exc).__name__, state.http_status or "-", state.consecutive_failures,
            )
        else:
            recovered = state.status == "down"
            state.success()
            if recovered:
                logger.info("upstream_recovered component=%s", name)

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.upstream_health_probe_interval_seconds)
            if self.settings.upstream_health_probes_enabled:
                await self.probe_upstreams()
            else:
                self.sam3_state.disabled()
                self.llm_state.disabled()
            await self.refresh_storage()

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.evidence_cleanup_interval_seconds)
            try:
                report = await asyncio.to_thread(self.store.maintain)
                self.storage_status = report.status
                if report.removed:
                    logger.info(
                        "evidence_pruned expired=%s pressure=%s temporary=%s bytes_released=%s",
                        report.expired_events_removed, report.pressure_events_removed,
                        report.temporary_paths_removed, report.bytes_released,
                    )
            except Exception as exc:
                logger.exception(
                    "evidence_maintenance_failed error_type=%s", type(exc).__name__,
                )
                await self.refresh_storage()

    async def _status_log_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.status_log_interval_seconds)
            logger.info(
                "runtime_status live=%s ready=%s sam3_health=%s llm_health=%s "
                "storage_ready=%s storage_used_percent=%.2f storage_free_bytes=%s "
                "sam3_queue=%s llm_queue=%s counts=%s",
                self.live, self.ready, self.sam3_state.status, self.llm_state.status,
                self.storage_status.ready, self.storage_status.used_percent,
                self.storage_status.free_bytes, self.pipeline.sam3_queue.qsize(),
                self.pipeline.llm_queue.qsize(), dict(self.pipeline.counts),
            )



