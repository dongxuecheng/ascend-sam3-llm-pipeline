"""Two bounded queues, per-stream admission, and no per-frame retry."""

import asyncio
import logging
import time
from collections import Counter


import httpx

from app import alarm
from app.clients import LLMClient, Sam3Client
from app.config import Settings
from app.dedup import AlarmDeduplicator, LLMStreamGate
from app.domain import Candidate, Frame
from app.storage import EvidenceStore, SavedEvent
from app.time_utils import as_beijing, utc_now


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self, settings: Settings, sam3: Sam3Client, llm: LLMClient,
        store: EvidenceStore, alarm_deduplicator: AlarmDeduplicator,
    ):
        self.settings, self.sam3, self.llm, self.store = settings, sam3, llm, store
        self.alarm_deduplicator = alarm_deduplicator
        self.llm_gate = LLMStreamGate(settings.llm_stream_cooldown_seconds)
        self.sam3_queue: asyncio.Queue[Frame] = asyncio.Queue(settings.sam3_queue_size)
        self.llm_queue: asyncio.Queue[Candidate] = asyncio.Queue(settings.llm_queue_size)
        self.workers: list[asyncio.Task] = []
        self.counts: Counter[str] = Counter()
        self.accepting = False

    def start(self) -> None:
        for stage, queue, process, count in (
            ("sam3", self.sam3_queue, self._detect, self.settings.sam3_concurrency),
            ("llm", self.llm_queue, self._confirm, self.settings.llm_concurrency),
        ):
            self.workers.extend(
                asyncio.create_task(self._worker(stage, queue, process), name=f"{stage}-{i}")
                for i in range(count)
            )
        self.accepting = True
        logger.info(
            "pipeline_started sam3_workers=%s llm_workers=%s sam3_queue_limit=%s "
            "llm_queue_limit=%s sam3_confidence_threshold=%s "
            "llm_stream_cooldown_seconds=%s alarm_stream_cooldown_seconds=%s",
            self.settings.sam3_concurrency, self.settings.llm_concurrency,
            self.sam3_queue.maxsize, self.llm_queue.maxsize,
            self.settings.sam3_confidence_threshold,
            self.settings.llm_stream_cooldown_seconds,
            self.settings.alarm_stream_cooldown_seconds,
        )

    @property
    def healthy(self) -> bool:
        return self.accepting and bool(self.workers) and all(not worker.done() for worker in self.workers)

    def submit(self, frame: Frame) -> None:
        self.sam3_queue.put_nowait(frame)
        self.counts["accepted"] += 1
        logger.debug(
            "frame_accepted machine=%s stream=%s sam3_queue=%s",
            frame.machine_id, frame.stream_id, self.sam3_queue.qsize(),
        )

    async def drain(self) -> None:
        await self.sam3_queue.join()
        await self.llm_queue.join()

    async def stop(self) -> None:
        self.accepting = False
        try:
            await asyncio.wait_for(self.drain(), self.settings.shutdown_timeout_seconds)
        except TimeoutError:
            logger.warning(
                "shutdown_deadline sam3_queued=%s llm_queued=%s; unfinished frames discarded",
                self.sam3_queue.qsize(), self.llm_queue.qsize(),
            )
        finally:
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("pipeline_stopped counts=%s", dict(self.counts))

    async def _worker(self, stage: str, queue: asyncio.Queue, process) -> None:
        while True:
            item = await queue.get()
            started = time.monotonic()
            try:
                await process(item, started)
            except Exception as exc:
                self.counts[f"{stage}_failed"] += 1
                frame = item.frame if isinstance(item, Candidate) else item
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else "-"
                logger.warning(
                    "%s_failed machine=%s stream=%s error_type=%s http_status=%s "
                    "elapsed_ms=%.1f; frame skipped",
                    stage, frame.machine_id, frame.stream_id, type(exc).__name__,
                    status, (time.monotonic() - started) * 1000,
                )
            finally:
                queue.task_done()

    async def _detect(self, frame: Frame, started: float) -> None:
        detections = await self.sam3.detect(frame)
        elapsed = (time.monotonic() - started) * 1000
        logger.log(
            logging.INFO if detections else logging.DEBUG,
            "sam3_result machine=%s stream=%s boxes=%s elapsed_ms=%.1f",
            frame.machine_id, frame.stream_id, len(detections), elapsed,
        )
        if not detections:
            self.counts["sam3_negative"] += 1
            return

        self.counts["sam3_candidates"] += 1
        admission = self.llm_gate.admit(frame.machine_id, frame.stream_id)
        if not admission.admitted:
            self.counts[f"llm_suppressed_{admission.reason}"] += 1
            logger.info(
                "llm_suppressed machine=%s stream=%s reason=%s remaining_ms=%.1f",
                frame.machine_id, frame.stream_id, admission.reason,
                admission.remaining_seconds * 1000,
            )
            return

        try:
            await self.llm_queue.put(Candidate(frame, detections, elapsed))
            self.counts["llm_enqueued"] += 1
        except BaseException:
            self.llm_gate.release(frame.machine_id, frame.stream_id)
            raise

    async def _confirm(self, candidate: Candidate, started: float) -> None:
        frame = candidate.frame
        try:
            await self._confirm_reserved(candidate, started)
        finally:
            self.llm_gate.release(frame.machine_id, frame.stream_id)

    async def _confirm_reserved(self, candidate: Candidate, started: float) -> None:
        frame = candidate.frame
        reply = await self.llm.confirm(frame)
        elapsed = (time.monotonic() - started) * 1000
        if not reply.verdict.confirmed:
            self.counts[f"llm_{reply.verdict.result}"] += 1
            logger.info(
                "llm_skipped machine=%s stream=%s result=%s elapsed_ms=%.1f",
                frame.machine_id, frame.stream_id, reply.verdict.result, elapsed,
            )
            return
        self.counts["confirmed"] += 1
        try:
            event = await asyncio.to_thread(self.store.save, candidate, reply, elapsed)
        except Exception as exc:
            self.counts["save_failed"] += 1
            logger.exception(
                "save_failed machine=%s stream=%s; attempting alarm without persisted evidence",
                frame.machine_id, frame.stream_id,
            )
            annotated = None
            try:
                annotated = await asyncio.to_thread(self.store._annotate, candidate)
            except Exception:
                logger.exception(
                    "fallback_annotation_failed machine=%s stream=%s",
                    frame.machine_id, frame.stream_id,
                )
            fallback = alarm.UnsavedAlarmEvent(
                machine_id=frame.machine_id,
                stream_id=frame.stream_id,
                result=reply.verdict.result,
                original=frame.image.original,
                original_extension=frame.image.original_extension,
                annotated=annotated,
                metadata={
                    "machine_id": frame.machine_id,
                    "stream_id": frame.stream_id,
                    "stream_name": frame.stream_name,
                    "received_at": frame.received_at.isoformat(),
                    "received_at_beijing": as_beijing(frame.received_at).isoformat(),
                    "result": reply.verdict.result,
                    "reason": reply.verdict.reason,
                    "model": reply.model,
                    "storage_error_type": type(exc).__name__,
                    "sam3_detections": [
                        {"label": item.label, "score": item.score, "box": list(item.box)}
                        for item in candidate.detections
                    ],
                },
            )
            self.counts["alarm_after_save_failure"] += 1
            await self._deliver_alarm(fallback, frame, reply.verdict.result)
            return
        self.counts["saved"] += 1
        logger.info(
            "confirmed machine=%s stream=%s result=%s llm_ms=%.1f total_ms=%.1f evidence=%s",
            frame.machine_id, frame.stream_id, reply.verdict.result, elapsed,
            (time.monotonic() - frame.received_monotonic) * 1000, event.directory,
        )
        await self._deliver_alarm(event, frame, reply.verdict.result)

    async def _deliver_alarm(
        self, event: alarm.AlarmEvent, frame: Frame, result: str,
    ) -> None:
        decision = await self.alarm_deduplicator.reserve(
            frame.machine_id, frame.stream_id, result,
        )
        if decision.reservation is None:
            self.counts["alarm_suppressed"] += 1
            logger.info(
                "alarm_suppressed machine=%s stream=%s result=%s reason=%s remaining_ms=%.1f",
                frame.machine_id, frame.stream_id, result, decision.reason,
                decision.remaining_seconds * 1000,
            )
            await self._update_alarm_metadata(event, {
                "status": "suppressed_duplicate",
                "reason": decision.reason,
                "cooldown_seconds": self.settings.alarm_stream_cooldown_seconds,
                "remaining_seconds": round(decision.remaining_seconds, 3),
            })
            return

        reservation = decision.reservation
        try:
            sent = await alarm.send_alarm(event)
        except Exception as exc:
            await self.alarm_deduplicator.finish(reservation, sent=False)
            self.counts["alarm_failed"] += 1
            await self._update_alarm_metadata(event, {
                "status": "failed", "error_type": type(exc).__name__,
            })
            logger.exception(
                "alarm_failed evidence=%s; local evidence status unchanged",
                alarm.reference(event),
            )
            return

        if sent is not True:
            await self.alarm_deduplicator.finish(reservation, sent=False)
            return

        persisted = True
        try:
            await self.alarm_deduplicator.finish(reservation, sent=True)
        except Exception:
            persisted = False
            self.counts["alarm_state_failed"] += 1
            logger.exception(
                "alarm_state_failed machine=%s stream=%s; future duplicate suppression unavailable",
                frame.machine_id, frame.stream_id,
            )
        self.counts["alarm_sent"] += 1
        sent_at = utc_now()
        await self._update_alarm_metadata(event, {
            "status": "sent",
            "sent_at": sent_at.isoformat(),
            "sent_at_beijing": as_beijing(sent_at).isoformat(),
            "dedup_persisted": persisted,
        })

    async def _update_alarm_metadata(self, event: alarm.AlarmEvent, status: dict) -> None:
        if not isinstance(event, SavedEvent):
            return
        try:
            await asyncio.to_thread(self.store.update_alarm, event, status)
        except Exception:
            self.counts["alarm_metadata_failed"] += 1
            logger.exception("alarm_metadata_update_failed evidence=%s", event.directory)
