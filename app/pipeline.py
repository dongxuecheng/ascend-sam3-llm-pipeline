"""Two bounded queues, independent worker pools, and no per-frame retry."""

import asyncio
import logging
import time
from collections import Counter

import httpx

from app import alarm
from app.clients import LLMClient, Sam3Client
from app.config import Settings
from app.domain import Candidate, Frame
from app.storage import EvidenceStore


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, settings: Settings, sam3: Sam3Client, llm: LLMClient, store: EvidenceStore):
        self.settings, self.sam3, self.llm, self.store = settings, sam3, llm, store
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
        logger.info("pipeline_started sam3_workers=%s llm_workers=%s sam3_queue_limit=%s llm_queue_limit=%s",
                    self.settings.sam3_concurrency, self.settings.llm_concurrency,
                    self.sam3_queue.maxsize, self.llm_queue.maxsize)

    @property
    def healthy(self) -> bool:
        return self.accepting and bool(self.workers) and all(not worker.done() for worker in self.workers)

    def submit(self, frame: Frame) -> None:
        # Atomic on the single asyncio event loop. The route maps QueueFull to 429.
        self.sam3_queue.put_nowait(frame)
        self.counts["accepted"] += 1
        logger.debug("frame_accepted machine=%s stream=%s sam3_queue=%s",
                     frame.machine_id, frame.stream_id, self.sam3_queue.qsize())

    async def drain(self) -> None:
        await self.sam3_queue.join()
        await self.llm_queue.join()

    async def stop(self) -> None:
        self.accepting = False
        try:
            await asyncio.wait_for(self.drain(), self.settings.shutdown_timeout_seconds)
        except TimeoutError:
            logger.warning("shutdown_deadline sam3_queued=%s llm_queued=%s; unfinished frames discarded",
                           self.sam3_queue.qsize(), self.llm_queue.qsize())
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
                logger.warning("%s_failed machine=%s stream=%s error_type=%s http_status=%s elapsed_ms=%.1f; frame skipped",
                               stage, frame.machine_id, frame.stream_id, type(exc).__name__,
                               status, (time.monotonic() - started) * 1000)
            finally:
                queue.task_done()

    async def _detect(self, frame: Frame, started: float) -> None:
        detections = await self.sam3.detect(frame)
        elapsed = (time.monotonic() - started) * 1000
        logger.log(logging.INFO if detections else logging.DEBUG,
                   "sam3_result machine=%s stream=%s boxes=%s elapsed_ms=%.1f",
                   frame.machine_id, frame.stream_id, len(detections), elapsed)
        if detections:
            # Backpressure: do not drop an identified candidate when LLM is busy.
            await self.llm_queue.put(Candidate(frame, detections, elapsed))
            self.counts["sam3_candidates"] += 1
        else:
            self.counts["sam3_negative"] += 1

    async def _confirm(self, candidate: Candidate, started: float) -> None:
        frame = candidate.frame
        reply = await self.llm.confirm(frame)
        elapsed = (time.monotonic() - started) * 1000
        if not reply.verdict.confirmed:
            self.counts[f"llm_{reply.verdict.result}"] += 1
            logger.info("llm_skipped machine=%s stream=%s result=%s elapsed_ms=%.1f",
                        frame.machine_id, frame.stream_id, reply.verdict.result, elapsed)
            return
        self.counts["confirmed"] += 1
        try:
            event = await asyncio.to_thread(self.store.save, candidate, reply, elapsed)
        except Exception:
            self.counts["save_failed"] += 1
            logger.exception("save_failed machine=%s stream=%s; no alarm invoked", frame.machine_id, frame.stream_id)
            return
        self.counts["saved"] += 1
        logger.info("confirmed machine=%s stream=%s result=%s llm_ms=%.1f total_ms=%.1f evidence=%s",
                    frame.machine_id, frame.stream_id, reply.verdict.result, elapsed,
                    (time.monotonic() - frame.received_monotonic) * 1000, event.directory)
        try:
            await alarm.send_alarm(event)
        except Exception:
            self.counts["alarm_failed"] += 1
            logger.exception("alarm_failed evidence=%s; saved files retained", event.directory)
