"""Two bounded queues, independent worker pools, and no per-frame retry."""

import asyncio
import logging
import time
from collections import Counter

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
        self.workers = [
            asyncio.create_task(self._sam3_worker(), name=f"sam3-{i}")
            for i in range(self.settings.sam3_concurrency)
        ] + [
            asyncio.create_task(self._llm_worker(), name=f"llm-{i}")
            for i in range(self.settings.llm_concurrency)
        ]
        self.accepting = True

    @property
    def healthy(self) -> bool:
        return self.accepting and bool(self.workers) and all(not worker.done() for worker in self.workers)

    def submit(self, frame: Frame) -> None:
        # Atomic on the single asyncio event loop. The route maps QueueFull to 429.
        self.sam3_queue.put_nowait(frame)
        self.counts["accepted"] += 1

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

    async def _sam3_worker(self) -> None:
        while True:
            frame = await self.sam3_queue.get()
            started = time.monotonic()
            try:
                detections = await self.sam3.detect(frame)
                elapsed = (time.monotonic() - started) * 1000
                if detections:
                    # Backpressure: do not drop an identified candidate when LLM is busy.
                    await self.llm_queue.put(Candidate(frame, detections, elapsed))
                    self.counts["sam3_candidates"] += 1
                else:
                    self.counts["sam3_negative"] += 1
            except Exception as exc:
                self.counts["sam3_failed"] += 1
                logger.warning("sam3_failed machine=%s stream=%s error_type=%s; frame skipped",
                               frame.machine_id, frame.stream_id, type(exc).__name__)
            finally:
                self.sam3_queue.task_done()

    async def _llm_worker(self) -> None:
        while True:
            candidate = await self.llm_queue.get()
            frame = candidate.frame
            started = time.monotonic()
            try:
                try:
                    reply = await self.llm.confirm(frame)
                except Exception as exc:
                    self.counts["llm_failed"] += 1
                    logger.warning("llm_failed machine=%s stream=%s error_type=%s; frame skipped",
                                   frame.machine_id, frame.stream_id, type(exc).__name__)
                    continue
                llm_elapsed_ms = (time.monotonic() - started) * 1000
                if not reply.verdict.confirmed:
                    self.counts[f"llm_{reply.verdict.result}"] += 1
                    logger.info("llm_skipped machine=%s stream=%s result=%s",
                                frame.machine_id, frame.stream_id, reply.verdict.result)
                    continue
                self.counts["confirmed"] += 1
                try:
                    event = await asyncio.to_thread(self.store.save, candidate, reply, llm_elapsed_ms)
                except Exception:
                    self.counts["save_failed"] += 1
                    logger.exception("save_failed machine=%s stream=%s; no alarm invoked",
                                     frame.machine_id, frame.stream_id)
                    continue
                self.counts["saved"] += 1
                logger.info("confirmed machine=%s stream=%s result=%s total_ms=%.1f evidence=%s",
                            frame.machine_id, frame.stream_id, reply.verdict.result,
                            (time.monotonic() - frame.received_monotonic) * 1000, event.directory)
                try:
                    await alarm.send_alarm(event)
                except Exception:
                    self.counts["alarm_failed"] += 1
                    logger.exception("alarm_failed evidence=%s; saved files retained", event.directory)
            finally:
                self.llm_queue.task_done()
