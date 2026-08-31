import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime
from starlette.types import ASGIApp, Receive, Scope, Send

from app import __version__
from app.clients import LLMClient, Sam3Client
from app.config import Settings
from app.dedup import AlarmDeduplicator
from app.domain import Frame
from app.images import ImageTooLarge, InvalidImage, prepare_image
from app.monitoring import RuntimeMonitor
from app.pipeline import Pipeline
from app.storage import EvidenceStore
from app.time_utils import as_utc, utc_now


logger = logging.getLogger(__name__)


class UploadLimitMiddleware:
    """Bound the whole multipart body before the framework spools uploaded files."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/v1/frames":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["frame_received_at"] = utc_now()
        state["frame_received_monotonic"] = time.monotonic()
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError
        except ValueError:
            await JSONResponse({"accepted": False, "detail": "Invalid Content-Length"}, 400)(scope, receive, send)
            return
        if length > self.max_bytes:
            await JSONResponse({"accepted": False, "detail": "Upload too large"}, 413)(scope, receive, send)
            return
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await JSONResponse({"accepted": False, "detail": "Upload too large"}, 413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        buffered = iter(messages)

        async def replay() -> dict:
            message = next(buffered, None)
            return message if message is not None else await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store = EvidenceStore(settings.pipeline_data_dir, settings=settings)
        initial_maintenance = await asyncio.to_thread(store.initialize)
        logger.info(
            "evidence_storage_initialized ready=%s used_percent=%.2f free_bytes=%s "
            "expired_removed=%s pressure_removed=%s temporary_removed=%s",
            initial_maintenance.status.ready, initial_maintenance.status.used_percent,
            initial_maintenance.status.free_bytes, initial_maintenance.expired_events_removed,
            initial_maintenance.pressure_events_removed, initial_maintenance.temporary_paths_removed,
        )
        alarm_deduplicator = AlarmDeduplicator(
            store.root, settings.alarm_stream_cooldown_seconds,
            retention_days=settings.alarm_state_retention_days,
            future_tolerance_seconds=settings.max_capture_clock_skew_seconds,
        )
        await asyncio.to_thread(alarm_deduplicator.initialize)
        connections = settings.sam3_concurrency + settings.llm_concurrency + 4
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
        ) as http:
            sam3 = Sam3Client(http, settings)
            llm = LLMClient(http, settings)
            pipeline = Pipeline(settings, sam3, llm, store, alarm_deduplicator)
            monitor = RuntimeMonitor(settings, pipeline, sam3, llm, store)
            application.state.pipeline = pipeline
            application.state.monitor = monitor
            pipeline.start()
            await monitor.start()
            try:
                yield
            finally:
                await monitor.stop()
                await pipeline.stop()

    application = FastAPI(title="SAM3 + LLM Fire/Smoke Confirmation", version=__version__, lifespan=lifespan)
    application.add_middleware(UploadLimitMiddleware, max_bytes=settings.max_image_bytes + 64 * 1024)
    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    if origins:
        application.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["POST"],
            allow_headers=["Content-Type", "X-API-Key"], allow_credentials=False,
        )

    @application.post("/v1/frames", status_code=202)
    async def submit_frame(
        request: Request,
        image: Annotated[UploadFile, File()],
        machine_id: Annotated[str, Form(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
        stream_id: Annotated[str, Form(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
        stream_name: Annotated[str | None, Form(max_length=256)] = None,
        captured_at: Annotated[AwareDatetime | None, Form()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ):
        if settings.pipeline_api_key and not hmac.compare_digest(
            (x_api_key or "").encode(), settings.pipeline_api_key.encode()
        ):
            raise HTTPException(401, "Invalid X-API-Key")
        pipeline: Pipeline = request.app.state.pipeline
        monitor: RuntimeMonitor = request.app.state.monitor
        if not pipeline.healthy or not monitor.ready:
            return JSONResponse({
                "accepted": False,
                "detail": "Service or required dependency is unavailable",
            }, 503)
        if pipeline.sam3_queue.full():
            pipeline.counts["rejected_full"] += 1
            return JSONResponse({"accepted": False, "detail": "Queue is full"}, 429)
        received_at = request.state.frame_received_at
        received_monotonic = request.state.frame_received_monotonic
        data = await image.read(settings.max_image_bytes + 1)
        try:
            prepared = await asyncio.to_thread(
                prepare_image, data, settings.max_image_bytes, settings.max_image_pixels,
            )
        except ImageTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except InvalidImage as exc:
            raise HTTPException(400, str(exc)) from exc
        if captured_at is not None:
            skew = (as_utc(received_at) - as_utc(captured_at)).total_seconds()
            if abs(skew) > settings.max_capture_clock_skew_seconds:
                pipeline.counts["capture_clock_skew_warning"] += 1
                logger.warning(
                    "capture_clock_skew machine=%s stream=%s skew_seconds=%.3f",
                    machine_id, stream_id, skew,
                )
        frame = Frame(prepared, machine_id, stream_id, stream_name,
                      captured_at, received_at, received_monotonic)
        if not pipeline.healthy or not monitor.ready:
            return JSONResponse({
                "accepted": False,
                "detail": "Service or required dependency is unavailable",
            }, 503)
        try:
            pipeline.submit(frame)
        except asyncio.QueueFull:
            pipeline.counts["rejected_full"] += 1
            return JSONResponse({"accepted": False, "detail": "Queue is full"}, 429)
        return {"accepted": True}

    def health_response(request: Request, *, require_ready: bool) -> JSONResponse:
        monitor: RuntimeMonitor = request.app.state.monitor
        payload = monitor.snapshot()
        available = monitor.ready if require_ready else monitor.live
        return JSONResponse(payload, status_code=200 if available else 503)

    @application.get("/health/live")
    async def live(request: Request):
        return health_response(request, require_ready=False)

    @application.get("/health/ready")
    async def ready(request: Request):
        return health_response(request, require_ready=True)

    @application.get("/health")
    async def health(request: Request):
        return health_response(request, require_ready=True)

    @application.get("/status")
    async def status(request: Request):
        monitor: RuntimeMonitor = request.app.state.monitor
        return JSONResponse(monitor.snapshot())

    return application


settings = Settings.from_env()
app = create_app(settings)


if __name__ == "__main__":
    import uvicorn

    from app.logging_setup import configured_logging

    with configured_logging(settings):
        uvicorn.run(app, host=settings.pipeline_host, port=settings.pipeline_port, workers=1,
                    log_config=None, limit_concurrency=64, timeout_keep_alive=5,
                    timeout_graceful_shutdown=10)
