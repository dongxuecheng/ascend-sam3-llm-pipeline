import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime
from starlette.types import ASGIApp, Receive, Scope, Send

from app.clients import LLMClient, Sam3Client
from app.config import Settings
from app.domain import Frame
from app.images import ImageTooLarge, InvalidImage, prepare_image
from app.pipeline import Pipeline
from app.storage import EvidenceStore


class UploadLimitMiddleware:
    """Bound the whole multipart body before the framework spools uploaded files."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/v1/frames":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["frame_received_at"] = datetime.now(timezone.utc)
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
        store = EvidenceStore(settings.pipeline_data_dir)
        await asyncio.to_thread(store.initialize)
        connections = settings.sam3_concurrency + settings.llm_concurrency + 2
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
        ) as http:
            pipeline = Pipeline(settings, Sam3Client(http, settings), LLMClient(http, settings), store)
            application.state.pipeline = pipeline
            pipeline.start()
            try:
                yield
            finally:
                await pipeline.stop()

    application = FastAPI(title="SAM3 + LLM Fire/Smoke Confirmation", version="1.0.0", lifespan=lifespan)
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
        if not pipeline.healthy:
            return JSONResponse({"accepted": False, "detail": "Service is stopping or unavailable"}, 503)
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
        frame = Frame(prepared, machine_id, stream_id, stream_name,
                      captured_at, received_at, received_monotonic)
        # Recheck after image decoding: other requests may have filled the queue.
        if not pipeline.healthy:
            return JSONResponse({"accepted": False, "detail": "Service is stopping or unavailable"}, 503)
        try:
            pipeline.submit(frame)
        except asyncio.QueueFull:
            pipeline.counts["rejected_full"] += 1
            return JSONResponse({"accepted": False, "detail": "Queue is full"}, 429)
        return {"accepted": True}

    @application.get("/health")
    async def health(request: Request):
        pipeline: Pipeline = request.app.state.pipeline
        return JSONResponse({
            "status": "ok" if pipeline.healthy else "unavailable",
            "upstreams": "not_checked",
            "queues": {"sam3": pipeline.sam3_queue.qsize(), "llm": pipeline.llm_queue.qsize()},
            "counts": dict(pipeline.counts),
        }, status_code=200 if pipeline.healthy else 503)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # A single API process owns the global queues; never use multiple Uvicorn workers.
    uvicorn.run(app, host=settings.pipeline_host, port=settings.pipeline_port, workers=1,
                limit_concurrency=64, timeout_keep_alive=5, timeout_graceful_shutdown=10)
