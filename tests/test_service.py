import asyncio
import base64
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image
from pydantic import ValidationError

from app.clients import parse_detections
from app.config import DEFAULT_LLM_SYSTEM_PROMPT, DEFAULT_LLM_USER_PROMPT, Settings
from app.domain import Candidate, Frame, LLMReply, LLMVerdict, PROMPT_VERSION
from app.storage import EvidenceStore
from app.images import ImageTooLarge, InvalidImage, prepare_image
from app.main import create_app


def picture(fmt="PNG", orientation=None):
    image = Image.new("RGB", (160, 120), (25, 30, 35))
    output = io.BytesIO()
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(output, format=fmt, **options)
    return output.getvalue()


def frame(data=None):
    data = picture() if data is None else data
    return Frame(prepare_image(data, 8 * 1024 * 1024, 16_000_000),
                 "machine-1", "stream_1", "test camera", None,
                 datetime.now(timezone.utc), time.monotonic())


BOXES = [
    {"label": "fire", "score": 0.87, "box": [10, 20, 70, 100]},
    {"label": "smoke", "score": 0.72, "box": [90, 30, 150, 100]},
]


def reply(result="fire", *, content=None, finish="stop"):
    return httpx.Response(200, json={"choices": [{
        "message": {"content": content if content is not None else json.dumps(
            {"result": result, "reason": "visible evidence"})},
        "finish_reason": finish,
    }]})


class Stub:
    def __init__(self, boxes=None, llm_result="fire"):
        self.boxes = BOXES if boxes is None else boxes
        self.llm_result = llm_result
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        if request.url.path == "/predict/file":
            return httpx.Response(200, json={"results": self.boxes})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path == "/v1/chat/completions":
            return reply(self.llm_result)
        raise AssertionError(f"Unexpected upstream request: {request.url}")


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = picture()

    async def asyncTearDown(self):
        self.temp.cleanup()

    @asynccontextmanager
    async def running(self, handler=None, **overrides):
        config = {
            "pipeline_data_dir": self.root,
            "llm_model": "test-model",
            "sam3_timeout_seconds": 3,
            "llm_timeout_seconds": 3,
            "shutdown_timeout_seconds": 0.3,
        }
        config.update(overrides)
        app = create_app(Settings(**config), transport=httpx.MockTransport(handler or Stub()))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://pipeline") as client:
                yield app, client

    async def upload(self, client, *, data=None, fields=None, headers=None):
        form = {"machine_id": "machine-1", "stream_id": "stream_1",
                "stream_name": "摄像头一", "captured_at": "2026-08-28T10:00:00+08:00"}
        if fields:
            form.update(fields)
        return await client.post("/v1/frames", data=form, headers=headers,
                                 files={"image": ("input.png", self.data if data is None else data, "image/png")})

    async def test_confirmed_images_are_saved_and_alarm_called_after_files_exist(self):
        for result in ("fire", "smoke", "fire_smoke"):
            with self.subTest(result=result):
                stub = Stub(llm_result=result)

                async def check_alarm(event):
                    self.assertTrue(event.original.is_file())
                    self.assertTrue(event.annotated.is_file())
                    self.assertTrue(event.metadata.is_file())

                with patch("app.alarm.send_alarm", new=AsyncMock(side_effect=check_alarm)) as alarm:
                    async with self.running(stub) as (app, client):
                        response = await self.upload(client)
                        self.assertEqual(response.status_code, 202)
                        self.assertEqual(response.json(), {"accepted": True})
                        await asyncio.wait_for(app.state.pipeline.drain(), 3)
                        alarm.assert_awaited_once()
                        event = alarm.await_args.args[0]
                        self.assertEqual(event.original.read_bytes(), self.data)
                        metadata = json.loads(event.metadata.read_text(encoding="utf-8"))
                        self.assertEqual(metadata["llm"]["result"], result)
                        self.assertEqual(metadata["llm"]["model"], "test-model")
                        self.assertEqual(metadata["llm"]["prompt_version"], PROMPT_VERSION)
                        self.assertEqual(metadata["sam3"]["class_names"], ["fire", "smoke"])
                        self.assertEqual(metadata["stream_name"], "摄像头一")
                        self.assertEqual(metadata["captured_at"], "2026-08-28T10:00:00+08:00")
                        self.assertEqual(len(metadata["sam3"]["detections"]), 2)
                        self.assertEqual(metadata["alarm"]["status"], "not_configured")
                        with Image.open(event.annotated) as annotated:
                            self.assertEqual(annotated.size, (160, 120))
                            self.assertGreater(annotated.getpixel((10, 60))[0], 100)
                        sam_request, llm_request = stub.requests
                        self.assertIn(b'name="class_names"\r\n\r\nfire,smoke', sam_request.content)
                        self.assertIn(b'name="return_mask"\r\n\r\nfalse', sam_request.content)
                        payload = json.loads(llm_request.content)
                        url = payload["messages"][1]["content"][0]["image_url"]["url"]
                        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.data)
                        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                        self.assertEqual(payload["temperature"], 0)
                        self.assertEqual(payload["messages"][0]["content"], DEFAULT_LLM_SYSTEM_PROMPT)
                        self.assertEqual(payload["messages"][1]["content"][1]["text"], DEFAULT_LLM_USER_PROMPT)

    async def test_custom_words_and_prompts_reach_models_and_saved_evidence(self):
        configured_env = {
            "SAM3_CLASS_NAMES": " flame, dense smoke, flame ",
            "LLM_SYSTEM_PROMPT": r"你是现场图像审核助手。\n不要猜测。",
            "LLM_USER_PROMPT": r'识别火焰或烟雾。\n只输出{"result":"fire|smoke|fire_smoke|none|uncertain","reason":"可见依据"}',
        }
        with patch.dict(os.environ, configured_env, clear=True):
            settings = Settings.from_env()
        stub = Stub(boxes=[
            {**BOXES[0], "label": "flame"},
            {**BOXES[1], "label": "dense smoke"},
            BOXES[0],  # A label not in the configured list must still be ignored.
            {**BOXES[0], "label": "flame", "score": 0.3},
        ], llm_result="fire_smoke")
        with patch("app.alarm.send_alarm", new=AsyncMock()) as alarm:
            async with self.running(
                stub, sam3_class_names=settings.sam3_class_names,
                llm_system_prompt=settings.llm_system_prompt,
                llm_user_prompt=settings.llm_user_prompt,
            ) as (app, client):
                self.assertEqual((await self.upload(client)).status_code, 202)
                await app.state.pipeline.drain()
                alarm.assert_awaited_once()
                sam_request, llm_request = stub.requests
                self.assertIn(b'name="class_names"\r\n\r\nflame,dense smoke', sam_request.content)
                self.assertIn(b'name="confidence"\r\n\r\n0.3', sam_request.content)
                self.assertIn(b'name="return_mask"\r\n\r\nfalse', sam_request.content)
                payload = json.loads(llm_request.content)
                self.assertEqual(payload["messages"][0]["content"], "你是现场图像审核助手。\n不要猜测。")
                expected_user = configured_env["LLM_USER_PROMPT"].replace("\\n", "\n")
                self.assertEqual(payload["messages"][1]["content"][1]["text"], expected_user)
                event = alarm.await_args.args[0]
                metadata = json.loads(event.metadata.read_text(encoding="utf-8"))
                self.assertEqual(metadata["sam3"]["class_names"], ["flame", "dense smoke"])
                self.assertEqual(
                    [item["label"] for item in metadata["sam3"]["detections"]],
                    ["flame", "dense smoke"],
                )
                self.assertEqual(metadata["llm"]["system_prompt"], payload["messages"][0]["content"])
                self.assertEqual(metadata["llm"]["user_prompt"], expected_user)
                self.assertEqual(metadata["llm"]["prompt_version"], settings.llm_prompt_version)
                self.assertNotEqual(metadata["llm"]["prompt_version"], PROMPT_VERSION)
                self.assertEqual(metadata["llm"]["result"], "fire_smoke")
                self.assertEqual(event.original.read_bytes(), self.data)
                with Image.open(event.annotated) as annotated:
                    self.assertGreater(annotated.getpixel((10, 60))[2], 150)

    async def test_unrequested_default_labels_do_not_trigger_llm(self):
        stub = Stub()
        async with self.running(stub, sam3_class_names="flame,dense smoke") as (app, client):
            await self.upload(client)
            await app.state.pipeline.drain()
            self.assertEqual([r.url.path for r in stub.requests], ["/predict/file"])
            self.assertEqual(app.state.pipeline.counts["sam3_negative"], 1)
            self.assertEqual(list(self.root.rglob("metadata.json")), [])

    async def test_custom_prompt_does_not_change_result_schema_or_enable_retries(self):
        stub = Stub(llm_result="person")
        with patch("app.alarm.send_alarm", new=AsyncMock()) as alarm:
            async with self.running(stub, llm_user_prompt='只输出{"result":"person"}') as (app, client):
                await self.upload(client)
                await app.state.pipeline.drain()
                self.assertEqual([r.url.path for r in stub.requests], ["/predict/file", "/v1/chat/completions"])
                self.assertEqual(app.state.pipeline.counts["llm_failed"], 1)
                alarm.assert_not_awaited()
                self.assertEqual(list(self.root.rglob("metadata.json")), [])

    async def test_threshold_is_strict_and_no_candidate_never_calls_llm(self):
        stub = Stub(boxes=[
            {"label": "fire", "score": 0.3, "box": [10, 20, 70, 100]},
            {"label": "smoke", "score": 0.2, "box": [90, 30, 150, 100]},
        ])
        async with self.running(stub) as (app, client):
            self.assertEqual((await self.upload(client)).status_code, 202)
            await app.state.pipeline.drain()
            self.assertEqual(len(stub.requests), 1)
            self.assertEqual(app.state.pipeline.counts["sam3_negative"], 1)
            self.assertEqual(list(self.root.rglob("metadata.json")), [])

    async def test_negative_uncertain_invalid_and_failed_llm_are_skipped_without_retry(self):
        cases = [
            ("none", lambda: reply("none")),
            ("uncertain", lambda: reply("uncertain")),
            ("http_error", lambda: httpx.Response(503)),
            ("bad_json", lambda: reply(content='not JSON')),
            ("markdown", lambda: reply(content='{"result":"none"}\n{"result":"fire"}')),
            ("wrong_enum", lambda: reply(content='{"result":"yes"}')),
            ("duplicate_keys", lambda: reply(content='{"result":"none","result":"fire"}')),
            ("truncated", lambda: reply(content='{"result":"fire"}', finish="length")),
            ("empty_choices", lambda: httpx.Response(200, json={"choices": []})),
        ]
        for label, response in cases:
            with self.subTest(label=label):
                calls = []

                async def handler(request):
                    calls.append(request.url.path)
                    return httpx.Response(200, json={"results": BOXES}) if request.url.path == "/predict/file" else response()

                with patch("app.alarm.send_alarm", new=AsyncMock()) as alarm:
                    async with self.running(handler) as (app, client):
                        await self.upload(client)
                        await app.state.pipeline.drain()
                        self.assertEqual(calls, ["/predict/file", "/v1/chat/completions"])
                        alarm.assert_not_awaited()
                        self.assertEqual(list(self.root.rglob("metadata.json")), [])

    async def test_llm_deadline_skips_frame_and_worker_continues(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            if request.url.path == "/predict/file":
                return httpx.Response(200, json={"results": BOXES})
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.2)
            return reply()

        async with self.running(handler, llm_timeout_seconds=0.03, llm_concurrency=1) as (app, client):
            await self.upload(client)
            await app.state.pipeline.drain()
            await self.upload(client)
            await app.state.pipeline.drain()
            self.assertEqual(calls, 2)
            self.assertEqual(app.state.pipeline.counts["llm_failed"], 1)
            self.assertEqual(app.state.pipeline.counts["saved"], 1)

    async def test_upstream_failure_logs_status_without_credentials_or_response_body(self):
        async def handler(request):
            if request.url.path == "/predict/file":
                return httpx.Response(200, json={"results": BOXES})
            return httpx.Response(401, text="private-model-response")

        async with self.running(handler, llm_api_key="private-api-key") as (app, client):
            with self.assertLogs("app.pipeline", level="WARNING") as captured:
                await self.upload(client)
                await app.state.pipeline.drain()
            text = "\n".join(captured.output)
            self.assertIn("llm_failed", text)
            self.assertIn("machine=machine-1 stream=stream_1", text)
            self.assertIn("http_status=401", text)
            self.assertIn("elapsed_ms=", text)
            self.assertNotIn("private-api-key", text)
            self.assertNotIn("private-model-response", text)
            self.assertEqual(app.state.pipeline.counts["llm_failed"], 1)

    async def test_sam3_http_or_payload_failure_does_not_call_llm(self):
        for response in (httpx.Response(500), httpx.Response(200, json={"success": False}),
                         httpx.Response(200, json={"results": "invalid"})):
            with self.subTest(response=response):
                requests = []

                async def handler(request):
                    requests.append(request)
                    return response

                async with self.running(handler) as (app, client):
                    await self.upload(client)
                    await app.state.pipeline.drain()
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(app.state.pipeline.counts["sam3_failed"], 1)

    async def test_sam3_total_deadline_skips_without_llm_or_retry(self):
        calls = []

        async def handler(request):
            calls.append(request.url.path)
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={"results": BOXES})

        async with self.running(handler, sam3_timeout_seconds=0.02) as (app, client):
            await self.upload(client)
            await app.state.pipeline.drain()
            self.assertEqual(calls, ["/predict/file"])
            self.assertEqual(app.state.pipeline.counts["sam3_failed"], 1)

    async def test_auto_model_discovery_is_cached(self):
        stub = Stub(llm_result="none")
        async with self.running(stub, llm_model="") as (app, client):
            await asyncio.gather(*(self.upload(client) for _ in range(4)))
            await app.state.pipeline.drain()
            self.assertEqual(sum(r.url.path == "/v1/models" for r in stub.requests), 1)
            self.assertEqual(sum(r.url.path == "/v1/chat/completions" for r in stub.requests), 4)

    async def test_ambiguous_model_discovery_skips_without_completion_request(self):
        calls = []

        async def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/predict/file":
                return httpx.Response(200, json={"results": BOXES})
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

        async with self.running(handler, llm_model="") as (app, client):
            await self.upload(client)
            await app.state.pipeline.drain()
            self.assertEqual(calls, ["/predict/file", "/v1/models"])
            self.assertEqual(app.state.pipeline.counts["llm_failed"], 1)

    async def test_202_does_not_wait_for_inference_and_full_queue_returns_429(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"results": []})

        async with self.running(handler, sam3_concurrency=1, sam3_queue_size=1) as (app, client):
            try:
                response = await asyncio.wait_for(self.upload(client), 2)
                self.assertEqual(response.json(), {"accepted": True})
                await asyncio.wait_for(entered.wait(), 2)
                self.assertEqual((await self.upload(client)).status_code, 202)
                rejected = await self.upload(client)
                self.assertEqual(rejected.status_code, 429)
                self.assertFalse(rejected.json()["accepted"])
                self.assertEqual(app.state.pipeline.counts["accepted"], 2)
            finally:
                release.set()
            await app.state.pipeline.drain()

    async def test_llm_backpressure_bounds_both_queues_without_dropping_candidates(self):
        entered, release = asyncio.Event(), asyncio.Event()
        sam_calls = 0

        async def handler(request):
            nonlocal sam_calls
            if request.url.path == "/predict/file":
                sam_calls += 1
                return httpx.Response(200, json={"results": BOXES})
            entered.set()
            await release.wait()
            return reply("none")

        async with self.running(handler, sam3_concurrency=1, llm_concurrency=1,
                                sam3_queue_size=1, llm_queue_size=1) as (app, client):
            pipeline = app.state.pipeline
            try:
                await self.upload(client)
                await entered.wait()
                await self.upload(client)
                await until(lambda: pipeline.llm_queue.full())
                await self.upload(client)
                await until(lambda: sam_calls == 3)
                await self.upload(client)
                self.assertEqual((await self.upload(client)).status_code, 429)
                self.assertEqual(pipeline.sam3_queue.qsize(), 1)
                self.assertEqual(pipeline.llm_queue.qsize(), 1)
            finally:
                release.set()
            await pipeline.drain()
            self.assertEqual(pipeline.counts["llm_none"], 4)
            self.assertEqual(pipeline.counts["accepted"], 4)

    async def test_independent_worker_pools_enforce_concurrency_during_burst(self):
        gates = {"sam": asyncio.Event(), "llm": asyncio.Event()}
        active, peak = {"sam": 0, "llm": 0}, {"sam": 0, "llm": 0}

        async def handler(request):
            stage = "sam" if request.url.path == "/predict/file" else "llm"
            active[stage] += 1
            peak[stage] = max(peak[stage], active[stage])
            try:
                await gates[stage].wait()
            finally:
                active[stage] -= 1
            return httpx.Response(200, json={"results": BOXES}) if stage == "sam" else reply("none")

        async with self.running(handler, sam3_concurrency=2, llm_concurrency=2) as (app, client):
            try:
                responses = await asyncio.gather(*(self.upload(client) for _ in range(6)))
                self.assertTrue(all(r.status_code == 202 for r in responses))
                await until(lambda: active["sam"] == 2)
                gates["sam"].set()
                await until(lambda: active["llm"] == 2)
                self.assertEqual(peak, {"sam": 2, "llm": 2})
            finally:
                for gate in gates.values():
                    gate.set()
            await app.state.pipeline.drain()
            self.assertEqual(app.state.pipeline.counts["llm_none"], 6)

    async def test_save_failure_does_not_invoke_alarm_and_worker_survives(self):
        with patch("app.alarm.send_alarm", new=AsyncMock()) as alarm:
            async with self.running() as (app, client):
                with patch.object(app.state.pipeline.store, "save", side_effect=OSError("disk full")):
                    await self.upload(client)
                    await app.state.pipeline.drain()
                    alarm.assert_not_awaited()
                await self.upload(client)
                await app.state.pipeline.drain()
                self.assertEqual(app.state.pipeline.counts["save_failed"], 1)
                alarm.assert_awaited_once()

    async def test_alarm_failure_keeps_saved_evidence(self):
        with patch("app.alarm.send_alarm", new=AsyncMock(side_effect=OSError("unavailable"))):
            async with self.running() as (app, client):
                await self.upload(client)
                await app.state.pipeline.drain()
                self.assertEqual(app.state.pipeline.counts["alarm_failed"], 1)
                self.assertEqual(len(list(self.root.rglob("metadata.json"))), 1)

    async def test_invalid_uploads_and_identifiers_never_reach_models(self):
        stub = Stub()
        async with self.running(stub) as (_, client):
            self.assertEqual((await self.upload(client, data=b"not an image")).status_code, 400)
            self.assertEqual((await self.upload(client, fields={"stream_id": "../../escape"})).status_code, 422)
            self.assertEqual((await self.upload(client, fields={"machine_id": "a/b"})).status_code, 422)
            self.assertEqual((await self.upload(client, fields={"captured_at": "2026-08-28T10:00:00"})).status_code, 422)
            self.assertEqual(stub.requests, [])

    async def test_file_size_pixel_limit_and_chunked_body_limit(self):
        async with self.running(max_image_bytes=1024, max_image_pixels=100) as (_, client):
            self.assertEqual((await self.upload(client, data=b"x" * 1025)).status_code, 413)
            self.assertEqual((await self.upload(client)).status_code, 413)

            async def chunks():
                for _ in range(80):
                    yield b"x" * 1024

            response = await client.post("/v1/frames", content=chunks(),
                                         headers={"Content-Type": "multipart/form-data; boundary=test"})
            self.assertEqual(response.status_code, 413)

    async def test_optional_auth_cors_and_no_task_api(self):
        async with self.running(Stub(boxes=[]), pipeline_api_key="test-key",
                                cors_origins="http://frontend:3000") as (app, client):
            self.assertEqual((await self.upload(client)).status_code, 401)
            accepted = await self.upload(client, headers={"X-API-Key": "test-key",
                                                         "Origin": "http://frontend:3000"})
            self.assertEqual(accepted.status_code, 202)
            self.assertEqual(accepted.headers["access-control-allow-origin"], "http://frontend:3000")
            await app.state.pipeline.drain()
            self.assertEqual((await client.get("/v1/tasks/anything")).status_code, 404)
            health = (await client.get("/health")).json()
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["upstreams"], "not_checked")
            self.assertEqual(health["counts"]["accepted"], 1)

    async def test_shutdown_has_deadline_and_cancels_inflight_work(self):
        entered = asyncio.Event()

        async def handler(request):
            entered.set()
            await asyncio.Event().wait()

        async with self.running(handler, shutdown_timeout_seconds=0.02) as (app, client):
            await self.upload(client)
            await entered.wait()
            await asyncio.wait_for(app.state.pipeline.stop(), 1)
            self.assertFalse(app.state.pipeline.healthy)
            self.assertTrue(all(worker.done() for worker in app.state.pipeline.workers))
            self.assertEqual((await client.get("/health")).status_code, 503)
            self.assertEqual((await self.upload(client)).status_code, 503)


class ValidationTests(unittest.TestCase):
    def test_partial_evidence_is_cleaned_when_annotation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            store.initialize()
            source = frame()
            candidate = Candidate(source, parse_detections({"results": BOXES}, source), 1.0)
            result = LLMReply(LLMVerdict(result="fire"), '{"result":"fire"}', "test-model")
            with patch.object(store, "_annotate", side_effect=OSError("write failure")):
                with self.assertRaises(OSError):
                    store.save(candidate, result, 1.0)
            self.assertEqual(list(Path(directory).rglob("original.png")), [])
            self.assertEqual(list(Path(directory).rglob(".tmp-*")), [])

    def test_strict_score_and_box_validation(self):
        source = frame()
        valid = parse_detections({"results": BOXES}, source)
        self.assertEqual(len(valid), 2)
        for score in (float("nan"), float("inf"), True, "0.9", 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_detections({"results": [{**BOXES[0], "score": score}]}, source)
        with self.assertRaises(ValueError):
            parse_detections({"results": [{**BOXES[0], "box": [60, 20, 10, 40]}]}, source)

    def test_exif_is_normalized_for_both_models_but_original_is_preserved(self):
        original = picture("JPEG", orientation=6)
        prepared = prepare_image(original, 8 * 1024 * 1024, 16_000_000)
        self.assertEqual(prepared.original, original)
        self.assertTrue(prepared.orientation_normalized)
        self.assertEqual((prepared.width, prepared.height), (120, 160))
        self.assertEqual(prepared.inference_mime, "image/png")
        with Image.open(io.BytesIO(prepared.inference)) as normalized:
            self.assertEqual(normalized.size, (120, 160))
            self.assertIn(normalized.getexif().get(274, 1), (None, 1))

    def test_empty_truncated_or_excessive_image_is_rejected(self):
        for value in (b"", b"random", picture()[:40]):
            with self.subTest(value=value[:10]), self.assertRaises(InvalidImage):
                prepare_image(value, 100_000, 16_000_000)
        with self.assertRaises(ImageTooLarge):
            prepare_image(picture(), 100_000, 10)

    def test_unknown_and_non_string_conclusions_cannot_be_confirmed(self):
        for result in (True, 1, "yes", "fire and smoke", None):
            with self.subTest(result=result), self.assertRaises(ValidationError):
                LLMVerdict.model_validate({"result": result})

    def test_invalid_detection_words_and_empty_prompts_fail_startup_validation(self):
        for value in ("", " ", ",", "fire,,smoke", "fire,", "fire,\nsmoke"):
            with self.subTest(class_names=value), self.assertRaises(ValidationError):
                Settings(sam3_class_names=value)
        for field in ("llm_system_prompt", "llm_user_prompt"):
            for value in ("", " \t ", r"\n"):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    Settings(**{field: value})

    def test_prompt_versions_follow_effective_content_and_legacy_env_uses_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = Settings.from_env()
        self.assertEqual(defaults.sam3_classes, ("fire", "smoke"))
        self.assertEqual(defaults.llm_system_prompt, DEFAULT_LLM_SYSTEM_PROMPT)
        self.assertEqual(defaults.llm_user_prompt, DEFAULT_LLM_USER_PROMPT)
        self.assertEqual(defaults.llm_prompt_version, PROMPT_VERSION)
        same_defaults = Settings(llm_user_prompt=DEFAULT_LLM_USER_PROMPT.replace("\n", r"\n"))
        self.assertEqual(same_defaults.llm_prompt_version, PROMPT_VERSION)
        custom = Settings(llm_user_prompt=r"自定义第一行\n第二行")
        equivalent = Settings(llm_user_prompt="自定义第一行\n第二行")
        self.assertEqual(custom.llm_prompt_version, equivalent.llm_prompt_version)
        self.assertTrue(custom.llm_prompt_version.startswith("sha256:"))
        for override in ({"llm_system_prompt": "修改系统指令"}, {"llm_user_prompt": "修改用户指令"}):
            changed = Settings(**{**custom.model_dump(include={"llm_system_prompt", "llm_user_prompt"}), **override})
            self.assertNotEqual(changed.llm_prompt_version, custom.llm_prompt_version)

    def test_unbounded_queue_or_nonfinite_timeout_cannot_be_configured(self):
        for setting in ({"sam3_queue_size": 0}, {"llm_concurrency": 0},
                        {"llm_timeout_seconds": float("inf")}, {"sam3_url": "not a URL"},
                        {"log_retention_days": 0}, {"log_retention_days": 31}, {"log_level": "INVALID"}):
            with self.subTest(setting=setting), self.assertRaises(ValidationError):
                Settings(**setting)


if __name__ == "__main__":
    unittest.main()
