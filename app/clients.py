"""HTTP adapters. Each frame is attempted once; no automatic retries."""

import asyncio
import base64
import json
import math
from typing import Any

import httpx

from app.config import Settings
from app.domain import (
    DEFAULT_SAM3_CONFIDENCE_THRESHOLD, Detection, Frame, LLMReply, LLMVerdict,
    SAM3_CLASSES,
)


class UpstreamError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise UpstreamError("LLM returned duplicate JSON fields")
        result[key] = value
    return result


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpstreamError("SAM3 returned a non-numeric score or coordinate")
    result = float(value)
    if not math.isfinite(result):
        raise UpstreamError("SAM3 returned a non-finite number")
    return result


def parse_detections(
    payload: Any, frame: Frame, *, class_names: tuple[str, ...] = SAM3_CLASSES,
    confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
) -> tuple[Detection, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise UpstreamError("SAM3 response has no results array")
    detections = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            raise UpstreamError("SAM3 returned an invalid detection")
        if item.get("label") not in class_names:
            continue
        score = _number(item.get("score"))
        if not 0 <= score <= 1:
            raise UpstreamError("SAM3 score is outside [0, 1]")
        if score <= confidence_threshold:
            continue
        raw = item.get("box")
        if not isinstance(raw, list) or len(raw) != 4:
            raise UpstreamError("SAM3 box must contain four pixel coordinates")
        x1, y1, x2, y2 = map(_number, raw)
        width, height = frame.image.width, frame.image.height
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x1 >= x2 or y1 >= y2:
            raise UpstreamError("SAM3 returned an empty or inverted box")
        detections.append(Detection(item["label"], score, (x1, y1, x2, y2)))
    return tuple(detections)


class Sam3Client:
    def __init__(self, http: httpx.AsyncClient, settings: Settings):
        self.http, self.settings = http, settings

    async def probe(self) -> None:
        url = httpx.URL(str(self.settings.sam3_url)).copy_with(path="/health", query=None)
        async with asyncio.timeout(self.settings.upstream_health_probe_timeout_seconds):
            response = await self.http.get(url)
            response.raise_for_status()

    async def detect(self, frame: Frame) -> tuple[Detection, ...]:
        async with asyncio.timeout(self.settings.sam3_timeout_seconds):
            response = await self.http.post(
                str(self.settings.sam3_url),
                data={"class_names": self.settings.sam3_class_names,
                      "confidence": str(self.settings.sam3_confidence_threshold),
                      "return_mask": "false"},
                files={"image": ("frame", frame.image.inference, frame.image.inference_mime)},
                timeout=self.settings.sam3_timeout_seconds,
            )
            response.raise_for_status()
            return parse_detections(
                response.json(), frame, class_names=self.settings.sam3_classes,
                confidence_threshold=self.settings.sam3_confidence_threshold,
            )


class LLMClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings):
        self.http, self.settings = http, settings
        self.model = settings.llm_model.strip()
        self.model_lock = asyncio.Lock()
        self.base_url = str(settings.llm_base_url).rstrip("/")
        self.headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}

    async def _model_name(self) -> str:
        if not self.model:
            async with self.model_lock:
                if not self.model:
                    response = await self.http.get(self.base_url + "/models", headers=self.headers)
                    response.raise_for_status()
                    payload = response.json()
                    models = payload.get("data") if isinstance(payload, dict) else None
                    if not isinstance(models, list) or len(models) != 1:
                        raise UpstreamError("Set LLM_MODEL when /models does not return exactly one model")
                    model = models[0].get("id") if isinstance(models[0], dict) else None
                    if not isinstance(model, str) or not model.strip():
                        raise UpstreamError("LLM returned an invalid model id")
                    self.model = model
        return self.model

    async def probe(self) -> None:
        url = httpx.URL(self.base_url).copy_with(path="/health", query=None)
        async with asyncio.timeout(self.settings.upstream_health_probe_timeout_seconds):
            response = await self.http.get(url, headers=self.headers)
            response.raise_for_status()
            await self._model_name()

    async def confirm(self, frame: Frame) -> LLMReply:
        async with asyncio.timeout(self.settings.llm_timeout_seconds):
            model = await self._model_name()
            encoded = base64.b64encode(frame.image.inference).decode("ascii")
            response_schema = LLMVerdict.model_json_schema()
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.settings.llm_system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:{frame.image.inference_mime};base64,{encoded}"
                        }},
                        {"type": "text", "text": self.settings.llm_user_prompt},
                    ]},
                ],
                "temperature": 0,
                "max_tokens": self.settings.llm_max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fire_smoke_verdict",
                        "schema": response_schema,
                        "strict": True,
                    },
                },
            }
            response = await self.http.post(
                self.base_url + "/chat/completions", json=payload,
                headers=self.headers, timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            try:
                choice = body["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") != "stop":
                    raise UpstreamError("LLM response was truncated or did not finish normally")
                if not isinstance(content, str) or not content.strip():
                    raise UpstreamError("LLM returned no text")
            except (KeyError, IndexError, TypeError) as exc:
                raise UpstreamError("Invalid LLM response structure") from exc
            verdict = LLMVerdict.model_validate(json.loads(content, object_pairs_hook=_unique_json_object))
            return LLMReply(verdict, content, model)
