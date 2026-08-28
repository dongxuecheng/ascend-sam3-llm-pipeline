"""Run the real HTTP entrypoint against local model stubs; no NPU or Docker needed."""
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


class MockModels(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self.send_json({"data": [{"id": "http-smoke-model"}]})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/predict/file":
            assert b'name="return_mask"\r\n\r\nfalse' in body
            assert b'name="class_names"\r\n\r\nfire,smoke' in body
            self.send_json({"results": [
                {"label": "fire", "score": 0.87, "box": [100, 160, 300, 320]},
                {"label": "smoke", "score": 0.72, "box": [330, 40, 530, 220]},
            ]})
        elif self.path == "/v1/chat/completions":
            payload = json.loads(body)
            assert payload["chat_template_kwargs"]["enable_thinking"] is False
            self.send_json({"choices": [{"message": {"content": json.dumps({
                "result": "fire_smoke", "reason": "Synthetic test response; not real model inference",
            })}, "finish_reason": "stop"}]})
        else:
            self.send_error(404)


def main():
    output = ROOT / ".test-artifacts" / ("http-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True)
    original = Image.new("RGB", (640, 360), "#18212f")
    draw = ImageDraw.Draw(original)
    draw.rectangle((100, 160, 300, 320), fill="#8a431f")
    draw.ellipse((330, 40, 530, 220), fill="#68717d")
    buffer = io.BytesIO()
    original.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    model_server = ThreadingHTTPServer(("127.0.0.1", 0), MockModels)
    thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        api_port = probe.getsockname()[1]
    model_port = model_server.server_port
    env = {key: value for key, value in os.environ.items() if key not in {
        "PIPELINE_HOST", "PIPELINE_PORT", "PIPELINE_DATA_DIR", "PIPELINE_API_KEY",
        "SAM3_URL", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "CORS_ORIGINS",
        "SAM3_CLASS_NAMES", "LLM_SYSTEM_PROMPT", "LLM_USER_PROMPT",
        "PIPELINE_LOG_DIR", "LOG_LEVEL", "LOG_RETENTION_DAYS",
    }}
    env.update({
        "PIPELINE_HOST": "127.0.0.1", "PIPELINE_PORT": str(api_port),
        "PIPELINE_DATA_DIR": str(output / "events"), "PIPELINE_API_KEY": "local-smoke-key",
        "PIPELINE_LOG_DIR": str(output / "logs"), "LOG_LEVEL": "INFO", "LOG_RETENTION_DAYS": "30",
        "SAM3_URL": f"http://127.0.0.1:{model_port}/predict/file",
        "LLM_BASE_URL": f"http://127.0.0.1:{model_port}/v1",
        "LLM_MODEL": "", "LLM_API_KEY": "", "CORS_ORIGINS": "",
        "SAM3_CONCURRENCY": "4", "LLM_CONCURRENCY": "2",
        "SAM3_QUEUE_SIZE": "15", "LLM_QUEUE_SIZE": "15",
        "SHUTDOWN_TIMEOUT_SECONDS": "2", "PYTHONUNBUFFERED": "1",
    })
    process = None
    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            options = {}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen([sys.executable, "-m", "app.main"], cwd=ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT, **options)
            with httpx.Client(base_url=f"http://127.0.0.1:{api_port}", trust_env=False, timeout=5) as client:
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("API process exited; inspect " + str(output / "server.log"))
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("API did not start")
                    time.sleep(0.05)

                def upload(index):
                    response = client.post("/v1/frames", headers={"X-API-Key": "local-smoke-key"},
                                           data={"machine_id": f"frontend-{index // 5 + 1}",
                                                 "stream_id": f"camera-{index % 5 + 1}",
                                                 "captured_at": datetime.now(timezone.utc).isoformat()},
                                           files={"image": ("test.png", image_bytes, "image/png")})
                    assert response.status_code == 202, response.text
                    assert response.json() == {"accepted": True}

                # Fifteen distinct streams upload together; no staggered scheduling.
                with ThreadPoolExecutor(max_workers=15) as pool:
                    list(pool.map(upload, range(15)))
                deadline = time.monotonic() + 15
                while True:
                    health = client.get("/health").json()
                    if health["counts"].get("saved") == 15:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Did not save all frames: {health}")
                    time.sleep(0.05)
                metadata_files = list((output / "events").rglob("metadata.json"))
                assert len(metadata_files) == 15
                for metadata_file in metadata_files:
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    assert metadata["llm"]["result"] == "fire_smoke"
                    assert metadata["llm"]["model"] == "http-smoke-model"
                    assert (metadata_file.parent / "original.png").read_bytes() == image_bytes
                    assert (metadata_file.parent / "annotated.jpg").is_file()
                log_files = list((output / "logs").glob("pipeline-*.log"))
                assert len(log_files) == 1
                log_text = log_files[0].read_text(encoding="utf-8")
                assert "Uvicorn running" in log_text and "pipeline_started" in log_text
                assert log_text.count(" INFO app.pipeline confirmed ") == 15
                assert "GET /health" not in log_text
                assert "local-smoke-key" not in log_text and "data:image" not in log_text
                (output / "result.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
                print(json.dumps({"status": "passed", "accepted": 15, "saved": 15,
                                  "output": str(output),
                                  "annotated_example": str(metadata_files[0].parent / "annotated.jpg")},
                                 ensure_ascii=False, indent=2))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        model_server.shutdown()
        model_server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
