import io
import logging
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.logging_setup import BeijingFormatter, DailyLogHandler, configured_logging


class LoggingTests(unittest.TestCase):
    def test_retention_keeps_at_most_30_calendar_days_and_only_owns_log_files(self):
        today = date(2026, 8, 28)
        for days in (1, 30):
            with self.subTest(days=days), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for age in range(40):
                    (root / f"pipeline-{today - timedelta(days=age)}.log").write_text("old\n")
                for name in ("original.png", "metadata.json", "other.log", "pipeline-2026-02-30.log"):
                    (root / name).write_text("keep")
                nested = root / "pipeline-2000-01-01.log"
                nested.mkdir()
                (nested / "metadata.json").write_text("keep nested evidence")
                with patch("app.logging_setup.beijing_today", return_value=today):
                    handler = DailyLogHandler(root, days)
                handler.close()
                retained = {path.name for path in root.glob("pipeline-*.log")
                            if path.is_file() and path.name != "pipeline-2026-02-30.log"}
                self.assertEqual(retained, {
                    f"pipeline-{today - timedelta(days=age)}.log" for age in range(days)
                })
                self.assertEqual((nested / "metadata.json").read_text(), "keep nested evidence")
                for name in ("original.png", "metadata.json", "other.log", "pipeline-2026-02-30.log"):
                    self.assertEqual((root / name).read_text(), "keep")

    def test_beijing_midnight_rollover_closes_old_file_before_pruning_and_preserves_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instant = datetime(2026, 8, 28, 15, 59, 59, 123000, tzinfo=timezone.utc)
            with patch("app.logging_setup.datetime", wraps=datetime) as clock:
                clock.now.side_effect = lambda tz: instant.astimezone(tz)
                handler = DailyLogHandler(root, 1)
                handler.setFormatter(BeijingFormatter("%(asctime)s %(message)s"))
                try:
                    handler.handle(logging.makeLogRecord({
                        "created": instant.timestamp(), "msg": "第一天：烟雾",
                    }))
                    old_file = root / "pipeline-2026-08-28.log"
                    self.assertEqual(old_file.read_text(encoding="utf-8"),
                                     "2026-08-28T23:59:59.123+08:00 第一天：烟雾\n")
                    # UTC is still August 28; Beijing has entered the next calendar day.
                    instant = datetime(2026, 8, 28, 16, 0, 0, 456000, tzinfo=timezone.utc)
                    handler.handle(logging.makeLogRecord({
                        "created": instant.timestamp(), "msg": "第二天：火焰",
                    }))
                    self.assertFalse(old_file.exists())
                    self.assertEqual((root / "pipeline-2026-08-29.log").read_text(encoding="utf-8"),
                                     "2026-08-29T00:00:00.456+08:00 第二天：火焰\n")
                finally:
                    handler.close()

    def test_background_cleanup_works_without_requests_and_stops_on_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            today = date(2026, 8, 28)
            settings = Settings(pipeline_log_dir=root)
            old_handlers = logging.getLogger().handlers
            with redirect_stderr(io.StringIO()), \
                    patch("app.logging_setup.beijing_today", return_value=today) as clock, \
                    patch("app.logging_setup.CLEANUP_INTERVAL_SECONDS", 0.01):
                with configured_logging(settings):
                    old_file = root / f"pipeline-{today}.log"
                    self.assertTrue(old_file.exists())
                    clock.return_value = today + timedelta(days=31)
                    deadline = time.monotonic() + 2
                    while old_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertFalse(old_file.exists())
                self.assertIs(logging.getLogger().handlers, old_handlers)
            self.assertFalse(any(t.name == "log-retention" for t in threading.enumerate()))

    def test_application_and_uvicorn_share_logs_without_health_or_http_client_noise(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()) as console:
            root = Path(directory)
            with configured_logging(Settings(pipeline_log_dir=root, log_level="debug")):
                logging.getLogger("app.pipeline").info("火焰确认 machine=test stream=camera")
                logging.getLogger("uvicorn.error").error("test server failure")
                access = logging.getLogger("uvicorn.access")
                access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/health", "1.1", 200)
                access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", "/health", "1.1", 503)
                access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1", "POST", "/v1/frames?key=secret-query", "1.1", 202)
                logging.getLogger("httpx").info("secret-url-should-not-be-logged")
                logging.getLogger("httpcore").debug("secret-headers-should-not-be-logged")
                try:
                    raise OSError("test storage failure")
                except OSError:
                    logging.getLogger("app.pipeline").exception("save_failed")
            content = next(root.glob("pipeline-*.log")).read_text(encoding="utf-8")
            self.assertEqual(console.getvalue(), content)
            self.assertIn("火焰确认", content)
            self.assertIn("test server failure", content)
            self.assertNotIn('GET /health HTTP/1.1" 200', content)
            self.assertIn('GET /health HTTP/1.1" 503', content)
            self.assertIn('POST /v1/frames HTTP/1.1" 202', content)
            self.assertNotIn("secret-", content)
            self.assertIn("Traceback", content)
            self.assertIn("OSError: test storage failure", content)
            self.assertRegex(content, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+08:00 INFO app.pipeline")

    def test_file_rotation_failure_is_reported_without_raising_to_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.logging_setup.beijing_today", return_value=date(2026, 8, 28)) as clock:
                handler = DailyLogHandler(Path(directory), 30)
                try:
                    clock.return_value = date(2026, 8, 29)
                    record = logging.makeLogRecord({"msg": "inference result"})
                    with patch.object(handler, "maintain", side_effect=OSError("disk unavailable")), \
                            patch.object(handler, "handleError") as report:
                        handler.handle(record)
                    report.assert_called_once_with(record)
                finally:
                    handler.close()

    def test_cleanup_never_follows_symbolic_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "evidence.txt"
            target.write_text("keep")
            link = root / "pipeline-2000-01-01.log"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("This Windows account cannot create symbolic links")
            handler = DailyLogHandler(root, 30)
            handler.close()
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
