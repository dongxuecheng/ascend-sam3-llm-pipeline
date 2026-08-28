"""Beijing-time daily logs with age-based cleanup, even without requests."""

import logging
import re
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings


CLEANUP_INTERVAL_SECONDS = 60
# Keep timestamps and calendar dates independent of the host timezone and tzdata.
BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def beijing_today() -> date:
    return datetime.now(BEIJING_TIMEZONE).date()


class BeijingFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        timestamp = datetime.fromtimestamp(record.created, BEIJING_TIMEZONE)
        return timestamp.strftime(datefmt) if datefmt else timestamp.isoformat(timespec="milliseconds")


class DailyLogHandler(logging.FileHandler):
    def __init__(self, directory: Path, retention_days: int):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.day = beijing_today()
        self._prune(self.day)
        super().__init__(self.directory / f"pipeline-{self.day}.log", encoding="utf-8")

    def _prune(self, today: date) -> int:
        cutoff = today - timedelta(days=self.retention_days - 1)
        removed = 0
        for path in self.directory.iterdir():
            match = re.fullmatch(r"pipeline-(\d{4}-\d{2}-\d{2})\.log", path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            try:
                day = date.fromisoformat(match[1])
            except ValueError:
                continue
            if day < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def maintain(self) -> int:
        self.acquire()
        try:
            if self._closed:
                return 0
            today = beijing_today()
            if today != self.day:
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.baseFilename = str(self.directory / f"pipeline-{today}.log")
                self.day = today
            return self._prune(today)
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            if beijing_today() != self.day:
                self.maintain()
            super().emit(record)
        except Exception:
            # A logging failure must not turn a successful inference into a failure.
            self.handleError(record)


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if record.name != "uvicorn.access" or not isinstance(args, tuple) or len(args) != 5:
            return True
        client, method, path, version, status = args
        path = str(path).split("?", 1)[0]  # Query strings may contain credentials.
        record.args = client, method, path, version, status
        return not (method == "GET" and path == "/health" and status == 200)


@contextmanager
def configured_logging(settings: Settings):
    file_handler = DailyLogHandler(settings.pipeline_log_dir, settings.log_retention_days)
    console_handler = logging.StreamHandler()
    formatter = BeijingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler.addFilter(AccessLogFilter())

    root = logging.getLogger()
    old_root = root.handlers, root.level
    root.handlers = [file_handler, console_handler]
    root.setLevel(settings.log_level)
    # Uvicorn must share our handlers; HTTP clients must not print URLs/headers.
    saved_loggers = {}
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        saved_loggers[logger] = logger.handlers, logger.level, logger.propagate
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.WARNING if name in {"httpx", "httpcore"} else logging.NOTSET)

    stopped = threading.Event()

    def cleanup():
        while not stopped.wait(CLEANUP_INTERVAL_SECONDS):
            try:
                removed = file_handler.maintain()
                if removed:
                    logging.getLogger(__name__).info("logs_pruned files=%s", removed)
            except Exception as exc:
                logging.getLogger(__name__).error("log_cleanup_failed error_type=%s", type(exc).__name__)

    worker = threading.Thread(target=cleanup, name="log-retention", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
        root.handlers, root.level = old_root
        for logger, (handlers, level, propagate) in saved_loggers.items():
            logger.handlers, logger.level, logger.propagate = handlers, level, propagate
        file_handler.close()
        console_handler.close()
