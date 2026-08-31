"""Integration point for the future external alarm API."""

import logging
from dataclasses import dataclass

from app.storage import SavedEvent


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UnsavedAlarmEvent:
    """Alarm payload retained in memory when local evidence persistence fails."""

    machine_id: str
    stream_id: str
    result: str
    original: bytes
    original_extension: str
    annotated: bytes | None
    metadata: dict

    @property
    def reference(self) -> str:
        return f"unsaved:{self.machine_id}/{self.stream_id}"


AlarmEvent = SavedEvent | UnsavedAlarmEvent


def is_configured() -> bool:
    """Change to return True together with the real alarm implementation."""
    return False


def reference(event: AlarmEvent) -> str:
    return str(event.directory) if isinstance(event, SavedEvent) else event.reference


async def send_alarm(event: AlarmEvent) -> bool:
    """Return True only after the external alarm service confirms delivery."""
    logger.info("alarm_not_configured evidence=%s", reference(event))
    return False
