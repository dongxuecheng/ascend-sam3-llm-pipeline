"""Integration point for the future external alarm API."""

import logging

from app.storage import SavedEvent


logger = logging.getLogger(__name__)


async def send_alarm(event: SavedEvent) -> bool:
    """Return True only after the external alarm service confirms delivery."""
    # TODO: Implement the agreed alarm protocol here. No network request is made.
    # event contains paths to the original, annotated image and JSON metadata.
    logger.info("alarm_not_configured evidence=%s", event.directory)
    return False
