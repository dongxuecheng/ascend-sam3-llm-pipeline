"""Timezone helpers shared by logs and evidence metadata."""

from datetime import datetime, timedelta, timezone


BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def as_beijing(value: datetime) -> datetime:
    return value.astimezone(BEIJING_TIMEZONE)
