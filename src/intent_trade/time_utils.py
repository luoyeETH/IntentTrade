"""UTC storage to configured display-timezone helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def to_display_timezone(value: datetime, timezone_name: str = "Asia/Shanghai") -> datetime:
    """Treat naive persisted timestamps as UTC and return a localized value."""

    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return aware.astimezone(ZoneInfo(timezone_name))


def format_display_time(
    value: datetime,
    timezone_name: str = "Asia/Shanghai",
    *,
    include_seconds: bool = False,
) -> str:
    localized = to_display_timezone(value, timezone_name)
    pattern = "%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M"
    return localized.strftime(pattern)
