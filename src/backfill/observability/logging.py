from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

# Keys that could carry personal data. They are removed from every log event, whatever logged them.
PII_KEYS = frozenset(
    {
        "lat",
        "lon",
        "latitude",
        "longitude",
        "location",
        "home",
        "profile_home",
        "planned_location",
        "actual_location",
        "phone",
        "email",
        "name",
        "password",
        "token",
        "authorization",
        "pings",
        "riders",
        "evidence",
    }
)


def scrub_pii(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in list(event):
        if key.lower() in PII_KEYS:
            event[key] = "[redacted]"
    return event


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    renderer: Any = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            scrub_pii,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
