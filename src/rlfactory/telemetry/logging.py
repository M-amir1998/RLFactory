"""Structured logging with run context and secret redaction.

Every record carries the active context (``run_id``, ``stage``, ``task_id``, ...) set via
:func:`log_context`. Messages and fields are passed through :func:`redact` so accidentally logged
tokens (Hugging Face, OpenAI-style, AWS, GitHub, bearer headers, private keys) never reach disk.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Any, Literal

_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar("rlfactory_log_context", default=None)
_FIELDS_ATTR = "rlf_fields"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"),
)
REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def current_context() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach ``fields`` to every log record emitted inside the ``with`` block."""
    token = _CONTEXT.set({**current_context(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def log_event(
    logger: logging.Logger, message: str, level: int = logging.INFO, **fields: Any
) -> None:
    """Log ``message`` with structured ``fields`` (merged with the active context)."""
    logger.log(level, message, extra={_FIELDS_ATTR: fields})


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, int | float | bool) or value is None:
        return value
    return redact(str(value))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        payload.update(_sanitize(current_context()))
        payload.update(_sanitize(getattr(record, _FIELDS_ATTR, {})))
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = redact(super().format(record))
        fields = {**current_context(), **getattr(record, _FIELDS_ATTR, {})}
        if not fields:
            return base
        rendered = " ".join(f"{key}={_sanitize(value)}" for key, value in sorted(fields.items()))
        return f"{base} | {rendered}"


def configure_logging(
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
    stream: IO[str] | None = None,
) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger("rlfactory")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("rlfactory") else f"rlfactory.{name}")
