"""Centralised logging configuration.

Initialised once at application start (see backend/main.py). Writes to the
console and to a rotating file under `<project-root>/logs/app.log`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config.settings import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class ClosedFileSafeStreamHandler(logging.StreamHandler):
    """Console handler that stays silent once the stream is closed.

    pytest and some hosting environments close captured stdout while worker
    threads are still logging; without this guard the stdlib handler prints
    'ValueError: I/O operation on closed file' during teardown.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        stream = self.stream
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            super().emit(record)
        except ValueError:
            pass


def configure_logging(force: bool = False) -> Path:
    """Configure the root logger. Safe to call more than once."""
    global _configured
    settings = get_settings()

    if _configured and not force:
        return settings.log_path

    settings.log_path.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_path / "app.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    root.handlers.clear()

    console = ClosedFileSafeStreamHandler()
    console.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _configured = True

    logging.getLogger(__name__).info(
        "Logging configured | profile=%s | file=%s",
        settings.system_profile,
        log_file,
    )
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
