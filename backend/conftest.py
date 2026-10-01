"""Pytest bootstrap: make `backend/` importable regardless of cwd."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Tests must never write conversation snapshots into the project's data/
# directory (Phase 7). An explicit environment still wins: a test that wants
# persistence sets MEMORY_BACKEND/... itself.
os.environ.setdefault("MEMORY_BACKEND", "none")
# Phase 15: no turn-audit rows in project data/ either; tests that need the
# audit either inject a TurnLog(tmp) or set TURN_LOG_ENABLED themselves.
os.environ.setdefault("TURN_LOG_ENABLED", "false")


def pytest_configure() -> None:
    """Keep third-party HTTP/cache chatter out of test output.

    The application loggers are untouched: only these libraries, which emit
    one DEBUG line per socket frame, are quieted.
    """
    for name in (
        "httpx", "httpcore", "huggingface_hub", "fsspec", "filelock",
        "urllib3", "datasets", "transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
