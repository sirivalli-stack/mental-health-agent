"""Shared test infrastructure (Phase 16).

Test doubles plus the two environment helpers the integration suites share:

* ``FakeLLM`` / ``FakeGuardrail`` / ``fast_post_gate`` - the default suite
  never touches the network, Ollama or the Mila BERT weights;
* ``audit_env`` - scopes ``TURN_LOG_ENABLED`` + ``DATABASE_PATH`` (and resets
  the settings/turn-log/engine singletons) for one test;
* ``api_client`` - a ``TestClient`` with the LLM and guardrail faked.

Used by the pipeline/API modules (Phases 12, 15, 16); no test writes into
the project's ``data/`` directory.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config.settings import reset_settings_cache
from app.pipelines.turn_log import reset_turn_log
from app.safety.post_generation import PostGenerationGate
from app.state_engine import reset_state_engine

DIGIT = re.compile(r"\d")

BENIGN = "I have my end-sem exams next week and I cannot focus at all."
BLOCKER = "what is the suicide hotline number?"


class FakeLLM:
    """Stands in for ``LLMService``; records every call."""

    def __init__(self, text: str = "That sounds rough - what part is hardest?",
                 error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def complete(self, profile, state, message, *, safety_notes=None, **kwargs):  # noqa: ANN001
        self.calls.append(
            {"profile": profile, "message": message, "safety_notes": safety_notes}
        )
        if self.error is not None:
            raise self.error
        from app.services.llm_service import LLMReply

        return LLMReply(text=self.text, model="fake", latency_ms=1.0)


class FakeGuardrail:
    """Stands in for the Mila output-guardrail classifier."""

    def __init__(self, label: str = "safe", score: float = 0.02) -> None:
        self.label = label
        self.score = score

    def predict(self, text: str) -> tuple[str, float]:  # noqa: ARG002
        return self.label, self.score


def fast_post_gate() -> PostGenerationGate:
    """Output gate with the guardrail faked (deterministic layer intact)."""
    return PostGenerationGate(guardrail=FakeGuardrail())


def sid(prefix: str = "t") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


@contextmanager
def audit_env(db_path: Path):
    """TURN_LOG_ENABLED=true + DATABASE_PATH=db for one test."""
    keys = ("TURN_LOG_ENABLED", "DATABASE_PATH")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["TURN_LOG_ENABLED"] = "true"
    os.environ["DATABASE_PATH"] = str(db_path)
    reset_settings_cache()
    reset_turn_log()
    reset_state_engine()
    try:
        yield db_path
    finally:
        reset_turn_log()
        reset_state_engine()
        reset_settings_cache()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def api_client(fake_llm):  # noqa: ANN001
    """TestClient with LLM + guardrail faked (same swap as the API fixture)."""
    from app.pipelines import chat as chat_mod
    from app.safety.post_generation import build_post_gate as _real_gate
    from app.services.llm_service import get_llm_service as _real_llm
    from main import app

    chat_mod.get_llm_service = lambda: fake_llm  # type: ignore[assignment]
    chat_mod.build_post_gate = fast_post_gate
    try:
        with TestClient(app) as client:
            yield client
    finally:
        chat_mod.get_llm_service = _real_llm  # type: ignore[assignment]
        chat_mod.build_post_gate = _real_gate


def all_text(row: dict) -> str:
    """Every stored value of a row, as one searchable string."""
    return json.dumps({k: str(v) for k, v in row.items()}, ensure_ascii=False)
