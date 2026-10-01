"""Scripted multi-turn runs without HTTP (Phase 15). [OURS]

``run_conversation`` is the programmatic integration surface: it drives the
same ``run_chat_turn`` pipeline the API uses, over a list of messages, and
returns every :class:`~app.pipelines.chat.ChatOutcome` plus the shared
session id. Tests, demos and the Phase 17+ evaluation harness use it to run
identical scripts through profiles A-D without a web server in the loop.

Nothing here is a second pipeline: same state update, same gates, same
fakes/injection points (``engine`` / ``llm`` / ``pre_gate`` / ``post_gate`` /
``turn_log``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from app.models.schemas import SystemProfile, UserState
from app.pipelines.chat import ChatOutcome, ChatSource, run_chat_turn


@dataclass(frozen=True)
class ConversationOutcome:
    """Everything one scripted conversation produced."""

    session_id: str
    profile: SystemProfile
    turns: tuple[ChatOutcome, ...]

    @property
    def final_state(self) -> UserState | None:
        return self.turns[-1].state if self.turns else None

    @property
    def sources(self) -> tuple[ChatSource, ...]:
        return tuple(turn.source for turn in self.turns)

    @property
    def replies(self) -> tuple[str, ...]:
        return tuple(turn.reply for turn in self.turns)


def run_conversation(
    messages: Iterable[str],
    *,
    session_id: str | None = None,
    profile: SystemProfile | str | None = None,
    engine=None,  # noqa: ANN001 - same injection surface as run_chat_turn
    llm=None,  # noqa: ANN001
    pre_gate=None,  # noqa: ANN001
    post_gate=None,  # noqa: ANN001
    turn_log=None,  # noqa: ANN001
) -> ConversationOutcome:
    """Run ``messages`` sequentially through the chat pipeline.

    One shared session (generated when not supplied), one outcome per
    message, turn indices counting up - exactly what the API would produce
    for the same script. Empty input raises ``ValueError``; a message the
    pipeline rejects (empty/too long) still appears as a ``pre_blocked``
    turn, mirroring the API.
    """
    texts = list(messages)
    if not texts:
        raise ValueError("messages must contain at least one message")
    sid = session_id or uuid4().hex

    outcomes = tuple(
        run_chat_turn(
            text,
            session_id=sid,
            profile=profile,
            engine=engine,
            llm=llm,
            pre_gate=pre_gate,
            post_gate=post_gate,
            turn_log=turn_log,
        )
        for text in texts
    )
    return ConversationOutcome(
        session_id=sid,
        profile=outcomes[0].profile,
        turns=outcomes,
    )
