"""Conversation history, preferences and personalization routes (Phase 20).

Every route here is owner-scoped: the session id alone is never enough -
the caller's token must resolve to the user who owns the conversation,
otherwise the answer is 404 (the same shape as an unknown id, so ids
cannot be probed for existence).

* ``GET    /api/conversations``            list the caller's conversations
* ``DELETE /api/conversations``            forget ALL the caller's conversations
* ``GET    /api/conversations/{id}``       one conversation's metadata
* ``GET    /api/conversations/{id}/messages`` the stored transcript
* ``PATCH  /api/conversations/{id}``       rename
* ``DELETE /api/conversations/{id}``       forget it everywhere
* ``GET/PUT /api/preferences``             explicit conversation settings
* ``GET/PUT /api/settings``                per-user app settings (Phase 21)
* ``GET    /api/personalization``          what the system learned + flag
* ``DELETE /api/personalization``          clear learned memory
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import user_or_401
from app.auth.personalization import profile_from_preferences
from app.auth.schemas import (
    ConversationBulkDeleteResponse,
    ConversationListResponse,
    ConversationPublic,
    MessageListResponse,
    PersonalizationClearResponse,
    PersonalizationView,
    LearnedItem,
    MessageResponse,
    RenameConversationRequest,
    StoredMessage,
    UserPreferences,
    UserSettings,
)
from app.auth.store import get_auth_store
from app.config.settings import get_settings
from app.memory.snapshot import SessionSnapshot
from app.models.schemas import ErrorResponse
from app.pipelines.turn_log import purge_session_turns
from app.state_engine import get_state_engine

router = APIRouter(prefix="/api", tags=["conversations"])


def _not_found(error: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=ErrorResponse(error=error, detail=detail).model_dump(),
    )


def _owned_conversation(session_id: str, user_id: str) -> str:
    """Return the session id when the caller owns it; 404 otherwise."""
    owner = get_auth_store().conversation_owner(session_id)
    if owner is None or owner != user_id:
        # Unknown and not-yours are indistinguishable on purpose.
        raise _not_found("conversation_not_found", session_id)
    return session_id


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(user=Depends(user_or_401)) -> ConversationListResponse:  # noqa: ANN001
    rows = get_auth_store().list_conversations(user["id"])
    return ConversationListResponse(
        conversations=[ConversationPublic(**row) for row in rows]
    )


@router.delete("/conversations", response_model=ConversationBulkDeleteResponse)
def delete_all_conversations(
    user=Depends(user_or_401),  # noqa: ANN001
) -> ConversationBulkDeleteResponse:
    """Forget **every** conversation the caller owns - engine RAM, memory
    snapshots, audit rows, stored transcripts and ownership. Other users'
    conversations are untouched (ids are collected caller-scoped first)."""
    store = get_auth_store()
    session_ids = store.conversation_ids_for_user(user["id"])
    engine = get_state_engine()
    for session_id in session_ids:
        engine.reset(session_id)
        purge_session_turns(session_id)
        store.delete_conversation_data(session_id)
    return ConversationBulkDeleteResponse(deleted=len(session_ids))


@router.get("/conversations/{session_id}", response_model=ConversationPublic)
def get_conversation(
    session_id: str, user=Depends(user_or_401)  # noqa: ANN001
) -> ConversationPublic:
    _owned_conversation(session_id, user["id"])
    for row in get_auth_store().list_conversations(user["id"]):
        if row["session_id"] == session_id:
            return ConversationPublic(**row)
    raise _not_found("conversation_not_found", session_id)


@router.get(
    "/conversations/{session_id}/messages", response_model=MessageListResponse
)
def get_conversation_messages(
    session_id: str, user=Depends(user_or_401)  # noqa: ANN001
) -> MessageListResponse:
    """The stored transcript, oldest first (owner only)."""
    store = get_auth_store()
    _owned_conversation(session_id, user["id"])
    messages = store.messages_for_conversation(session_id, user["id"])
    title = store.conversation_title(session_id) or "Conversation"
    return MessageListResponse(
        session_id=session_id,
        title=title,
        messages=[StoredMessage(**m) for m in messages],
    )


@router.patch("/conversations/{session_id}", response_model=ConversationPublic)
def rename_conversation(
    session_id: str,
    body: RenameConversationRequest,
    user=Depends(user_or_401),  # noqa: ANN001
) -> ConversationPublic:
    store = get_auth_store()
    _owned_conversation(session_id, user["id"])
    store.rename_conversation(session_id, user["id"], body.title)
    for row in store.list_conversations(user["id"]):
        if row["session_id"] == session_id:
            return ConversationPublic(**row)
    raise _not_found("conversation_not_found", session_id)


@router.delete("/conversations/{session_id}", response_model=MessageResponse)
def delete_conversation(
    session_id: str, user=Depends(user_or_401)  # noqa: ANN001
) -> MessageResponse:
    """Forget a conversation everywhere: engine RAM, memory snapshots,
    the metadata-only turn audit, the stored transcript and ownership."""
    _owned_conversation(session_id, user["id"])
    engine = get_state_engine()
    engine.reset(session_id)          # RAM + persisted snapshot
    purge_session_turns(session_id)   # audit rows (best effort)
    get_auth_store().delete_conversation_data(session_id)
    return MessageResponse(detail="conversation deleted")


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


@router.get("/preferences", response_model=UserPreferences)
def get_preferences(user=Depends(user_or_401)) -> UserPreferences:  # noqa: ANN001
    return UserPreferences(**get_auth_store().get_preferences(user["id"]))


@router.put("/preferences", response_model=UserPreferences)
def put_preferences(
    body: UserPreferences, user=Depends(user_or_401)  # noqa: ANN001
) -> UserPreferences:
    """Persist explicit conversation settings.

    They are seeded into *new* conversations; existing conversations keep
    the profile they already have (their in-chat learnings are not
    rewritten behind the user's back - the Personalization screen offers
    an explicit "clear" for that).
    """
    stored = get_auth_store().set_preferences(
        user["id"],
        {
            "response_length": body.response_length,
            "tone": body.tone,
            "asks_questions": body.asks_questions,
            "preferred_language": body.preferred_language,
            "personalization_enabled": body.personalization_enabled,
        },
    )
    return UserPreferences(**stored)


# ---------------------------------------------------------------------------
# App settings (Phase 21): notifications and similar shell preferences
# ---------------------------------------------------------------------------


@router.get("/settings", response_model=UserSettings)
def get_user_settings(user=Depends(user_or_401)) -> UserSettings:  # noqa: ANN001
    return UserSettings(**get_auth_store().get_settings(user["id"]))


@router.put("/settings", response_model=UserSettings)
def put_user_settings(
    body: UserSettings, user=Depends(user_or_401)  # noqa: ANN001
) -> UserSettings:
    """Persist per-user app settings (in-app reminder preferences, ...)."""
    stored = get_auth_store().set_settings(
        user["id"], {"daily_reminders": body.daily_reminders}
    )
    return UserSettings(**stored)


# ---------------------------------------------------------------------------
# Personalization transparency + control
# ---------------------------------------------------------------------------


def _session_profile(session_id: str):  # noqa: ANN202
    """The stored profile of one session: RAM first, then the snapshot."""
    engine = get_state_engine()
    record = engine.record(session_id)
    if record is not None:
        return record.state.profile
    snapshot = engine.memory.load(session_id)
    if snapshot is not None:
        return snapshot.state.profile
    return None


@router.get("/personalization", response_model=PersonalizationView)
def personalization_view(user=Depends(user_or_401)) -> PersonalizationView:  # noqa: ANN001
    """What the system actually learned, aggregated across conversations.

    Only explicit in-chat statements are ever learned (Phase 8 extractor).
    The default language is not reported as a learning, mirroring
    ``has_learnings`` on the backend.
    """
    store = get_auth_store()
    prefs = store.get_preferences(user["id"])
    counts: dict[tuple[str, str], int] = {}
    sessions_with_learnings = 0
    for session_id in store.conversation_ids_for_user(user["id"]):
        profile = _session_profile(session_id)
        if profile is None:
            continue
        learned = _learned_entries(profile, prefs)
        if learned:
            sessions_with_learnings += 1
        for kind, value in learned:
            counts[(kind, value)] = counts.get((kind, value), 0) + 1
    items = [
        LearnedItem(kind=kind, value=value, conversations=n)
        for (kind, value), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return PersonalizationView(
        enabled=bool(prefs["personalization_enabled"]),
        items=items,
        conversations_with_learnings=sessions_with_learnings,
    )


@router.delete("/personalization", response_model=PersonalizationClearResponse)
def clear_personalization(user=Depends(user_or_401)) -> PersonalizationClearResponse:  # noqa: ANN001
    """Remove everything learned in chat; keep the explicit settings.

    Each conversation's profile is rewritten to exactly the seed derived
    from the user's Preferences screen - learned names, styles,
    preferences and topics to avoid are dropped, then the result is
    persisted, so the memory is genuinely gone, not merely hidden.
    """
    store = get_auth_store()
    prefs = store.get_preferences(user["id"])
    seed = profile_from_preferences(prefs)
    engine = get_state_engine()
    cleared = 0
    for session_id in store.conversation_ids_for_user(user["id"]):
        record = engine.record(session_id)
        if record is not None:
            engine.set_profile(session_id, seed.model_copy(deep=True))
            engine.memory.save(SessionSnapshot.from_record(record))
            cleared += 1
            continue
        snapshot = engine.memory.load(session_id)
        if snapshot is None:
            continue
        snapshot.state.profile = seed.model_copy(deep=True)
        engine.memory.save(snapshot)
        cleared += 1
    return PersonalizationClearResponse(cleared_conversations=cleared)


def _learned_entries(profile, prefs: dict) -> list[tuple[str, str]]:  # noqa: ANN001
    """(kind, value) pairs the *extractor* learned in chat - seeds excluded.

    Anything that matches the seed derived from the user's Preferences
    screen is a setting, not a learning, and is not reported here; the
    extractor's style vocabulary (brief/detailed/plain) never overlaps the
    seeded tones (warm/gentle/direct), so style entries are unambiguous.
    """
    from app.auth.personalization import (
        SEED_PREFERENCE_STRINGS,
        profile_from_preferences,
    )
    from app.personalization.profile import has_learnings

    if not has_learnings(profile):
        return []
    seed = profile_from_preferences(prefs)
    entries: list[tuple[str, str]] = []
    if profile.name:
        entries.append(("name", profile.name))
    if (
        profile.communication_style
        and profile.communication_style != seed.communication_style
    ):
        entries.append(("communication_style", profile.communication_style))
    if (
        profile.preferred_language
        and profile.preferred_language != seed.preferred_language
    ):
        entries.append(("preferred_language", profile.preferred_language))
    for item in profile.preferences:
        if item not in SEED_PREFERENCE_STRINGS:
            entries.append(("preference", item))
    for item in profile.topics_to_avoid:
        entries.append(("topic_to_avoid", item))
    return entries
