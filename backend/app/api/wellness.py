"""Wellness routes: daily mood check-ins + private journal (Phase 23).

Owner-scoped like every other personal route: every query carries the
caller's ``user_id``, so ids cannot be probed across accounts (missing
and not-yours are both 404, matching the conversations routes).

* ``GET    /api/moods``        check-in history (newest first)
* ``PUT    /api/moods``        today's (or a date's) check-in - upsert
* ``DELETE /api/moods/{id}``   remove one check-in
* ``GET    /api/journal``      the caller's entries (newest first)
* ``POST   /api/journal``      create an entry
* ``GET    /api/journal/{id}`` one entry
* ``PUT    /api/journal/{id}`` partial update
* ``DELETE /api/journal/{id}`` delete an entry

Honesty notes, enforced by the schemas and the frontend copy:
check-ins and journal entries are *self-reported records the user owns*.
Nothing here is analysed, scored, diagnosed or fed to the classifiers,
and no row is ever visible to another account.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import user_or_401
from app.auth.schemas import (
    JournalCreateRequest,
    JournalEntry,
    JournalListResponse,
    JournalUpdateRequest,
    MoodEntry,
    MoodListResponse,
    MoodUpsertRequest,
)
from app.auth.store import get_auth_store
from app.models.schemas import ErrorResponse

router = APIRouter(prefix="/api", tags=["wellness"])


def _not_found(error: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=ErrorResponse(error=error, detail=detail).model_dump(),
    )


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Mood check-ins
# ---------------------------------------------------------------------------


@router.get("/moods", response_model=MoodListResponse)
def list_moods(
    limit: int = 90, user=Depends(user_or_401)  # noqa: ANN001
) -> MoodListResponse:
    """The caller's own check-in history (never anyone else's)."""
    bounded = max(1, min(limit, 366))
    rows = get_auth_store().list_mood_entries(user["id"], limit=bounded)
    return MoodListResponse(entries=[MoodEntry(**row) for row in rows])


@router.put("/moods", response_model=MoodEntry)
def put_mood(
    body: MoodUpsertRequest, user=Depends(user_or_401)  # noqa: ANN001
) -> MoodEntry:
    """Record the check-in for one day (defaults to today, UTC).

    Re-checking the same day replaces that day's entry instead of
    stacking duplicates - one honest record per date.
    """
    stored = get_auth_store().upsert_mood_entry(
        user["id"], body.date or _today(), body.mood, body.note
    )
    return MoodEntry(**stored)


@router.delete("/moods/{entry_id}", response_model=MoodEntry)
def delete_mood(
    entry_id: int, user=Depends(user_or_401)  # noqa: ANN001
) -> MoodEntry:
    store = get_auth_store()
    entries = store.list_mood_entries(user["id"], limit=366)
    target = next((e for e in entries if e["id"] == entry_id), None)
    if target is None or not store.delete_mood_entry(user["id"], entry_id):
        # unknown and not-yours are indistinguishable on purpose
        raise _not_found("mood_entry_not_found", str(entry_id))
    return MoodEntry(**target)


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


@router.get("/journal", response_model=JournalListResponse)
def list_journal(
    user=Depends(user_or_401)  # noqa: ANN001
) -> JournalListResponse:
    rows = get_auth_store().list_journal_entries(user["id"])
    return JournalListResponse(entries=[JournalEntry(**row) for row in rows])


@router.post("/journal", response_model=JournalEntry, status_code=201)
def create_journal(
    body: JournalCreateRequest, user=Depends(user_or_401)  # noqa: ANN001
) -> JournalEntry:
    stored = get_auth_store().create_journal_entry(
        user["id"], body.title, body.body
    )
    return JournalEntry(**stored)


@router.get("/journal/{entry_id}", response_model=JournalEntry)
def get_journal(
    entry_id: int, user=Depends(user_or_401)  # noqa: ANN001
) -> JournalEntry:
    row = get_auth_store().get_journal_entry(user["id"], entry_id)
    if row is None:
        raise _not_found("journal_entry_not_found", str(entry_id))
    return JournalEntry(**row)


@router.put("/journal/{entry_id}", response_model=JournalEntry)
def update_journal(
    entry_id: int,
    body: JournalUpdateRequest,
    user=Depends(user_or_401),  # noqa: ANN001
) -> JournalEntry:
    row = get_auth_store().update_journal_entry(
        user["id"], entry_id, title=body.title, body=body.body
    )
    if row is None:
        raise _not_found("journal_entry_not_found", str(entry_id))
    return JournalEntry(**row)


@router.delete("/journal/{entry_id}", response_model=JournalEntry)
def delete_journal(
    entry_id: int, user=Depends(user_or_401)  # noqa: ANN001
) -> JournalEntry:
    store = get_auth_store()
    row = store.get_journal_entry(user["id"], entry_id)
    if row is None or not store.delete_journal_entry(user["id"], entry_id):
        raise _not_found("journal_entry_not_found", str(entry_id))
    return JournalEntry(**row)
