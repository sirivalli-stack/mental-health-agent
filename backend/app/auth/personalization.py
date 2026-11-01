"""Mapping user settings -> the session profile the LLM receives (Phase 20).

The Phase 8 pipeline already renders a per-session :class:`UserProfile`
into the system prompt. Explicit, user-managed settings from the
Preferences screen are seeded into that profile when a conversation
starts, using only fields the pipeline already understands:

* ``tone``                -> ``communication_style`` (rendered verbatim),
* ``response_length`` / ``asks_questions`` -> factual ``preferences``
  entries (the same shape "please keep it brief" would produce in chat),
* ``preferred_language``  -> ``preferred_language``.

Nothing here is inferred about the user - every string originates in a
choice the user made on the Settings screen, which is why seeding
survives the personalization toggle: the toggle governs *learned*
extractor memory, not the settings the user typed.
"""

from __future__ import annotations

from app.models.schemas import UserProfile

# Phrased like an explicit request, because that is exactly what it is.
LENGTH_PREFERENCES = {
    "short": "prefers short replies of two or three sentences",
    "detailed": "prefers detailed, thorough replies",
}

QUESTIONS_PREFERENCE = "appreciates thoughtful follow-up questions"
NO_QUESTIONS_PREFERENCE = "prefers direct suggestions rather than questions"

# Every string the seeder can ever produce in `preferences` - used to tell
# settings-derived entries apart from ones the extractor learned in chat.
SEED_PREFERENCE_STRINGS = frozenset(
    set(LENGTH_PREFERENCES.values()) | {QUESTIONS_PREFERENCE, NO_QUESTIONS_PREFERENCE}
)


def profile_from_preferences(prefs: dict) -> UserProfile:
    """Build the seed profile for a new conversation from user settings."""
    preferences: list[str] = []
    length = prefs.get("response_length", "balanced")
    if length in LENGTH_PREFERENCES:
        preferences.append(LENGTH_PREFERENCES[length])
    preferences.append(
        QUESTIONS_PREFERENCE
        if prefs.get("asks_questions", True)
        else NO_QUESTIONS_PREFERENCE
    )
    tone = (prefs.get("tone") or "").strip().lower()
    return UserProfile(
        preferred_language=(prefs.get("preferred_language") or "en").lower(),
        communication_style=tone or None,
        preferences=preferences,
    )


def derive_title(message: str, *, max_chars: int = 48) -> str:
    """Conversation title from the first user message (fallback-friendly)."""
    text = " ".join(message.split()).strip()
    if not text:
        return "New conversation"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
