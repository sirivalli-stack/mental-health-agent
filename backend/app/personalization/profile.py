"""Merging and rendering user profiles (Phase 8). [OURS]

Two pure functions, both unit-tested and side-effect free:

* :func:`merge_profile` - apply one turn's :class:`ProfileUpdate` to the
  stored profile. Scalars are overwritten only when the user stated them
  again; lists are appended, de-duplicated case-insensitively and capped at
  ``max_items`` (keeping the most recent entries). ``max_items=0`` disables
  list accumulation entirely - useful for ablations.
* :func:`render_profile` - turn the profile into the compact block the LLM
  receives. Only what the user actually said is rendered: the
  default language is not reported as a "learning", and nothing is invented.
"""

from __future__ import annotations

from app.models.schemas import UserProfile
from app.personalization.extractor import LANGUAGE_CODES, ProfileUpdate

DEFAULT_PROFILE_MAX_ITEMS = 8

_REVERSE_LANGUAGES = {code: name.title() for name, code in LANGUAGE_CODES.items()}


def merge_profile(
    current: UserProfile,
    update: ProfileUpdate,
    *,
    max_items: int = DEFAULT_PROFILE_MAX_ITEMS,
) -> UserProfile:
    """Return ``current`` with ``update`` applied (never mutates ``current``)."""
    if update.is_empty():
        return current.model_copy(deep=True)

    data = current.model_dump()
    if update.name:
        data["name"] = " ".join(update.name.split())
    if update.preferred_language:
        data["preferred_language"] = update.preferred_language.strip().lower()
    if update.communication_style:
        data["communication_style"] = update.communication_style.strip().lower()

    for field_name in ("preferences", "topics_to_avoid"):
        items: list[str] = list(getattr(current, field_name))
        for incoming in getattr(update, field_name):
            incoming = " ".join(incoming.split())
            if not incoming:
                continue
            if not any(existing.casefold() == incoming.casefold() for existing in items):
                items.append(incoming)
        if max_items == 0:
            items = []                          # accumulation disabled
        elif max_items > 0 and len(items) > max_items:
            items = items[-max_items:]          # newest survive the cap
        data[field_name] = items

    return UserProfile(**data)


def has_learnings(profile: UserProfile) -> bool:
    """True when at least one turn told us something about this user."""
    return bool(
        profile.name
        or profile.communication_style
        or profile.preferred_language != "en"
        or profile.preferences
        or profile.topics_to_avoid
    )


def render_profile(profile: UserProfile) -> str:
    """Compact, factual block for the system prompt (``""`` if nothing known)."""
    if not has_learnings(profile):
        return ""

    lines = [
        "Known about this user (from what they told you earlier - "
        "do not ask them again):"
    ]
    if profile.name:
        lines.append(f"- preferred name: {profile.name}")
    if profile.preferred_language and profile.preferred_language != "en":
        language = _REVERSE_LANGUAGES.get(
            profile.preferred_language, profile.preferred_language
        )
        lines.append(f"- preferred language: {language} ({profile.preferred_language})")
    if profile.communication_style:
        lines.append(f"- communication style: {profile.communication_style}")
    if profile.preferences:
        lines.append(f"- likes / prefers: {', '.join(profile.preferences)}")
    if profile.topics_to_avoid:
        lines.append(f"- topics to avoid: {', '.join(profile.topics_to_avoid)}")
    return "\n".join(lines)
