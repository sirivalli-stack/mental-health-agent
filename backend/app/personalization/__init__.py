"""Personalization (Phase 8).

Public surface::

    from app.personalization import (
        ProfileUpdate, RulePersonalizer, NullPersonalizer,
        merge_profile, render_profile,
    )
"""

from app.personalization.extractor import (
    LANGUAGE_CODES,
    LANGUAGE_PATTERN,
    MAX_FRAGMENT,
    NAME_PATTERNS,
    AVOID_PATTERN,
    PREFERENCE_PATTERNS,
    STYLE_RULES,
    NullPersonalizer,
    ProfileExtractor,
    ProfileUpdate,
    RulePersonalizer,
)
from app.personalization.profile import (
    DEFAULT_PROFILE_MAX_ITEMS,
    has_learnings,
    merge_profile,
    render_profile,
)

__all__ = [
    "AVOID_PATTERN",
    "DEFAULT_PROFILE_MAX_ITEMS",
    "LANGUAGE_CODES",
    "LANGUAGE_PATTERN",
    "MAX_FRAGMENT",
    "NAME_PATTERNS",
    "PREFERENCE_PATTERNS",
    "STYLE_RULES",
    "NullPersonalizer",
    "ProfileExtractor",
    "ProfileUpdate",
    "RulePersonalizer",
    "has_learnings",
    "merge_profile",
    "render_profile",
]
