"""Request/response schemas for auth, conversations and preferences (Phase 20).

Validation is strict but dependency-free (no ``EmailStr`` - that would add
the ``email-validator`` package; a conservative regex + length caps do the
job and are unit-tested instead).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
# Deliberately conservative: one "@", no spaces, a dot in the domain.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")

MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 200
MAX_TITLE_CHARS = 80


def validate_username(value: str) -> str:
    """3-32 chars: letters/digits first, then ``_ . -`` allowed."""
    v = value.strip()
    if not USERNAME_PATTERN.match(v):
        raise ValueError(
            "username must be 3-32 characters, start with a letter or digit, "
            "and use only letters, digits, '_', '.' or '-'"
        )
    return v


def validate_email(value: str) -> str:
    v = value.strip().lower()
    if len(v) > 254 or not EMAIL_PATTERN.match(v):
        raise ValueError("enter a valid email address")
    return v


def validate_password(value: str) -> str:
    if not MIN_PASSWORD_CHARS <= len(value) <= MAX_PASSWORD_CHARS:
        raise ValueError(
            f"password must be at least {MIN_PASSWORD_CHARS} characters"
        )
    return value


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)
    username: str
    email: str
    password: str

    @field_validator("display_name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("display name must not be empty")
        return v

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        return validate_username(v)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return validate_email(v)

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        return validate_password(v)


class LoginRequest(BaseModel):
    """``identifier`` is an email *or* a username (both are supported)."""

    identifier: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)
    remember: bool = False

    @field_validator("identifier")
    @classmethod
    def _clean_identifier(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("enter your email or username")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _password(cls, v: str) -> str:
        return validate_password(v)


class UpdateProfileRequest(BaseModel):
    """Editable profile fields. Omitted fields stay unchanged."""

    display_name: str | None = Field(default=None, max_length=80)
    username: str | None = None
    email: str | None = None

    @field_validator("display_name")
    @classmethod
    def _clean_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        if not v:
            raise ValueError("display name must not be empty")
        return v

    @field_validator("username")
    @classmethod
    def _username(cls, v: str | None) -> str | None:
        return validate_username(v) if v is not None else None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str | None) -> str | None:
        return validate_email(v) if v is not None else None


class UserPublic(BaseModel):
    id: str
    username: str
    email: str
    display_name: str
    created_at: datetime


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


class MessageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    detail: str | None = None


# ---------------------------------------------------------------------------
# Conversations + messages
# ---------------------------------------------------------------------------


class ConversationPublic(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationPublic] = Field(default_factory=list)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("title must not be empty")
        return v


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    meta: dict | None = None


class MessageListResponse(BaseModel):
    session_id: str
    title: str
    messages: list[StoredMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Preferences + personalization
# ---------------------------------------------------------------------------

ResponseLength = Literal["short", "balanced", "detailed"]
Tone = Literal["warm", "gentle", "direct"]


class UserPreferences(BaseModel):
    """Explicit, user-managed conversation settings (never inferred)."""

    response_length: ResponseLength = "balanced"
    tone: Tone = "warm"
    asks_questions: bool = True
    preferred_language: str = "en"
    personalization_enabled: bool = True
    updated_at: datetime | None = None

    @field_validator("preferred_language")
    @classmethod
    def _known_language(cls, v: str) -> str:
        from app.personalization.extractor import LANGUAGE_CODES

        code = (v or "en").strip().lower()
        if code not in set(LANGUAGE_CODES.values()):
            raise ValueError(f"unsupported language code: {code!r}")
        return code


class LearnedItem(BaseModel):
    """One learned fact, aggregated across the user's conversations."""

    kind: str          # name | communication_style | preferred_language | preference | topic_to_avoid
    value: str
    conversations: int


class PersonalizationView(BaseModel):
    enabled: bool
    items: list[LearnedItem] = Field(default_factory=list)
    conversations_with_learnings: int = 0


class PersonalizationClearResponse(BaseModel):
    cleared_conversations: int
    status: Literal["cleared"] = "cleared"
