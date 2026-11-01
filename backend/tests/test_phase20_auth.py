"""Phase 20 tests: authentication, ownership, history, preferences.

Everything runs against a temporary SQLite file (``auth_env``); the LLM and
the output guardrail are faked exactly as in Phases 12/16. The three
classifier services run for real, as they do in the other API suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.personalization import derive_title, profile_from_preferences
from app.auth.security import (
    hash_password,
    new_token,
    token_digest,
    verify_password,
)
from app.auth.store import AuthStore, reset_auth_store
from tests.fakes import BENIGN, FakeLLM, auth_env, fast_post_gate


# ==========================================================================
# fixtures
# ==========================================================================


@pytest.fixture()
def api(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    """TestClient with LLM + guardrail faked (no network, no BERT)."""
    from app.pipelines import chat as chat_mod

    fake = FakeLLM()
    monkeypatch.setattr(chat_mod, "get_llm_service", lambda: fake)
    monkeypatch.setattr(chat_mod, "build_post_gate", fast_post_gate)
    from main import app

    with TestClient(app) as client:
        yield client, fake


def register(client: TestClient, **overrides) -> dict:
    body = {
        "display_name": "Alex Rivera",
        "username": "alex",
        "email": "alex@example.com",
        "password": "correct-horse-1",
    }
    body.update(overrides)
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==========================================================================
# 1. security primitives
# ==========================================================================


def test_password_hash_roundtrip_and_rejection() -> None:
    stored = hash_password("correct-horse-1", iterations=10_000)
    assert stored.startswith("pbkdf2_sha256$10000$")
    assert "correct-horse-1" not in stored
    assert verify_password("correct-horse-1", stored)
    assert not verify_password("correct-horse-2", stored)
    assert not verify_password("", stored)
    assert not verify_password("x", "not-a-digest")
    assert not verify_password("x", "pbkdf2_sha256$abc$zz$yy")


def test_password_hashes_are_salted() -> None:
    a = hash_password("same-password", iterations=10_000)
    b = hash_password("same-password", iterations=10_000)
    assert a != b  # distinct salts
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_tokens_are_random_and_digested() -> None:
    a, b = new_token(), new_token()
    assert a != b and len(a) >= 40
    assert token_digest(a) == token_digest(a) != a
    assert token_digest(a) != token_digest(b)


def test_seed_profile_maps_settings() -> None:
    profile = profile_from_preferences(
        {
            "response_length": "short",
            "tone": "gentle",
            "asks_questions": False,
            "preferred_language": "ES",
        }
    )
    assert profile.communication_style == "gentle"
    assert profile.preferred_language == "es"
    assert "prefers short replies of two or three sentences" in profile.preferences
    assert "prefers direct suggestions rather than questions" in profile.preferences


def test_derive_title_truncates_and_falls_back() -> None:
    assert derive_title("  how   are  you ") == "how are you"
    assert derive_title("x" * 80) == "x" * 47 + "…"
    assert derive_title("   ") == "New conversation"


# ==========================================================================
# 2. auth endpoints
# ==========================================================================


def test_register_login_me_logout(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        body = register(client)
        assert body["user"]["display_name"] == "Alex Rivera"
        assert body["user"]["username"] == "alex"
        assert "password" not in body["user"]
        assert body["token"]

        me = client.get("/api/auth/me", headers=auth_headers(body["token"]))
        assert me.status_code == 200
        assert me.json()["email"] == "alex@example.com"

        # login by email and by username, both work
        by_email = client.post(
            "/api/auth/login",
            json={"identifier": "alex@example.com", "password": "correct-horse-1"},
        )
        by_name = client.post(
            "/api/auth/login",
            json={"identifier": "ALEX", "password": "correct-horse-1"},
        )
        assert by_email.status_code == 200 and by_name.status_code == 200

        out = client.post(
            "/api/auth/logout", headers=auth_headers(body["token"])
        )
        assert out.status_code == 200
        after = client.get("/api/auth/me", headers=auth_headers(body["token"]))
        assert after.status_code == 401


def test_login_failures_are_generic(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        register(client)
        wrong_password = client.post(
            "/api/auth/login",
            json={"identifier": "alex", "password": "wrong-password-1"},
        )
        unknown_user = client.post(
            "/api/auth/login",
            json={"identifier": "nobody", "password": "wrong-password-1"},
        )
        assert wrong_password.status_code == 401
        assert unknown_user.status_code == 401
        # identical body: the endpoint never reveals which accounts exist
        assert wrong_password.json() == unknown_user.json()
        assert "incorrect" in wrong_password.json()["detail"]["detail"]


def test_register_validation_and_conflicts(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        register(client)
        duplicate = client.post(
            "/api/auth/register",
            json={
                "display_name": "Other",
                "username": "alex",
                "email": "other@example.com",
                "password": "another-pass-1",
            },
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"]["error"] == "registration_conflict"

        short_password = client.post(
            "/api/auth/register",
            json={
                "display_name": "Sam",
                "username": "sam",
                "email": "sam@example.com",
                "password": "short",
            },
        )
        assert short_password.status_code == 422
        bad_email = client.post(
            "/api/auth/register",
            json={
                "display_name": "Sam",
                "username": "sam",
                "email": "not-an-email",
                "password": "long-enough-pass",
            },
        )
        assert bad_email.status_code == 422
        bad_username = client.post(
            "/api/auth/register",
            json={
                "display_name": "Sam",
                "username": "a",
                "email": "sam@example.com",
                "password": "long-enough-pass",
            },
        )
        assert bad_username.status_code == 422


def test_me_requires_a_token(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        assert client.get("/api/auth/me").status_code == 401
        assert client.get(
            "/api/auth/me", headers=auth_headers("bogus-token")
        ).status_code == 401


def test_profile_edit_and_password_change(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        body = register(client)
        headers = auth_headers(body["token"])

        edited = client.patch(
            "/api/auth/profile",
            json={"display_name": "Alex R.", "username": "alexr"},
            headers=headers,
        )
        assert edited.status_code == 200
        assert edited.json()["display_name"] == "Alex R."
        assert edited.json()["username"] == "alexr"

        bad_current = client.post(
            "/api/auth/password",
            json={"current_password": "nope", "new_password": "new-pass-123"},
            headers=headers,
        )
        assert bad_current.status_code == 401

        changed = client.post(
            "/api/auth/password",
            json={
                "current_password": "correct-horse-1",
                "new_password": "new-pass-123",
            },
            headers=headers,
        )
        assert changed.status_code == 200
        # every token was revoked, the old one included
        assert client.get("/api/auth/me", headers=headers).status_code == 401
        relogin = client.post(
            "/api/auth/login",
            json={"identifier": "alexr", "password": "new-pass-123"},
        )
        assert relogin.status_code == 200


def test_account_deletion_removes_everything(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, fake = api
        body = register(client)
        headers = auth_headers(body["token"])
        chat = client.post(
            "/api/chat", json={"message": BENIGN}, headers=headers
        ).json()
        session_id = chat["session_id"]

        deleted = client.request(
            "DELETE", "/api/auth/account", headers=headers
        )
        assert deleted.status_code == 200
        assert client.get("/api/auth/me", headers=headers).status_code == 401
        # the conversation is gone for everyone: no token can reach it now
        assert client.get(
            f"/api/conversations/{session_id}", headers=headers
        ).status_code == 401
        other = register(
            client,
            display_name="Bo",
            username="bobby",
            email="bo@example.com",
        )
        assert client.get(
            f"/api/conversations/{session_id}",
            headers=auth_headers(other["token"]),
        ).status_code == 404


# ==========================================================================
# 3. chat + ownership
# ==========================================================================


def test_chat_requires_auth_and_registers_the_conversation(
    tmp_path: Path, api
) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, fake = api
        assert client.post("/api/chat", json={"message": BENIGN}).status_code == 401

        body = register(client)
        headers = auth_headers(body["token"])
        response = client.post(
            "/api/chat", json={"message": BENIGN}, headers=headers
        )
        assert response.status_code == 200
        chat = response.json()
        assert chat["reply"] == fake.text
        session_id = chat["session_id"]

        conversations = client.get(
            "/api/conversations", headers=headers
        ).json()["conversations"]
        assert len(conversations) == 1
        assert conversations[0]["session_id"] == session_id
        # title derived from the first message (length-capped)
        assert conversations[0]["title"] == derive_title(BENIGN)
        assert conversations[0]["message_count"] == 2  # user + assistant

        messages = client.get(
            f"/api/conversations/{session_id}/messages", headers=headers
        ).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == BENIGN
        assert messages[1]["content"] == fake.text
        assert messages[1]["meta"]["source"] == "llm"

        # continuing keeps one conversation and one stored message per turn
        client.post(
            "/api/chat",
            json={"message": "thanks, that helps", "session_id": session_id},
            headers=headers,
        )
        after = client.get(
            f"/api/conversations/{session_id}/messages", headers=headers
        ).json()["messages"]
        assert len(after) == 4
        listed = client.get("/api/conversations", headers=headers).json()
        assert len(listed["conversations"]) == 1
        assert listed["conversations"][0]["message_count"] == 4


def test_conversations_are_isolated_between_users(
    tmp_path: Path, api
) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        bo = register(
            client, display_name="Bo", username="bobby", email="bo@example.com"
        )
        alex_h, bo_h = auth_headers(alex["token"]), auth_headers(bo["token"])
        session_id = client.post(
            "/api/chat", json={"message": BENIGN}, headers=alex_h
        ).json()["session_id"]

        # Bo cannot list, read, rename, delete or continue Alex's session
        assert client.get("/api/conversations", headers=bo_h).json() == {
            "conversations": []
        }
        assert client.get(
            f"/api/conversations/{session_id}", headers=bo_h
        ).status_code == 404
        assert client.get(
            f"/api/conversations/{session_id}/messages", headers=bo_h
        ).status_code == 404
        assert client.patch(
            f"/api/conversations/{session_id}",
            json={"title": "hijacked"},
            headers=bo_h,
        ).status_code == 404
        assert client.delete(
            f"/api/conversations/{session_id}", headers=bo_h
        ).status_code == 404
        assert client.post(
            "/api/chat",
            json={"message": BENIGN, "session_id": session_id},
            headers=bo_h,
        ).status_code == 404
        assert client.get(
            f"/api/sessions/{session_id}", headers=bo_h
        ).status_code == 404
        assert client.delete(
            f"/api/sessions/{session_id}", headers=bo_h
        ).status_code == 404

        # ... while Alex still can
        assert client.get(
            f"/api/sessions/{session_id}", headers=alex_h
        ).status_code == 200


def test_rename_and_delete_conversation(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        headers = auth_headers(alex["token"])
        session_id = client.post(
            "/api/chat", json={"message": BENIGN}, headers=headers
        ).json()["session_id"]

        renamed = client.patch(
            f"/api/conversations/{session_id}",
            json={"title": "Exam week feelings"},
            headers=headers,
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Exam week feelings"
        # a later turn does not overwrite the user's rename
        client.post(
            "/api/chat",
            json={"message": "still stressed", "session_id": session_id},
            headers=headers,
        )
        assert client.get(
            f"/api/conversations/{session_id}", headers=headers
        ).json()["title"] == "Exam week feelings"

        deleted = client.delete(
            f"/api/conversations/{session_id}", headers=headers
        )
        assert deleted.status_code == 200
        assert client.get(
            f"/api/conversations/{session_id}/messages", headers=headers
        ).status_code == 404
        assert client.get(
            f"/api/sessions/{session_id}", headers=headers
        ).status_code == 404
        assert client.get(
            "/api/conversations", headers=headers
        ).json()["conversations"] == []


def test_preferences_seed_new_conversations_only(
    tmp_path: Path, api
) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        headers = auth_headers(alex["token"])

        saved = client.put(
            "/api/preferences",
            json={
                "response_length": "short",
                "tone": "gentle",
                "asks_questions": False,
                "preferred_language": "en",
                "personalization_enabled": True,
            },
            headers=headers,
        )
        assert saved.status_code == 200
        assert client.get(
            "/api/preferences", headers=headers
        ).json()["response_length"] == "short"

        first = client.post(
            "/api/chat", json={"message": BENIGN}, headers=headers
        ).json()
        seeded = first["state"]["profile"]
        assert seeded["communication_style"] == "gentle"
        assert any("short replies" in p for p in seeded["preferences"])

        # the seed reaches the model prompt (profile D renders the block)
        # and existing conversations keep their profile across a settings
        # change - the settings apply to *new* conversations.
        client.put(
            "/api/preferences",
            json={
                "response_length": "detailed",
                "tone": "direct",
                "asks_questions": True,
                "preferred_language": "en",
                "personalization_enabled": True,
            },
            headers=headers,
        )
        second = client.post(
            "/api/chat",
            json={"message": BENIGN, "session_id": first["session_id"]},
            headers=headers,
        ).json()
        assert second["state"]["profile"]["communication_style"] == "gentle"


def test_preferences_validation(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        headers = auth_headers(alex["token"])
        bad = client.put(
            "/api/preferences",
            json={
                "response_length": "novel",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "en",
                "personalization_enabled": True,
            },
            headers=headers,
        )
        assert bad.status_code == 422
        bad_language = client.put(
            "/api/preferences",
            json={
                "response_length": "balanced",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "klingon",
                "personalization_enabled": True,
            },
            headers=headers,
        )
        assert bad_language.status_code == 422
        assert client.get("/api/preferences").status_code == 401


# ==========================================================================
# 4. personalization transparency + control
# ==========================================================================


def test_personalization_view_and_clear(tmp_path: Path, api) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        headers = auth_headers(alex["token"])
        empty = client.get("/api/personalization", headers=headers).json()
        assert empty == {
            "enabled": True,
            "items": [],
            "conversations_with_learnings": 0,
        }

        client.post(
            "/api/chat",
            json={"message": "my name is Sam and please keep it brief"},
            headers=headers,
        )
        view = client.get("/api/personalization", headers=headers).json()
        kinds = {(i["kind"], i["value"]) for i in view["items"]}
        assert ("name", "Sam") in kinds
        assert ("communication_style", "brief") in kinds
        assert view["conversations_with_learnings"] == 1

        cleared = client.delete("/api/personalization", headers=headers)
        assert cleared.status_code == 200
        assert cleared.json()["cleared_conversations"] == 1
        after = client.get("/api/personalization", headers=headers).json()
        assert after["items"] == []
        assert after["conversations_with_learnings"] == 0


def test_personalization_disabled_stops_learning(
    tmp_path: Path, api
) -> None:  # noqa: ANN001
    with auth_env(tmp_path / "auth.db"):
        client, _ = api
        alex = register(client)
        headers = auth_headers(alex["token"])
        client.put(
            "/api/preferences",
            json={
                "response_length": "balanced",
                "tone": "warm",
                "asks_questions": True,
                "preferred_language": "en",
                "personalization_enabled": False,
            },
            headers=headers,
        )
        response = client.post(
            "/api/chat",
            json={"message": "my name is Sam and please keep it brief"},
            headers=headers,
        ).json()
        profile = response["state"]["profile"]
        assert profile["name"] is None
        assert profile["communication_style"] == "warm"  # the setting, not a learning
        view = client.get("/api/personalization", headers=headers).json()
        assert view["enabled"] is False
        assert view["items"] == []


# ==========================================================================
# 5. store internals + anonymous mode
# ==========================================================================


def test_auth_store_rejects_non_database_file(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.db"
    garbage.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(ValueError):
        AuthStore(garbage)
    reset_auth_store(close=False)


def test_anonymous_mode_keeps_legacy_behaviour(tmp_path: Path, api) -> None:
    """AUTH_ENABLED=false: no token needed, nothing is persisted per user."""
    import os
    from contextlib import contextmanager

    from app.config.settings import reset_settings_cache
    from app.state_engine import reset_state_engine

    @contextmanager
    def anonymous():
        old = os.environ.get("AUTH_ENABLED")
        os.environ["AUTH_ENABLED"] = "false"
        reset_settings_cache()
        reset_state_engine()
        try:
            yield
        finally:
            reset_state_engine()
            reset_settings_cache()
            if old is not None:
                os.environ["AUTH_ENABLED"] = old

    with anonymous():
        client, fake = api
        response = client.post("/api/chat", json={"message": BENIGN})
        assert response.status_code == 200
        assert response.json()["reply"] == fake.text
        assert client.get("/api/conversations").status_code == 401
        assert client.get("/api/auth/me").status_code == 401
