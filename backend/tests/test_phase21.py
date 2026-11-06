"""Phase 21 tests: regenerate, app settings, bulk conversation deletion.

Same infrastructure as Phase 20: ``auth_env`` (temporary SQLite + auth on)
and a faked LLM/output guardrail, with the three classifiers running for
real exactly as in the other API suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.store import AuthStore
from tests.fakes import BENIGN, BLOCKER, FakeLLM, auth_env, fast_post_gate


# ==========================================================================
# fixtures + helpers (mirrors test_phase20_auth.py)
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
        "display_name": "Regen Tester",
        "username": "regen",
        "email": "regen@example.com",
        "password": "correct-horse-1",
    }
    body.update(overrides)
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def one_turn(client: TestClient, token: str, text: str = BENIGN) -> dict:
    response = client.post(
        "/api/chat", json={"message": text}, headers=auth_headers(token)
    )
    assert response.status_code == 200, response.text
    return response.json()


def transcript(client: TestClient, token: str, session_id: str) -> list[dict]:
    response = client.get(
        f"/api/conversations/{session_id}/messages", headers=auth_headers(token)
    )
    assert response.status_code == 200, response.text
    return response.json()["messages"]


# ==========================================================================
# 1. app settings (GET/PUT /api/settings)
# ==========================================================================


def test_settings_default_to_daily_reminders_on(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        response = client.get("/api/settings", headers=auth_headers(token))
        assert response.status_code == 200
        assert response.json() == {"daily_reminders": True, "updated_at": None}


def test_settings_persist_across_requests(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        put = client.put(
            "/api/settings",
            json={"daily_reminders": False},
            headers=auth_headers(token),
        )
        assert put.status_code == 200
        assert put.json()["daily_reminders"] is False
        assert put.json()["updated_at"] is not None

        again = client.get("/api/settings", headers=auth_headers(token))
        assert again.json()["daily_reminders"] is False

        # partial bodies are fine: omitted fields keep their schema defaults
        back_on = client.put(
            "/api/settings", json={}, headers=auth_headers(token)
        )
        assert back_on.json()["daily_reminders"] is True


def test_settings_require_authentication(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        assert client.get("/api/settings").status_code == 401
        assert (
            client.put("/api/settings", json={"daily_reminders": False}).status_code
            == 401
        )


def test_settings_are_isolated_per_user(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        first = register(client)["token"]
        second = register(
            client,
            username="second",
            email="second@example.com",
            display_name="Second User",
        )["token"]

        client.put(
            "/api/settings",
            json={"daily_reminders": False},
            headers=auth_headers(first),
        )

        mine = client.get("/api/settings", headers=auth_headers(first))
        theirs = client.get("/api/settings", headers=auth_headers(second))
        assert mine.json()["daily_reminders"] is False
        assert theirs.json()["daily_reminders"] is True


# ==========================================================================
# 2. regenerate (POST /api/chat/regenerate)
# ==========================================================================


def test_regenerate_replaces_only_the_reply(tmp_path: Path, api) -> None:
    client, fake = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        first = one_turn(client, token)
        session_id = first["session_id"]
        assert first["reply"] == fake.text
        before = transcript(client, token, session_id)
        assert [m["role"] for m in before] == ["user", "assistant"]

        fake.text = "A fresh take: what part feels heaviest right now?"
        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": session_id},
            headers=auth_headers(token),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["reply"] == fake.text
        assert body["session_id"] == session_id
        assert body["turn_index"] == first["turn_index"]  # state unchanged

        after = transcript(client, token, session_id)
        # still exactly one user + one assistant turn - no duplicates
        assert [m["role"] for m in after] == ["user", "assistant"]
        assert after[0]["content"] == before[0]["content"]
        assert after[1]["content"] == fake.text
        assert after[1]["meta"]["regenerated"] is True


def test_regenerate_needs_auth(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        assert (
            client.post(
                "/api/chat/regenerate", json={"session_id": "whatever"}
            ).status_code
            == 401
        )


def test_regenerate_rejects_other_users_sessions(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        owner = register(client)["token"]
        intruder = register(
            client,
            username="intruder",
            email="intruder@example.com",
            display_name="Intruder",
        )["token"]
        session_id = one_turn(client, owner)["session_id"]

        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": session_id},
            headers=auth_headers(intruder),
        )
        # same shape as an unknown id - ownership cannot be probed
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "conversation_not_found"


def test_regenerate_requires_a_trailing_reply(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        from app.auth.store import get_auth_store

        token = register(client)["token"]
        session_id = one_turn(client, token)["session_id"]
        user_id = get_auth_store().find_user("regen")["id"]
        # append a dangling user message: transcript no longer ends in a reply
        get_auth_store().append_message(
            session_id, user_id, role="user", content="and one more thing"
        )

        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": session_id},
            headers=auth_headers(token),
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "nothing_to_regenerate"


def test_regenerate_without_session_state_is_404(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        from app.state_engine import get_state_engine

        token = register(client)["token"]
        session_id = one_turn(client, token)["session_id"]
        get_state_engine().reset(session_id)  # forget pipeline state only

        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": session_id},
            headers=auth_headers(token),
        )
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "session_not_found"


def test_regenerate_of_a_blocked_turn_stays_blocked(tmp_path: Path, api) -> None:
    """Regeneration must never slip past the pre-generation gate."""
    client, fake = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        blocked = one_turn(client, token, BLOCKER)
        assert blocked["source"] == "pre_blocked"
        assert len(fake.calls) == 0  # the model never saw the crisis message

        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": blocked["session_id"]},
            headers=auth_headers(token),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "pre_blocked"
        assert body["reply"] == blocked["reply"]
        assert len(fake.calls) == 0  # still no model call on regenerate


def test_regenerate_propagates_an_unavailable_model(tmp_path: Path, api) -> None:
    client, fake = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        session_id = one_turn(client, token)["session_id"]

        from app.services.llm_service import LLMUnavailableError

        fake.error = LLMUnavailableError("ollama is down")
        response = client.post(
            "/api/chat/regenerate",
            json={"session_id": session_id},
            headers=auth_headers(token),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "llm_unavailable"
        # the deterministic fallback replaced the stored reply
        stored = transcript(client, token, session_id)
        assert stored[-1]["content"] == body["reply"]
        assert stored[-1]["meta"]["source"] == "llm_unavailable"


# ==========================================================================
# 3. bulk deletion (DELETE /api/conversations)
# ==========================================================================


def test_delete_all_removes_only_the_callers_conversations(
    tmp_path: Path, api
) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        first = register(client)["token"]
        second = register(
            client,
            username="neighbor",
            email="neighbor@example.com",
            display_name="Neighbor",
        )["token"]

        one_turn(client, first)
        one_turn(client, first, "Help me reflect on my week.")
        kept = one_turn(client, second)["session_id"]

        response = client.delete(
            "/api/conversations", headers=auth_headers(first)
        )
        assert response.status_code == 200
        assert response.json()["deleted"] == 2

        mine = client.get("/api/conversations", headers=auth_headers(first))
        assert mine.json()["conversations"] == []

        theirs = client.get("/api/conversations", headers=auth_headers(second))
        assert [c["session_id"] for c in theirs.json()["conversations"]] == [kept]

        # the other user's transcript is still readable
        assert transcript(client, second, kept)


def test_delete_all_requires_auth(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        assert client.delete("/api/conversations").status_code == 401


def test_delete_all_on_an_empty_account_is_ok(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        response = client.delete("/api/conversations", headers=auth_headers(token))
        assert response.status_code == 200
        assert response.json()["deleted"] == 0


# ==========================================================================
# 4. store-level guarantees
# ==========================================================================


def test_account_deletion_removes_settings_rows(tmp_path: Path) -> None:
    with auth_env(tmp_path / "auth.db"):
        store = AuthStore(tmp_path / "auth.db")
        user = store.create_user(
            display_name="Temp",
            username="temp",
            email="temp@example.com",
            password="correct-horse-1",
            iterations=10_000,
        )
        store.set_settings(user["id"], {"daily_reminders": False})
        assert store.get_settings(user["id"])["daily_reminders"] is False

        store.delete_user(user["id"])
        # the row is gone: the documented default is what remains
        assert store.get_settings(user["id"])["daily_reminders"] is True
        store.close()


def test_replace_last_assistant_requires_user_then_assistant(
    tmp_path: Path,
) -> None:
    with auth_env(tmp_path / "auth.db"):
        store = AuthStore(tmp_path / "auth.db")
        user = store.create_user(
            display_name="Shape",
            username="shape",
            email="shape@example.com",
            password="correct-horse-1",
            iterations=10_000,
        )
        sid = "shape-session"
        store.upsert_conversation(sid, user["id"], title="shape")

        # ends with a user message -> refuse
        store.append_message(sid, user["id"], role="user", content="hello")
        assert (
            store.replace_last_assistant_message(
                sid, user["id"], content="nope"
            )
            is None
        )

        # ends user, assistant -> replace works
        store.append_message(sid, user["id"], role="assistant", content="hi")
        replaced = store.replace_last_assistant_message(
            sid, user["id"], content="hello again", meta={"source": "llm"}
        )
        assert replaced is not None
        assert replaced["content"] == "hello again"
        assert replaced["meta"] == {"source": "llm"}
        rows = store.messages_for_conversation(sid, user["id"])
        assert [r["content"] for r in rows] == ["hello", "hello again"]

        # only the caller's rows are visible/replaceable
        other = store.create_user(
            display_name="Other",
            username="other-shape",
            email="other-shape@example.com",
            password="correct-horse-1",
            iterations=10_000,
        )
        assert (
            store.replace_last_assistant_message(
                sid, other["id"], content="mine"
            )
            is None
        )
        store.close()
