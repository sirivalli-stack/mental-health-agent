"""Phase 23 tests: wellness (daily mood check-ins + private journal).

Covers the new additive tables and routes: validation (seven moods,
YYYY-MM-DD dates, length caps), one-check-in-per-day upsert semantics,
full journal CRUD, strict owner scoping (foreign ids are 404s), 401s,
and account-deletion cleanup. Nothing here touches the ML pipeline -
wellness records are self-reported and never analysed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.store import AuthStore
from tests.fakes import auth_env, fast_post_gate, FakeLLM


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


def register(client: TestClient, name: str = "wellness", **overrides) -> dict:
    body = {
        "display_name": "Wellness Tester",
        "username": name,
        "email": f"{name}@example.com",
        "password": "correct-horse-1",
    }
    body.update(overrides)
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==========================================================================
# Mood check-ins
# ==========================================================================


def test_mood_requires_auth() -> None:
    from main import app

    with TestClient(app) as client:
        assert client.get("/api/moods").status_code == 401
        assert client.put("/api/moods", json={"mood": "calm"}).status_code == 401
        assert client.delete("/api/moods/1").status_code == 401
        assert client.get("/api/journal").status_code == 401


def test_put_mood_defaults_to_today_and_lists_newest_first(
    tmp_path: Path, api
) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)

        first = client.put(
            "/api/moods", json={"mood": "calm", "note": "quiet morning"}, headers=headers
        )
        assert first.status_code == 200, first.text
        entry = first.json()
        assert entry["mood"] == "calm"
        assert entry["note"] == "quiet morning"
        assert len(entry["date"]) == 10  # YYYY-MM-DD

        # a dated entry in the past sorts after today (newest first)
        older = client.put(
            "/api/moods",
            json={"mood": "anxious", "date": "2026-01-05"},
            headers=headers,
        )
        assert older.status_code == 200, older.text

        listed = client.get("/api/moods", headers=headers).json()["entries"]
        assert len(listed) == 2
        assert listed[0]["date"] >= listed[1]["date"]
        assert {e["mood"] for e in listed} == {"calm", "anxious"}


def test_rechecking_the_same_day_replaces_not_duplicates(
    tmp_path: Path, api
) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)

        client.put("/api/moods", json={"mood": "sad"}, headers=headers)
        client.put("/api/moods", json={"mood": "happy", "note": "better now!"}, headers=headers)

        listed = client.get("/api/moods", headers=headers).json()["entries"]
        assert len(listed) == 1  # one honest record per date
        assert listed[0]["mood"] == "happy"
        assert listed[0]["note"] == "better now!"


def test_mood_validation_rejects_unknown_moods_and_bad_dates(
    tmp_path: Path, api
) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)

        # never invent moods outside the seven
        assert (
            client.put("/api/moods", json={"mood": "depressed"}, headers=headers).status_code
            == 422
        )
        assert (
            client.put("/api/moods", json={"mood": "great"}, headers=headers).status_code
            == 422
        )
        # date must be a real ISO day
        assert (
            client.put(
                "/api/moods", json={"mood": "calm", "date": "05/11/2026"}, headers=headers
            ).status_code
            == 422
        )
        # note length cap
        assert (
            client.put(
                "/api/moods", json={"mood": "calm", "note": "x" * 2001}, headers=headers
            ).status_code
            == 422
        )
        # nothing was stored by the rejected calls
        assert client.get("/api/moods", headers=headers).json()["entries"] == []


def test_mood_entries_are_private_per_user(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        alice = register(client, name="alice")["token"]
        bob = register(client, name="bob")["token"]

        created = client.put(
            "/api/moods", json={"mood": "happy"}, headers=auth_headers(alice)
        ).json()

        # bob's list never contains alice's row
        bob_list = client.get("/api/moods", headers=auth_headers(bob)).json()["entries"]
        assert bob_list == []

        # bob cannot delete alice's entry (404, same as unknown id)
        assert (
            client.delete(
                f"/api/moods/{created['id']}", headers=auth_headers(bob)
            ).status_code
            == 404
        )
        # ...and alice still has it
        alice_list = client.get("/api/moods", headers=auth_headers(alice)).json()["entries"]
        assert len(alice_list) == 1


def test_delete_own_mood_entry(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)
        created = client.put("/api/moods", json={"mood": "angry"}, headers=headers).json()

        deleted = client.delete(f"/api/moods/{created['id']}", headers=headers)
        assert deleted.status_code == 200
        assert deleted.json()["mood"] == "angry"
        assert client.get("/api/moods", headers=headers).json()["entries"] == []
        # deleting again -> 404 (already gone)
        assert (
            client.delete(f"/api/moods/{created['id']}", headers=headers).status_code
            == 404
        )


# ==========================================================================
# Journal
# ==========================================================================


def test_journal_full_crud(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)

        created = client.post(
            "/api/journal",
            json={"title": "First entry", "body": "Today I felt steady."},
            headers=headers,
        )
        assert created.status_code == 201, created.text
        entry = created.json()
        assert entry["title"] == "First entry"
        assert entry["body"] == "Today I felt steady."

        # partial update: title only leaves the body alone
        updated = client.put(
            f"/api/journal/{entry['id']}",
            json={"title": "Renamed"},
            headers=headers,
        )
        assert updated.status_code == 200
        assert updated.json()["title"] == "Renamed"
        assert updated.json()["body"] == "Today I felt steady."

        # body-only update leaves the title alone
        updated2 = client.put(
            f"/api/journal/{entry['id']}",
            json={"body": "And tomorrow, too."},
            headers=headers,
        )
        assert updated2.json()["title"] == "Renamed"
        assert updated2.json()["body"] == "And tomorrow, too."

        listed = client.get("/api/journal", headers=headers).json()["entries"]
        assert len(listed) == 1
        fetched = client.get(f"/api/journal/{entry['id']}", headers=headers)
        assert fetched.status_code == 200

        deleted = client.delete(f"/api/journal/{entry['id']}", headers=headers)
        assert deleted.status_code == 200
        assert client.get("/api/journal", headers=headers).json()["entries"] == []
        assert (
            client.get(f"/api/journal/{entry['id']}", headers=headers).status_code
            == 404
        )


def test_journal_requires_text_and_caps_length(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)

        # empty body is rejected (a journal entry must say something)
        assert (
            client.post("/api/journal", json={"body": "   "}, headers=headers).status_code
            == 422
        )
        # body cap
        assert (
            client.post(
                "/api/journal", json={"body": "x" * 20_001}, headers=headers
            ).status_code
            == 422
        )
        # title cap
        assert (
            client.post(
                "/api/journal",
                json={"title": "t" * 121, "body": "ok"},
                headers=headers,
            ).status_code
            == 422
        )
        # title-only entry (no body) is rejected too
        assert (
            client.post("/api/journal", json={"title": "t"}, headers=headers).status_code
            == 422
        )


def test_journal_is_private_per_user(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        alice = register(client, name="alice2")["token"]
        bob = register(client, name="bob2")["token"]

        created = client.post(
            "/api/journal",
            json={"title": "Secret", "body": "Only mine."},
            headers=auth_headers(alice),
        ).json()

        assert client.get("/api/journal", headers=auth_headers(bob)).json()["entries"] == []
        # cross-user reads/writes/deletes are all 404
        assert (
            client.get(
                f"/api/journal/{created['id']}", headers=auth_headers(bob)
            ).status_code
            == 404
        )
        assert (
            client.put(
                f"/api/journal/{created['id']}",
                json={"body": "hacked"},
                headers=auth_headers(bob),
            ).status_code
            == 404
        )
        assert (
            client.delete(
                f"/api/journal/{created['id']}", headers=auth_headers(bob)
            ).status_code
            == 404
        )
        # alice's entry is untouched
        still = client.get(
            f"/api/journal/{created['id']}", headers=auth_headers(alice)
        ).json()
        assert still["body"] == "Only mine."


def test_unknown_journal_ids_are_404(tmp_path: Path, api) -> None:
    client, _ = api
    with auth_env(tmp_path / "auth.db"):
        token = register(client)["token"]
        headers = auth_headers(token)
        assert client.get("/api/journal/999999", headers=headers).status_code == 404
        assert (
            client.put("/api/journal/999999", json={"body": "x"}, headers=headers).status_code
            == 404
        )
        assert client.delete("/api/journal/999999", headers=headers).status_code == 404


# ==========================================================================
# Store-level: cleanup on account deletion
# ==========================================================================


def test_account_deletion_removes_wellness_rows(tmp_path: Path) -> None:
    with auth_env(tmp_path / "auth.db"):
        store = AuthStore(tmp_path / "auth.db")
        try:
            user = store.create_user(
                username="cleanup",
                email="cleanup@example.com",
                display_name="Cleanup",
                password="correct-horse-1",
                iterations=10_000,
            )
            uid = user["id"]
            store.upsert_mood_entry(uid, "2026-11-05", "calm", "note")
            store.create_journal_entry(uid, "t", "body")
            assert store.list_mood_entries(uid)
            assert store.list_journal_entries(uid)

            store.delete_user(uid)
            assert store.list_mood_entries(uid) == []
            assert store.list_journal_entries(uid) == []
        finally:
            store.close()


def test_store_mood_upsert_is_per_user_per_day(tmp_path: Path) -> None:
    with auth_env(tmp_path / "auth.db"):
        store = AuthStore(tmp_path / "auth.db")
        try:
            a = store.create_user(
                username="up-a", email="up-a@example.com", display_name="A",
                password="correct-horse-1", iterations=10_000,
            )["id"]
            b = store.create_user(
                username="up-b", email="up-b@example.com", display_name="B",
                password="correct-horse-1", iterations=10_000,
            )["id"]

            store.upsert_mood_entry(a, "2026-11-05", "sad")
            store.upsert_mood_entry(a, "2026-11-05", "happy", "changed")
            store.upsert_mood_entry(b, "2026-11-05", "angry")

            a_entries = store.list_mood_entries(a)
            assert len(a_entries) == 1
            assert a_entries[0]["mood"] == "happy"
            assert a_entries[0]["note"] == "changed"
            # same date, different user, independent row
            assert store.list_mood_entries(b)[0]["mood"] == "angry"
        finally:
            store.close()
