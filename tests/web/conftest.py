"""Shared fixtures for v2.0 web backend tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Deterministic JWT secret for all tests
os.environ.setdefault("JWT_SECRET", "test-secret")

# Make the project root importable so "from web.backend..." works
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the user DB at an isolated temp file and initialise schema."""
    from web.backend import db

    test_db = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    db.init_db()
    yield test_db


@pytest.fixture
def client(tmp_db):
    """FastAPI TestClient against an isolated DB and empty session store."""
    from web.backend.app import app
    from web.backend.session_store import store

    store._sessions.clear()
    store._by_user.clear()
    return TestClient(app)


@pytest.fixture
def second_client(tmp_db):
    """Independent TestClient (so two-user tests don't share headers)."""
    from web.backend.app import app

    return TestClient(app)


@pytest.fixture
def auth_client(client):
    """TestClient with an already-registered user and bearer token set."""
    client.post(
        "/api/auth/register",
        json={"username": "u1", "email": "u1@test.io", "password": "pw12345"},
    )
    resp = client.post(
        "/api/auth/login",
        json={"username": "u1", "password": "pw12345"},
    )
    token = resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    client.auth_token = token  # type: ignore[attr-defined]
    return client


@pytest.fixture
def auth_token(auth_client):
    """Raw bearer token (for WebSocket query-string auth)."""
    return auth_client.auth_token  # type: ignore[attr-defined]


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Stub every async pipeline phase with deterministic fakes."""
    from web.backend import pipeline

    async def fake_genre(content, history, ctx, emit):
        await emit({"type": "text_delta", "content": "genre-ok"})
        return {"genre": "techno", "duration_min": 60, "mood": "dark"}

    async def fake_plan(ctx, emit, memory_summary=""):
        ctx["playlist"] = [
            {"id": "t1", "display_name": "Track 1", "bpm": 128, "camelot_key": "9A", "genre": "techno"},
            {"id": "t2", "display_name": "Track 2", "bpm": 130, "camelot_key": "10A", "genre": "techno"},
        ]
        await emit({"type": "tool_call", "name": "propose_playlist", "input": {}})

    async def fake_critique(ctx, emit, memory_summary=""):
        return ("APPROVED", [], [])

    async def fake_editor(message, history, ctx, emit):
        if message.startswith("build"):
            ctx["last_build"] = message.split(maxsplit=1)[1] if " " in message else "default"
        return "done"

    async def fake_validate(session_name, ctx, emit):
        return ("PASS", [])

    async def fake_memory(genre, ctx):
        return ""

    def fake_write(**kwargs):
        return "saved"

    monkeypatch.setattr(pipeline, "phase_genre_guard", fake_genre)
    monkeypatch.setattr(pipeline, "phase_plan", fake_plan)
    monkeypatch.setattr(pipeline, "phase_critique", fake_critique)
    monkeypatch.setattr(pipeline, "phase_editor", fake_editor)
    monkeypatch.setattr(pipeline, "phase_validate", fake_validate)
    monkeypatch.setattr(pipeline, "load_memory", fake_memory)
    monkeypatch.setattr(pipeline, "write_session_record", fake_write)
    return pipeline
