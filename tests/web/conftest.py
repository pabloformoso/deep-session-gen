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

    # Reset cache (and its "loaded from DB" flag) so the store re-reads from
    # the per-test tmp DB instead of whatever prior tests wrote.
    store._reset()
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
    """Stub every async pipeline phase with the shared deterministic fakes."""
    from web.backend import mock_pipeline as fakes
    from web.backend import pipeline

    monkeypatch.setattr(pipeline, "phase_genre_guard", fakes.fake_genre)
    monkeypatch.setattr(pipeline, "phase_plan", fakes.fake_plan)
    monkeypatch.setattr(pipeline, "phase_critique", fakes.fake_critique)
    monkeypatch.setattr(pipeline, "phase_editor", fakes.fake_editor)
    monkeypatch.setattr(pipeline, "phase_validate", fakes.fake_validate)
    monkeypatch.setattr(pipeline, "load_memory", fakes.fake_memory)
    monkeypatch.setattr(pipeline, "write_session_record", fakes.fake_write)
    monkeypatch.setattr(pipeline, "check_catalog", fakes.fake_check_catalog)
    return pipeline
