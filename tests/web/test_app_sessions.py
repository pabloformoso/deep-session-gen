"""Integration tests for /api/sessions CRUD + rating endpoints."""
from __future__ import annotations


def test_create_session_returns_fresh_session(auth_client):
    r = auth_client.post("/api/sessions")
    assert r.status_code == 200
    data = r.json()
    assert data["id"]
    assert data["phase"] == "init"
    assert data["playlist"] == []


def test_list_sessions_filters_by_user(auth_client, second_client):
    # user 1 creates two sessions
    auth_client.post("/api/sessions")
    auth_client.post("/api/sessions")

    # user 2 registers on an independent client, should see zero sessions
    second_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@t.io", "password": "pw12345"},
    )
    token = second_client.post(
        "/api/auth/login", json={"username": "u2", "password": "pw12345"}
    ).json()["access_token"]

    u1_list = auth_client.get("/api/sessions").json()
    u2_list = second_client.get(
        "/api/sessions", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert len(u1_list) == 2
    assert u2_list == []


def test_get_own_session_returns_200(auth_client):
    sid = auth_client.post("/api/sessions").json()["id"]
    r = auth_client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["id"] == sid


def test_get_other_users_session_returns_404(auth_client, second_client):
    sid = auth_client.post("/api/sessions").json()["id"]
    second_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@t.io", "password": "pw12345"},
    )
    token = second_client.post(
        "/api/auth/login", json={"username": "u2", "password": "pw12345"}
    ).json()["access_token"]

    r = second_client.get(f"/api/sessions/{sid}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404


def test_delete_own_session(auth_client):
    sid = auth_client.post("/api/sessions").json()["id"]
    r = auth_client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 204
    assert auth_client.get(f"/api/sessions/{sid}").status_code == 404


def test_delete_other_users_session_returns_404(auth_client, second_client):
    sid = auth_client.post("/api/sessions").json()["id"]
    second_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@t.io", "password": "pw12345"},
    )
    token = second_client.post(
        "/api/auth/login", json={"username": "u2", "password": "pw12345"}
    ).json()["access_token"]
    r = second_client.delete(f"/api/sessions/{sid}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404


def test_rate_session_marks_complete(auth_client, mock_pipeline):
    sid = auth_client.post("/api/sessions").json()["id"]
    r = auth_client.post(f"/api/sessions/{sid}/rate", json={"rating": 5, "notes": "great"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    state = auth_client.get(f"/api/sessions/{sid}").json()
    assert state["phase"] == "complete"


def test_rate_unknown_session_returns_404(auth_client, mock_pipeline):
    r = auth_client.post("/api/sessions/nonexistent/rate", json={"rating": 3})
    assert r.status_code == 404
