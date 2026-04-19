"""Integration tests for /api/auth/* endpoints."""
from __future__ import annotations


def test_register_returns_token_and_user(client):
    r = client.post(
        "/api/auth/register",
        json={"username": "alice", "email": "a@t.io", "password": "pw12345"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    assert data["user"]["username"] == "alice"
    assert data["user"]["email"] == "a@t.io"


def test_register_duplicate_username_returns_400(client):
    client.post("/api/auth/register", json={"username": "alice", "email": "a@t.io", "password": "pw"})
    r = client.post("/api/auth/register", json={"username": "alice", "email": "other@t.io", "password": "pw"})
    assert r.status_code == 400
    assert "taken" in r.json()["detail"].lower()


def test_register_invalid_email_returns_422(client):
    r = client.post(
        "/api/auth/register",
        json={"username": "bob", "email": "not-an-email", "password": "pw12345"},
    )
    assert r.status_code == 422


def test_login_happy_path(client):
    client.post("/api/auth/register", json={"username": "alice", "email": "a@t.io", "password": "pw12345"})
    r = client.post("/api/auth/login", json={"username": "alice", "password": "pw12345"})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_login_wrong_password_returns_400(client):
    client.post("/api/auth/register", json={"username": "alice", "email": "a@t.io", "password": "pw12345"})
    r = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 400


def test_login_unknown_user_returns_400(client):
    r = client.post("/api/auth/login", json={"username": "ghost", "password": "pw"})
    assert r.status_code == 400


def test_me_without_token_returns_401(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_me_with_bad_token_returns_401(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


def test_register_then_login_then_me(client):
    """End-to-end auth happy path — the exact flow that 500'd in production."""
    client.post("/api/auth/register", json={"username": "alice", "email": "a@t.io", "password": "pw12345"})
    login = client.post("/api/auth/login", json={"username": "alice", "password": "pw12345"})
    token = login.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "alice"
