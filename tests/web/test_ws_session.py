"""Integration tests for the /ws/sessions/{id} WebSocket endpoint."""
from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect


def _receive_until(ws, predicate, limit=20):
    """Receive up to `limit` messages until `predicate(event)` is truthy."""
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if predicate(event):
            return events
    raise AssertionError(f"Predicate never satisfied in {limit} messages: {events}")


def test_ws_rejects_bad_token(auth_client, mock_pipeline):
    sid = auth_client.post("/api/sessions").json()["id"]
    with pytest.raises(WebSocketDisconnect):
        with auth_client.websocket_connect(f"/ws/sessions/{sid}?token=garbage"):
            pass


def test_ws_rejects_other_users_session(auth_client, second_client, mock_pipeline):
    sid = auth_client.post("/api/sessions").json()["id"]

    second_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@t.io", "password": "pw12345"},
    )
    other_token = second_client.post(
        "/api/auth/login", json={"username": "u2", "password": "pw12345"}
    ).json()["access_token"]

    with pytest.raises(WebSocketDisconnect):
        with second_client.websocket_connect(f"/ws/sessions/{sid}?token={other_token}"):
            pass


def test_ws_initial_state_event(auth_client, auth_token, mock_pipeline):
    sid = auth_client.post("/api/sessions").json()["id"]
    with auth_client.websocket_connect(f"/ws/sessions/{sid}?token={auth_token}") as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        assert first["data"]["id"] == sid


def test_ws_genre_intent_runs_planner(auth_client, auth_token, mock_pipeline):
    """genre_intent should confirm genre AND auto-run the Planner, ending on phase_complete: planning."""
    sid = auth_client.post("/api/sessions").json()["id"]
    with auth_client.websocket_connect(f"/ws/sessions/{sid}?token={auth_token}") as ws:
        ws.receive_json()  # initial state
        ws.send_json({"type": "genre_intent", "content": "60 minutes of dark techno"})

        # Drain until the final phase_complete for 'planning' arrives
        events = _receive_until(
            ws,
            lambda e: e.get("type") == "phase_complete" and e.get("phase") == "planning",
        )

    phases_started = [e.get("phase") for e in events if e["type"] == "phase_start"]
    phases_done = [e.get("phase") for e in events if e["type"] == "phase_complete"]
    assert "genre" in phases_done
    assert "planning" in phases_started
    assert "planning" in phases_done


def test_ws_get_state_roundtrip(auth_client, auth_token, mock_pipeline):
    sid = auth_client.post("/api/sessions").json()["id"]
    with auth_client.websocket_connect(f"/ws/sessions/{sid}?token={auth_token}") as ws:
        ws.receive_json()  # initial
        ws.send_json({"type": "get_state"})
        msg = ws.receive_json()
        assert msg["type"] == "state"
        assert msg["data"]["id"] == sid
