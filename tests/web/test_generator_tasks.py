"""G1 — the three generation endpoints (release · poll · audio proxy).

Every byte toward ACE-Step is served by ``httpx.MockTransport`` injected
through the ``generator._client`` seam (the G0 convention). No test may
open a socket toward a real generation box: it is off most of the time
by design, per the VRAM protocol, and CI has no LAN box at all.

Contract under test: ``docs/acestep-wizard-plan.md`` §"G1 contract" and
``docs/ACE-STEP-API-SPEC.md`` §§2–4.
"""
from __future__ import annotations

import json
from urllib.parse import quote

import httpx
import pytest

from agent.tools import GENRE_STYLE_PROMPTS

from web.backend import acestep_client as ac
from web.backend import generator
from web.backend.ws_manager import ws_manager


# ── Helpers ──────────────────────────────────────────────────────────


def _envelope(data, *, code: int = 200, error=None) -> dict:
    return {
        "data": data, "code": code, "error": error,
        "timestamp": 1700000000000, "extra": None,
    }


#: queued 2 · running 1 · 41 s per job — the numbers the ETA math uses.
_STATS = {
    "queued": 2, "running": 1, "avg_job_seconds": 41.0, "queue_maxsize": 200,
}

_RELEASED = {"task_id": "task-1", "status": "queued", "queue_position": 4}

#: ACE's own result root (``generator.DEFAULT_AUDIO_ROOTS``). The shared
#: validator root-checks a take's DECODED path on EVERY route — the proxy
#: included — so the fixture take has to name a file that could really be
#: one: an absolute POSIX path under this root, encoded ``quote(p, safe="")``
#: exactly as ACE hands it out.
_ACE_ROOT = generator.DEFAULT_AUDIO_ROOTS[0]
_TAKE_FILE = f"{_ACE_ROOT}/6f1c2b7e-9d4a-4c11-b0a3-2e5f8d7c1a90_0.wav"

_AUDIO_PATH = f"/v1/audio?path={quote(_TAKE_FILE, safe='')}"
_AUDIO_BYTES = b"RIFF....WAVEmock-pcm-bytes"

_TAKE = {
    "file": _AUDIO_PATH,
    "status": 1,
    "prompt": "dark melodic techno, hypnotic, driving",
    "lyrics": "[Verse]\nneon rain",
    "metas": {
        "bpm": 140, "duration": 183.4, "genres": "techno",
        "keyscale": "A Minor", "timesignature": "4",
    },
    "seed_value": "12345",
}

_BODY = {"prompt": "dark melodic techno, hypnotic", "genre_folder": "techno"}


def _box(*, release=_RELEASED, results=(), stats=_STATS, audio=None):
    """A responder for a healthy box; each surface overridable per test."""

    def responder(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json=_envelope({"status": "ok"}))
        if path == "/v1/stats":
            if stats is None:
                return httpx.Response(
                    500, json=_envelope(None, code=500, error="no stats")
                )
            return httpx.Response(200, json=_envelope(stats))
        if path == "/release_task":
            return httpx.Response(200, json=_envelope(release))
        if path == "/query_result":
            return httpx.Response(200, json=_envelope(list(results)))
        if path == "/v1/audio":
            if audio is not None:
                return audio(request)
            return httpx.Response(
                200, content=_AUDIO_BYTES, headers={"content-type": "audio/wav"}
            )
        return httpx.Response(404, json=_envelope(None, code=404, error="nope"))

    return responder


def _install_ace(monkeypatch, responder):
    """Point ``generator._client`` at a MockTransport; return the call log."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responder(request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        generator, "_client", lambda: ac.AceStepClient(transport=transport)
    )
    return calls


def _released_payload(calls) -> dict:
    """The JSON body of the single ``/release_task`` call."""
    releases = [c for c in calls if c.url.path == "/release_task"]
    assert len(releases) == 1, [c.url.path for c in calls]
    return json.loads(releases[0].content)


def _entry(task_id="task-1", status=1, result=None):
    """One ``query_result`` batch entry (``result`` is a JSON *string*)."""
    if result is None:
        result = json.dumps([_TAKE])
    return {"task_id": task_id, "status": status, "result": result}


class _FakeWS:
    """Stand-in for the WebSocket object the live handler registers."""


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_live_registry():
    """Isolate the process-wide WS registry between tests."""
    saved = dict(ws_manager._connections)
    ws_manager._connections.clear()
    yield ws_manager._connections
    ws_manager._connections.clear()
    ws_manager._connections.update(saved)


@pytest.fixture(autouse=True)
def no_catalog_reads(monkeypatch):
    """No test touches the developer's real tracks.json by accident.

    ``_resolve_genre`` only consults the catalog for genres the BPM
    window table does not know, so most tests never reach this — but the
    ones that do must be deterministic in a worktree (no ``tracks/``) and
    in CI alike.
    """
    monkeypatch.setattr(generator, "_catalog_genres", lambda: set())


@pytest.fixture
def ace_off(monkeypatch):
    monkeypatch.delenv(ac.ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ac.ENV_API_KEY, raising=False)


@pytest.fixture
def ace_on(monkeypatch):
    monkeypatch.setenv(ac.ENV_BASE_URL, "http://ace.test:8001")
    monkeypatch.delenv(ac.ENV_API_KEY, raising=False)


@pytest.fixture
def bare_token(client):
    """A valid JWT WITHOUT setting the client's Authorization header.

    ``auth_client`` mutates the shared TestClient's headers, which would
    mask the ``?token=`` path the ``<audio>`` element depends on.
    """
    client.post(
        "/api/auth/register",
        json={"username": "gen", "email": "gen@test.io", "password": "pw12345"},
    )
    resp = client.post(
        "/api/auth/login", json={"username": "gen", "password": "pw12345"}
    )
    return resp.json()["access_token"]


# ══ POST /api/generator/tasks ════════════════════════════════════════

# ── Auth ─────────────────────────────────────────────────────────────


def test_release_requires_auth(client, ace_on):
    assert client.post("/api/generator/tasks", json=_BODY).status_code == 401


def test_release_rejects_a_bad_token(client, ace_on):
    r = client.post(
        "/api/generator/tasks",
        json=_BODY,
        headers={"Authorization": "Bearer garbage"},
    )
    assert r.status_code == 401


# ── Refusals ─────────────────────────────────────────────────────────


def test_release_503_when_generator_disabled(auth_client, ace_off, monkeypatch):
    """No ACESTEP_BASE_URL ⇒ the feature is off, and no HTTP is attempted."""
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 503
    assert ac.ENV_BASE_URL in r.json()["detail"]
    assert calls == []


def test_release_503_when_the_box_is_unreachable(auth_client, ace_on, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_ace(monkeypatch, refuse)

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 503


def test_release_409_while_a_live_session_is_on_air(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    """The G0 guard doing its job: no GPU work during a broadcast."""
    calls = _install_ace(monkeypatch, _box())
    clean_live_registry[("sess-live", generator.LIVE_WS_CHANNEL)] = _FakeWS()

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail == generator.VRAM_CONFLICT_MESSAGE
    assert "VRAM" in detail and "on air" in detail
    # Refused before the GPU was ever asked for anything.
    assert calls == []


def test_release_resumes_once_the_live_session_ends(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    _install_ace(monkeypatch, _box())
    clean_live_registry[("sess-live", generator.LIVE_WS_CHANNEL)] = _FakeWS()
    assert auth_client.post("/api/generator/tasks", json=_BODY).status_code == 409

    clean_live_registry.pop(("sess-live", generator.LIVE_WS_CHANNEL))
    assert auth_client.post("/api/generator/tasks", json=_BODY).status_code == 200


def test_release_ignores_a_planning_websocket(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    """Wizard chat is not a broadcast — it holds no VRAM."""
    _install_ace(monkeypatch, _box())
    clean_live_registry[("sess-1", "planning")] = _FakeWS()

    assert auth_client.post("/api/generator/tasks", json=_BODY).status_code == 200


def test_release_429_when_the_ace_queue_is_full(auth_client, ace_on, monkeypatch):
    def queue_full(request):
        if request.url.path == "/release_task":
            return httpx.Response(
                429, json=_envelope(None, code=429, error="queue full")
            )
        return _box()(request)

    _install_ace(monkeypatch, queue_full)

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 429  # backpressure, not a crash


def test_release_502_when_ace_errors(auth_client, ace_on, monkeypatch):
    def boom(request):
        if request.url.path == "/release_task":
            return httpx.Response(500, json=_envelope(None, code=500, error="boom"))
        return _box()(request)

    _install_ace(monkeypatch, boom)

    assert auth_client.post("/api/generator/tasks", json=_BODY).status_code == 502


# ── Validation (422) ─────────────────────────────────────────────────


def test_release_422_on_an_unknown_genre(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post(
        "/api/generator/tasks", json={**_BODY, "genre_folder": "vaporgabber"}
    )

    assert r.status_code == 422
    assert "vaporgabber" in r.json()["detail"]
    assert calls == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("audio_duration", 60),       # under the 120 s catalog floor
        ("audio_duration", 900),      # over ACE-Step's 600 s ceiling
        ("batch_size", 0),
        ("batch_size", 9),            # max 8 takes per request
        ("bpm", 10),
        ("bpm", 400),
        ("prompt", ""),
        ("prompt", "   "),            # blank is not a prompt
        ("genre_folder", "   "),
        ("vocal_language", ""),
    ],
)
def test_release_422_on_out_of_range_fields(
    auth_client, ace_on, monkeypatch, field, value
):
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post("/api/generator/tasks", json={**_BODY, field: value})

    assert r.status_code == 422, (field, value)
    assert calls == []


def test_release_422_when_prompt_or_genre_missing(auth_client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box())
    assert auth_client.post(
        "/api/generator/tasks", json={"genre_folder": "techno"}
    ).status_code == 422
    assert auth_client.post(
        "/api/generator/tasks", json={"prompt": "x"}
    ).status_code == 422


def test_release_422_on_an_unknown_top_level_field(auth_client, ace_on, monkeypatch):
    """A typo must be loud — a silently ignored knob shows up hours later."""
    _install_ace(monkeypatch, _box())

    r = auth_client.post(
        "/api/generator/tasks", json={**_BODY, "inference_steps": 8}
    )

    assert r.status_code == 422


@pytest.mark.parametrize(
    "key", ["audio_format", "thinking", "bpm", "duration", "prompt", "caption"]
)
def test_release_422_when_experimental_shadows_a_server_field(
    auth_client, ace_on, monkeypatch, key
):
    """``experimental`` is a passthrough, not a way past the catalog rules."""
    _install_ace(monkeypatch, _box())

    r = auth_client.post(
        "/api/generator/tasks",
        json={**_BODY, "experimental": {key: "mp3"}},
    )

    assert r.status_code == 422


# ── Defaults the server fills ────────────────────────────────────────


def test_release_fills_the_contract_defaults(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 200
    payload = _released_payload(calls)
    assert payload["audio_format"] == "wav"      # the catalog contract
    assert payload["thinking"] is True           # spec §3.1 quality advice
    assert payload["audio_duration"] == 180
    assert payload["batch_size"] == 2
    assert payload["vocal_language"] == "en"
    assert payload["lyrics"] == ""               # empty = instrumental
    # v3.10.1 — the genre now reaches ACE as SOUND: its style descriptor
    # frames the request and the user's words specialise it. Asserting
    # both ends keeps the ORDER guarded, which is the part that matters.
    assert payload["prompt"].startswith(GENRE_STYLE_PROMPTS["techno"])
    assert payload["prompt"].endswith("dark melodic techno, hypnotic")
    assert "key_scale" not in payload            # omitted, LM completes it


@pytest.mark.parametrize(
    "genre, expected_bpm",
    [
        ("techno", 140),           # (120, 160)
        ("cyberpunk", 140),        # (120, 160)
        ("deep house", 125),       # (115, 135)
        ("lofi - ambient", 85),    # (60, 110)
        ("healing", 75),           # (50, 100)
    ],
)
def test_release_defaults_bpm_to_the_exact_window_centre(
    auth_client, ace_on, monkeypatch, genre, expected_bpm
):
    """Spec §5.5 — pinned server-side so ``metas.bpm`` comes back in-window."""
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post(
        "/api/generator/tasks", json={**_BODY, "genre_folder": genre}
    )

    assert r.status_code == 200
    assert _released_payload(calls)["bpm"] == expected_bpm

    lo, hi = generator._genre_bpm_windows()[genre]
    assert expected_bpm == (lo + hi) / 2


def test_release_bpm_default_is_case_insensitive(auth_client, ace_on, monkeypatch):
    """The catalog holds ``Healing`` with a capital H; the table is lower."""
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.post(
        "/api/generator/tasks", json={**_BODY, "genre_folder": "Healing"}
    )

    assert r.status_code == 200
    assert _released_payload(calls)["bpm"] == 75


def test_release_explicit_bpm_wins_over_the_default(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box())

    auth_client.post("/api/generator/tasks", json={**_BODY, "bpm": 133})

    assert _released_payload(calls)["bpm"] == 133


def test_release_accepts_a_catalog_genre_with_no_bpm_window(
    auth_client, ace_on, monkeypatch
):
    """``aural``/``synthware`` exist in the catalog but not in the table.

    The window table drives the DEFAULT, never the allow-list — a genre
    it does not know still generates, just without a server-pinned bpm.
    """
    calls = _install_ace(monkeypatch, _box())
    # NOT aural/synthware — both gained a window on 2026-09-07. This
    # test is about the mechanism, so it needs a genre that has none.
    monkeypatch.setattr(generator, "_catalog_genres", lambda: {"free jazz"})

    r = auth_client.post(
        "/api/generator/tasks", json={**_BODY, "genre_folder": "free jazz"}
    )

    assert r.status_code == 200
    assert "bpm" not in _released_payload(calls)


def test_release_forwards_the_experimental_panel(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box())

    auth_client.post(
        "/api/generator/tasks",
        json={
            **_BODY,
            "lyrics": "[Verse]\nneon rain",
            "key_scale": "A Minor",
            "batch_size": 4,
            "audio_duration": 240,
            "vocal_language": "es",
            "experimental": {"inference_steps": 8, "seed": 42, "time_signature": "4"},
        },
    )

    payload = _released_payload(calls)
    assert payload["inference_steps"] == 8
    assert payload["seed"] == 42
    assert payload["time_signature"] == "4"
    assert payload["lyrics"] == "[Verse]\nneon rain"
    assert payload["key_scale"] == "A Minor"
    assert payload["batch_size"] == 4
    assert payload["audio_duration"] == 240
    assert payload["vocal_language"] == "es"
    assert payload["audio_format"] == "wav"  # still server-owned


def test_release_forwards_the_api_key(auth_client, ace_on, monkeypatch):
    monkeypatch.setenv(ac.ENV_API_KEY, "sk-ace-release")
    calls = _install_ace(monkeypatch, _box())

    auth_client.post("/api/generator/tasks", json=_BODY)

    assert all(c.headers["authorization"] == "Bearer sk-ace-release" for c in calls)


# ── Response shape + ETA math ────────────────────────────────────────


def test_release_response_shape_is_exactly_the_contract(
    auth_client, ace_on, monkeypatch
):
    _install_ace(monkeypatch, _box())

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert set(body) == {"task_id", "queue_position", "eta_seconds"}
    assert body["task_id"] == "task-1"
    assert body["queue_position"] == 4


def test_release_eta_is_avg_job_seconds_times_queue_plus_running(
    auth_client, ace_on, monkeypatch
):
    """41 s/job × (position 4 + 1 running) = 205 s."""
    _install_ace(monkeypatch, _box())

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert body["eta_seconds"] == 205


def test_release_eta_is_null_when_stats_are_unavailable(
    auth_client, ace_on, monkeypatch
):
    _install_ace(monkeypatch, _box(stats=None))

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert body["eta_seconds"] is None
    assert body["task_id"] == "task-1"  # the release itself still succeeded


def test_release_eta_is_null_without_avg_job_seconds(
    auth_client, ace_on, monkeypatch
):
    _install_ace(monkeypatch, _box(stats={"queued": 2, "running": 1}))

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert body["eta_seconds"] is None


def test_release_tolerates_a_missing_queue_position(auth_client, ace_on, monkeypatch):
    """No position reported ⇒ assume the head of the queue, still an ETA."""
    _install_ace(
        monkeypatch, _box(release={"task_id": "task-9", "status": "queued"})
    )

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert body["queue_position"] is None
    assert body["eta_seconds"] == 41  # 41 × max(0 + 1 running, 1)


def test_eta_floors_at_one_job(auth_client, ace_on, monkeypatch):
    """An idle box with a free slot still needs one job's worth of time."""
    _install_ace(
        monkeypatch,
        _box(
            release={"task_id": "t", "status": "queued", "queue_position": 0},
            stats={"queued": 0, "running": 0, "avg_job_seconds": 41.0},
        ),
    )

    body = auth_client.post("/api/generator/tasks", json=_BODY).json()

    assert body["eta_seconds"] == 41


def test_eta_helper_rejects_nonsense_stats():
    assert generator._eta_seconds(None, 1) is None
    assert generator._eta_seconds({}, 1) is None
    assert generator._eta_seconds({"avg_job_seconds": "soon"}, 1) is None
    assert generator._eta_seconds({"avg_job_seconds": 0}, 1) is None
    assert generator._eta_seconds({"avg_job_seconds": 30.0, "running": None}, 2) == 60


# ══ GET /api/generator/tasks/{task_id} ═══════════════════════════════


def test_poll_requires_auth(client, ace_on):
    assert client.get("/api/generator/tasks/task-1").status_code == 401


def test_poll_503_when_generator_disabled(auth_client, ace_off, monkeypatch):
    calls = _install_ace(monkeypatch, _box())
    assert auth_client.get("/api/generator/tasks/task-1").status_code == 503
    assert calls == []


def test_poll_maps_running_to_pending(auth_client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box(results=[_entry(status=0, result="")]))

    body = auth_client.get("/api/generator/tasks/task-1").json()

    assert body["status"] == "pending"
    assert body["takes"] == []
    assert body["degraded"] is False
    assert body["eta_seconds"] == 41  # one job's worth while it runs


def test_poll_maps_status_one_to_done_with_takes(auth_client, ace_on, monkeypatch):
    two_takes = json.dumps([_TAKE, {**_TAKE, "seed_value": "67890", "lyrics": ""}])
    _install_ace(monkeypatch, _box(results=[_entry(status=1, result=two_takes)]))

    r = auth_client.get("/api/generator/tasks/task-1")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["task_id"] == "task-1"
    assert body["eta_seconds"] is None  # nothing left to wait for
    assert [t["index"] for t in body["takes"]] == [0, 1]

    first = body["takes"][0]
    assert first["file"] == _TAKE["file"]
    assert first["prompt"] == _TAKE["prompt"]
    assert first["lyrics"] == "[Verse]\nneon rain"
    assert first["seed_value"] == "12345"
    # metas passed through verbatim — G2 ingests these, never re-detects.
    assert first["metas"] == _TAKE["metas"]
    assert body["takes"][1]["seed_value"] == "67890"


def test_poll_maps_status_two_to_failed(auth_client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box(results=[_entry(status=2, result="")]))

    body = auth_client.get("/api/generator/tasks/task-1").json()

    assert body["status"] == "failed"
    assert body["takes"] == []
    assert body["eta_seconds"] is None


def test_poll_response_shape(auth_client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box(results=[_entry()]))

    body = auth_client.get("/api/generator/tasks/task-1").json()

    assert set(body) == {
        "task_id", "status", "takes", "eta_seconds", "degraded",
        "result_parse_error",
    }
    assert set(body["takes"][0]) == {
        "index", "file", "prompt", "lyrics", "metas", "seed_value",
    }


def test_poll_is_allowed_during_a_live_session(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    """Only task RELEASE touches the GPU — polling must keep working."""
    _install_ace(monkeypatch, _box(results=[_entry()]))
    clean_live_registry[("sess-live", generator.LIVE_WS_CHANNEL)] = _FakeWS()

    r = auth_client.get("/api/generator/tasks/task-1")

    assert r.status_code == 200
    assert r.json()["status"] == "done"


# ── Degraded polls: a blip must never 5xx ────────────────────────────


def test_poll_survives_a_transport_blip(auth_client, ace_on, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_ace(monkeypatch, refuse)

    r = auth_client.get("/api/generator/tasks/task-1")

    assert r.status_code == 200
    assert r.json() == {
        "task_id": "task-1", "status": "pending", "takes": [],
        "eta_seconds": None, "degraded": True, "result_parse_error": None,
    }


def test_poll_survives_an_ace_500(auth_client, ace_on, monkeypatch):
    def boom(request):
        if request.url.path == "/query_result":
            return httpx.Response(500, json=_envelope(None, code=500, error="boom"))
        return _box()(request)

    _install_ace(monkeypatch, boom)

    body = auth_client.get("/api/generator/tasks/task-1").json()

    assert body["status"] == "pending" and body["degraded"] is True


def test_poll_survives_a_malformed_batch(auth_client, ace_on, monkeypatch):
    def garbage(request):
        if request.url.path == "/query_result":
            return httpx.Response(200, json=_envelope("nonsense"))
        return _box()(request)

    _install_ace(monkeypatch, garbage)

    assert auth_client.get("/api/generator/tasks/task-1").json()["degraded"] is True


def test_poll_of_an_unknown_id_degrades_rather_than_404s(
    auth_client, ace_on, monkeypatch
):
    """A not-yet-registered id must not tear the wizard's task card down."""
    _install_ace(monkeypatch, _box(results=[_entry(task_id="someone-else")]))

    body = auth_client.get("/api/generator/tasks/task-1").json()

    assert body["status"] == "pending" and body["degraded"] is True


def test_poll_502_on_a_bad_api_key(auth_client, ace_on, monkeypatch):
    """A misconfiguration is not a blip — retrying will never fix it."""
    def unauthorized(request):
        if request.url.path == "/query_result":
            return httpx.Response(401, json=_envelope(None, code=401, error="nope"))
        return _box()(request)

    _install_ace(monkeypatch, unauthorized)

    assert auth_client.get("/api/generator/tasks/task-1").status_code == 502


def test_poll_carries_a_result_parse_error_through(auth_client, ace_on, monkeypatch):
    """One undecodable payload is reported, not fatal (client contract)."""
    _install_ace(
        monkeypatch, _box(results=[_entry(status=1, result="{not json at all")])
    )

    r = auth_client.get("/api/generator/tasks/task-1")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["takes"] == []
    assert "not valid JSON" in body["result_parse_error"]
    assert body["degraded"] is False  # the box answered fine; the payload did not


def test_poll_sends_the_id_as_a_batch_of_one(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box(results=[_entry()]))

    auth_client.get("/api/generator/tasks/task-1")

    polls = [c for c in calls if c.url.path == "/query_result"]
    assert json.loads(polls[0].content) == {"task_id_list": ["task-1"]}


# ══ GET /api/generator/audio ═════════════════════════════════════════


def test_audio_requires_auth(client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box())
    r = client.get("/api/generator/audio", params={"path": _AUDIO_PATH})
    assert r.status_code == 401


def test_audio_rejects_a_bad_query_token(client, ace_on, monkeypatch):
    _install_ace(monkeypatch, _box())
    r = client.get(
        "/api/generator/audio", params={"path": _AUDIO_PATH, "token": "garbage"}
    )
    assert r.status_code == 401


def test_audio_streams_the_take(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    assert r.status_code == 200
    assert r.content == _AUDIO_BYTES
    assert r.headers["content-type"] == "audio/wav"

    proxied = [c for c in calls if c.url.path == "/v1/audio"]
    assert len(proxied) == 1
    # Forwarded byte for byte — Apollo screens the inner path (it is under
    # ACE's result root) but never rewrites it; ACE's own validator is
    # still the far-side authority on what it will serve.
    assert proxied[0].url.params.get("path") == _TAKE_FILE
    assert proxied[0].url.host == "ace.test"


def test_audio_accepts_a_query_token_for_the_audio_element(
    client, ace_on, monkeypatch, bare_token
):
    """``<audio src>`` cannot set an Authorization header."""
    _install_ace(monkeypatch, _box())

    r = client.get(
        "/api/generator/audio", params={"path": _AUDIO_PATH, "token": bare_token}
    )

    assert r.status_code == 200
    assert r.content == _AUDIO_BYTES


def test_audio_forwards_range_and_mirrors_206(auth_client, ace_on, monkeypatch):
    def ranged(request):
        assert request.headers.get("range") == "bytes=0-3"
        return httpx.Response(
            206,
            content=b"RIFF",
            headers={
                "content-type": "audio/wav",
                "content-range": "bytes 0-3/26",
                "accept-ranges": "bytes",
            },
        )

    _install_ace(monkeypatch, _box(audio=ranged))

    r = auth_client.get(
        "/api/generator/audio",
        params={"path": _AUDIO_PATH},
        headers={"Range": "bytes=0-3"},
    )

    assert r.status_code == 206
    assert r.content == b"RIFF"
    assert r.headers["content-range"] == "bytes 0-3/26"
    assert r.headers["accept-ranges"] == "bytes"


def test_audio_forwards_the_api_key(auth_client, ace_on, monkeypatch):
    monkeypatch.setenv(ac.ENV_API_KEY, "sk-ace-audio")
    calls = _install_ace(monkeypatch, _box())

    auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    proxied = [c for c in calls if c.url.path == "/v1/audio"]
    assert proxied[0].headers["authorization"] == "Bearer sk-ace-audio"


def test_audio_is_allowed_during_a_live_session(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    _install_ace(monkeypatch, _box())
    clean_live_registry[("sess-live", generator.LIVE_WS_CHANNEL)] = _FakeWS()

    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    assert r.status_code == 200


@pytest.mark.parametrize(
    "bad_path",
    [
        "http://evil.test/v1/audio?path=x",     # host-carrying
        "https://ace.test:8001/v1/audio",       # even the real host
        "//evil.test/v1/audio",                 # protocol-relative
        "/tmp/out/take0.wav",                   # absolute file path, out of root
        "/etc/passwd",
        # The endpoint shape wrapping that same out-of-root path: the
        # proxy no longer forwards what publish would refuse.
        "/v1/audio?path=%2Ftmp%2Fout%2Ftake0.wav",
        "/v1/audio?path=%2Fetc%2Fpasswd",
        "C:\\Windows\\win.ini",                 # windows drive
        "out\\take0.wav",                       # backslashes
        "/v1/../secret",                        # traversal
        "   ",                                  # blank
    ],
)
def test_audio_400_on_a_non_relative_path(
    auth_client, ace_on, monkeypatch, bad_path
):
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.get("/api/generator/audio", params={"path": bad_path})

    assert r.status_code == 400, bad_path
    assert calls == []  # nothing was proxied anywhere


def test_audio_streams_a_bare_absolute_path_under_the_root(
    auth_client, ace_on, monkeypatch
):
    """The contrast to the 400 list above: in-root, so it streams.

    A bare absolute path is what the wizard persists, and the validator
    resolves it the same way for the proxy as for publish — re-encoded
    into ``/v1/audio?path=<quote(p, safe="")>``, slashes as ``%2F``. Only
    the ROOT decides; the shape does not (the proxy-side twin of
    ``test_publish_shape_is_the_absolute_path_under_the_root``).
    """
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.get("/api/generator/audio", params={"path": _TAKE_FILE})

    assert r.status_code == 200
    assert r.content == _AUDIO_BYTES
    proxied = [c for c in calls if c.url.path == "/v1/audio"]
    assert len(proxied) == 1
    assert proxied[0].url.params.get("path") == _TAKE_FILE
    # Re-encoded, not passed through raw: slashes travel as %2F.
    assert b"%2F" in proxied[0].url.query


def test_audio_422_without_a_path(auth_client, ace_on, monkeypatch):
    """``path`` is a required query param — FastAPI answers before us."""
    _install_ace(monkeypatch, _box())
    assert auth_client.get("/api/generator/audio").status_code == 422


def test_audio_503_when_generator_disabled(auth_client, ace_off, monkeypatch):
    calls = _install_ace(monkeypatch, _box())
    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})
    assert r.status_code == 503
    assert calls == []


def test_audio_503_when_the_box_is_unreachable(auth_client, ace_on, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_ace(monkeypatch, refuse)

    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    assert r.status_code == 503


def test_audio_404_when_the_take_is_gone(auth_client, ace_on, monkeypatch):
    def missing(request):
        return httpx.Response(404, json=_envelope(None, code=404, error="no file"))

    _install_ace(monkeypatch, _box(audio=missing))

    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    assert r.status_code == 404


def test_audio_502_when_ace_errors(auth_client, ace_on, monkeypatch):
    def boom(request):
        return httpx.Response(500, text="kaboom")

    _install_ace(monkeypatch, _box(audio=boom))

    r = auth_client.get("/api/generator/audio", params={"path": _AUDIO_PATH})

    assert r.status_code == 502


def test_audio_accepts_a_bare_relative_path(auth_client, ace_on, monkeypatch):
    """``audio_url`` wraps a bare path into ``/v1/audio?path=`` itself."""
    calls = _install_ace(monkeypatch, _box())

    r = auth_client.get("/api/generator/audio", params={"path": "out/take 0.wav"})

    assert r.status_code == 200
    proxied = [c for c in calls if c.url.path == "/v1/audio"]
    assert proxied[0].url.params.get("path") == "out/take 0.wav"


# ══ AceStepClient.stream_audio (the proxy's transport seam) ══════════


async def test_stream_audio_yields_an_open_response(ace_on):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"abc"))

    async with ac.AceStepClient(transport=transport).stream_audio(
        _AUDIO_PATH
    ) as resp:
        chunks = [c async for c in resp.aiter_bytes()]

    assert b"".join(chunks) == b"abc"


async def test_stream_audio_merges_request_headers(ace_on, monkeypatch):
    monkeypatch.setenv(ac.ENV_API_KEY, "sk-stream")
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"x")

    client = ac.AceStepClient(transport=httpx.MockTransport(handler))
    async with client.stream_audio(_AUDIO_PATH, headers={"range": "bytes=0-1"}):
        pass

    assert seen[0].headers["range"] == "bytes=0-1"
    assert seen[0].headers["authorization"] == "Bearer sk-stream"


async def test_stream_audio_raises_typed_errors(ace_on):
    transport = httpx.MockTransport(lambda r: httpx.Response(404, text="gone"))
    client = ac.AceStepClient(transport=transport)

    with pytest.raises(ac.AceStepError) as excinfo:
        async with client.stream_audio(_AUDIO_PATH):
            pass

    assert excinfo.value.status_code == 404


async def test_stream_audio_transport_failure_is_unavailable(ace_on):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    client = ac.AceStepClient(transport=httpx.MockTransport(refuse))

    with pytest.raises(ac.AceStepUnavailable):
        async with client.stream_audio(_AUDIO_PATH):
            pass


async def test_stream_audio_is_disabled_without_a_base_url(ace_off):
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda r: calls.append(r) or httpx.Response(200, content=b"")
    )

    with pytest.raises(ac.AceStepDisabled):
        async with ac.AceStepClient(transport=transport).stream_audio(_AUDIO_PATH):
            pass

    assert calls == []


# ══ The BPM table this module binds to ═══════════════════════════════


def test_the_bound_bpm_table_agrees_with_main_where_both_define_a_genre():
    """``agent.tools`` is a partial copy of ``main``'s canonical table.

    The defaulting logic binds to the ``agent.tools`` copy (see
    ``generator._genre_bpm_windows``); this pins the invariant that
    matters — where both files name a genre, the window is identical, so
    a default picked here is the canonical centre. Missing keys are a
    known, documented gap, not a silent disagreement.
    """
    import main  # noqa: PLC0415 — heavy import (librosa/moviepy); test-only

    bound = generator._genre_bpm_windows()
    shared = sorted(set(bound) & set(main.BPM_GENRE_RANGES))

    assert shared, "the two tables share no genre at all — one moved"
    for genre in shared:
        assert tuple(bound[genre]) == tuple(main.BPM_GENRE_RANGES[genre]), genre


def test_every_bound_window_yields_an_integer_centre():
    """``_default_bpm_for`` must round to a usable int, not just average.

    The exact-equality form of this assertion (``centre == (lo + hi) / 2``)
    only ever held by accident: every genre bound at the time had an even
    ``lo + hi``, so the unrounded midpoint was already a whole number. Sync
    with main.py's ``BPM_GENRE_RANGES`` (2026-08-29) added "soul jazz"
    (75, 140), whose midpoint is 107.5 — no int can equal that, so the old
    assertion was mathematically unsatisfiable for this window regardless
    of what ``_default_bpm_for`` returns. Assert what the test's name
    actually promises: an int, within rounding distance of the true
    midpoint.
    """
    for genre, (lo, hi) in generator._genre_bpm_windows().items():
        centre = generator._default_bpm_for(genre)
        assert isinstance(centre, int), genre
        assert abs(centre - (lo + hi) / 2) <= 0.5, genre
        assert generator.MIN_BPM <= centre <= generator.MAX_BPM, genre
