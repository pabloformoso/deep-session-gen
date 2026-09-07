"""G6 — the Generations Library: the recording hooks + the three routes.

The wizard's page state used to be the only record that a generation
ever happened. These tests pin the store that outlives it:

1. **The hooks**, driven through the EXISTING endpoints (release, poll,
   publish, edit) — never by calling ``db`` directly, because the whole
   contract is "the page stays dumb and the history happens anyway".
   Idempotency (two done-polls persist once), strict user scoping (a
   foreign poll may not rewrite the owner's row) and the rule that a
   store failure can never break an endpoint — the 3-second poll least
   of all.
2. **The routes**: the feed with its pagination bounds, the
   discard/restore PATCH with its validation, and ``refresh``'s
   ``stale``-vs-``degraded`` distinction (ACE answering that it has no
   such task is terminal; ACE not answering is not — never conflate).

Every byte toward ACE-Step is served by ``httpx.MockTransport`` through
the ``generator._client`` seam (the G0 convention): no socket is opened
toward a real generation box, and no test touches the developer's real
catalog. The DB is the house ``tmp_db`` fixture — one file per test.

Contract: ``docs/acestep-wizard-plan.md`` §"G6 contract / Backend".
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

from agent.tools import GENRE_STYLE_PROMPTS

from web.backend import acestep_client as ac
from web.backend import db, generator
from web.backend.ws_manager import ws_manager


# ── ACE-Step fixtures on the wire ────────────────────────────────────

#: The real shape (confirmed with the ACE session): an absolute POSIX
#: path under their tmp api_audio root. The library stores the DECODED
#: path, so every take here has to name one that could really be a take.
ACE_ROOT = generator.DEFAULT_AUDIO_ROOTS[0]
TAKE_0_FILE = f"{ACE_ROOT}/6f1c2b7e-9d4a-4c11-b0a3-2e5f8d7c1a90_0.wav"
TAKE_1_FILE = f"{ACE_ROOT}/6f1c2b7e-9d4a-4c11-b0a3-2e5f8d7c1a90_1.wav"


def _endpoint(path: str) -> str:
    """ACE's wire form for a result file — ``quote(p, safe="")``."""
    return f"/v1/audio?path={quote(path, safe='')}"


_STATS = {"queued": 2, "running": 1, "avg_job_seconds": 41.0, "queue_maxsize": 200}
_RELEASED = {"task_id": "task-1", "status": "queued", "queue_position": 4}

_TAKE_0 = {
    "file": _endpoint(TAKE_0_FILE),
    "status": 1,
    "prompt": "dark melodic techno, hypnotic, driving",
    "lyrics": "[Verse]\nneon rain",
    "metas": {
        "bpm": 138, "duration": 181.4, "genres": "techno",
        "keyscale": "A Minor", "timesignature": "4",
    },
    "seed_value": "12345",
}
_TAKE_1 = {
    **_TAKE_0,
    "file": _endpoint(TAKE_1_FILE),
    "lyrics": "",
    "seed_value": "67890",
}

_BODY = {"prompt": "dark melodic techno, hypnotic", "genre_folder": "techno"}

#: Over the 120 s session-eligibility floor, so the real ingest accepts it.
LONG_SEC = 121


def _envelope(data, *, code: int = 200, error=None) -> dict:
    return {
        "data": data, "code": code, "error": error,
        "timestamp": 1700000000000, "extra": None,
    }


def _entry(task_id="task-1", status=1, takes=(_TAKE_0, _TAKE_1)):
    """One ``query_result`` batch entry (``result`` is a JSON *string*)."""
    return {
        "task_id": task_id,
        "status": status,
        "result": json.dumps(list(takes)) if takes else "",
    }


def _box(*, release=_RELEASED, results=(), stats=_STATS, audio=b"", audio_status=200):
    """A responder for a healthy box; each surface overridable per test."""

    def responder(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json=_envelope({"status": "ok"}))
        if path == "/v1/stats":
            return httpx.Response(200, json=_envelope(stats))
        if path == "/release_task":
            return httpx.Response(200, json=_envelope(release))
        if path == "/query_result":
            return httpx.Response(200, json=_envelope(list(results)))
        if path == "/v1/audio":
            if audio_status >= 400:
                return httpx.Response(
                    audio_status,
                    json=_envelope(None, code=audio_status, error="no such file"),
                )
            return httpx.Response(
                200, content=audio, headers={"content-type": "audio/wav"}
            )
        return httpx.Response(404, json=_envelope(None, code=404, error="nope"))

    return responder


def _install_ace(monkeypatch, responder):
    """Point ``generator._client`` at a MockTransport; return the call log.

    Callable more than once per test — a second install replaces the
    first, which is how a test walks release → poll → edit with a
    different answer at each stage.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responder(request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        generator, "_client", lambda: ac.AceStepClient(transport=transport)
    )
    return calls


def _wav_bytes(path: Path, seconds: float) -> bytes:
    """A real, catalog-conformant (44.1 kHz/16-bit/stereo) silent WAV."""
    frames = int(44100 * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00" * (frames * 2 * 2))
    return path.read_bytes()


def _publish_body(**over) -> dict:
    body = {
        "file": TAKE_1_FILE,
        "metas": {"bpm": 138, "keyscale": "A Minor", "duration": 181.4},
        "prompt": "dark melodic techno, hypnotic, driving",
        "display_name": "Neon Rain",
        "genre_folder": "techno",
    }
    body.update(over)
    return body


class _FakeWS:
    """Stand-in for the WebSocket object the live handler registers."""


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_live_registry():
    """Isolate the process-wide WS registry (release/edit carry the 409)."""
    saved = dict(ws_manager._connections)
    ws_manager._connections.clear()
    yield ws_manager._connections
    ws_manager._connections.clear()
    ws_manager._connections.update(saved)


@pytest.fixture(autouse=True)
def no_catalog_reads(monkeypatch):
    """No test touches the developer's real tracks.json by accident."""
    monkeypatch.setattr(generator, "_catalog_genres", lambda: set())


@pytest.fixture
def ace_on(monkeypatch):
    monkeypatch.setenv(ac.ENV_BASE_URL, "http://ace.test:8001")
    monkeypatch.delenv(ac.ENV_API_KEY, raising=False)
    monkeypatch.delenv(generator.ENV_AUDIO_ROOT, raising=False)


@pytest.fixture
def other_client(second_client):
    """A SECOND registered user on the same DB — the scoping counterpart."""
    second_client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@test.io", "password": "pw12345"},
    )
    resp = second_client.post(
        "/api/auth/login", json={"username": "u2", "password": "pw12345"}
    )
    second_client.headers.update(
        {"Authorization": f"Bearer {resp.json()['access_token']}"}
    )
    return second_client


@pytest.fixture(scope="module")
def long_wav(tmp_path_factory) -> bytes:
    return _wav_bytes(tmp_path_factory.mktemp("ace_takes") / "take.wav", LONG_SEC)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch) -> Path:
    """A tmp ``tracks/`` tree — publish runs the REAL ingest against it."""
    monkeypatch.chdir(tmp_path)
    tracks = tmp_path / "tracks"
    (tracks / "techno").mkdir(parents=True)
    (tracks / "tracks.json").write_text(
        json.dumps({"tracks": []}, indent=2), encoding="utf-8"
    )
    return tracks


# ── Small drivers ────────────────────────────────────────────────────


def _release(client, monkeypatch, *, task_id="task-1", body=None):
    """Release one generation; returns the task id the endpoint answered."""
    _install_ace(
        monkeypatch,
        _box(release={"task_id": task_id, "status": "queued", "queue_position": 1}),
    )
    r = client.post("/api/generator/tasks", json=body or _BODY)
    assert r.status_code == 200, r.text
    return r.json()["task_id"]


def _poll(client, monkeypatch, *, task_id="task-1", status=1, takes=(_TAKE_0, _TAKE_1)):
    _install_ace(
        monkeypatch,
        _box(results=[_entry(task_id=task_id, status=status, takes=takes)]),
    )
    r = client.get(f"/api/generator/tasks/{task_id}")
    assert r.status_code == 200, r.text
    return r.json()


def _feed(client, **params):
    r = client.get("/api/generator/generations", params=params or None)
    assert r.status_code == 200, r.text
    return r.json()


def _user_id(username: str = "u1") -> int:
    """The id behind a conftest test user — for the store-level setup calls."""
    row = db.get_user_by_username(username)
    assert row is not None, username
    return row["id"]


# ══ The feed: what release + poll record ═════════════════════════════


def test_feed_requires_auth(client):
    assert client.get("/api/generator/generations").status_code == 401


def test_feed_is_empty_for_a_user_who_never_generated(auth_client):
    assert _feed(auth_client) == []


def test_release_records_a_pending_generation(auth_client, ace_on, monkeypatch):
    """The row exists the moment the GPU work is queued, not when it lands."""
    _release(auth_client, monkeypatch)

    feed = _feed(auth_client)

    assert len(feed) == 1
    gen = feed[0]
    assert gen["id"] == "task-1"
    assert gen["status"] == "pending"
    assert gen["takes"] == []
    assert gen["created_at"]


def test_the_recorded_request_is_the_actual_outgoing_payload(
    auth_client, ace_on, monkeypatch
):
    """Server-pinned defaults included — that is what was really asked for."""
    _release(
        auth_client,
        monkeypatch,
        body={**_BODY, "batch_size": 3, "experimental": {"inference_steps": 8}},
    )

    request = _feed(auth_client)[0]["request"]

    assert request["prompt"].startswith(GENRE_STYLE_PROMPTS["techno"])
    assert request["prompt"].endswith("dark melodic techno, hypnotic")
    assert request["audio_format"] == "wav"       # the catalog contract
    assert request["thinking"] is True
    assert request["bpm"] == 140                  # the window centre, pinned
    assert request["batch_size"] == 3
    assert request["inference_steps"] == 8        # the experimental passthrough
    # The one field ACE never sees, carried so the feed can label the card.
    assert request["genre_folder"] == "techno"


def test_a_done_poll_fills_in_the_takes(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    gen = _feed(auth_client)[0]

    assert gen["status"] == "done"
    assert [t["index"] for t in gen["takes"]] == [0, 1]
    first = gen["takes"][0]
    assert first["file"] == _TAKE_0["file"]
    assert first["decoded_path"] == TAKE_0_FILE   # decoded once, by the validator
    assert first["metas"] == _TAKE_0["metas"]     # verbatim — publish ingests these
    assert first["prompt"] == _TAKE_0["prompt"]
    assert first["lyrics"] == "[Verse]\nneon rain"
    assert first["seed_value"] == "12345"
    assert first["state"] == "fresh"
    assert first["published_track_id"] is None
    assert gen["takes"][1]["seed_value"] == "67890"


def test_a_stored_take_is_poll_shaped_plus_the_library_fields(
    auth_client, ace_on, monkeypatch
):
    """The library renders through the wizard's take components."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    gen = _feed(auth_client)[0]

    assert set(gen) == {"id", "user_id", "created_at", "status", "request", "takes"}
    assert set(gen["takes"][0]) == {
        "index", "file", "prompt", "lyrics", "metas", "seed_value",
        "decoded_path", "state", "published_track_id",
    }


def test_a_failed_poll_marks_the_generation_failed(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch, status=2, takes=())

    gen = _feed(auth_client)[0]

    assert gen["status"] == "failed"
    assert gen["takes"] == []


def test_a_degraded_poll_leaves_the_generation_pending(
    auth_client, ace_on, monkeypatch
):
    """A blip says nothing about the job — the row must not move."""
    _release(auth_client, monkeypatch)

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_ace(monkeypatch, refuse)
    assert auth_client.get("/api/generator/tasks/task-1").json()["degraded"] is True

    assert _feed(auth_client)[0]["status"] == "pending"


def test_the_feed_is_newest_first(auth_client, ace_on, monkeypatch):
    for task_id in ("task-1", "task-2", "task-3"):
        _release(auth_client, monkeypatch, task_id=task_id)

    assert [g["id"] for g in _feed(auth_client)] == ["task-3", "task-2", "task-1"]


def test_an_edit_records_its_own_generation_carrying_the_lineage(
    auth_client, ace_on, monkeypatch
):
    """An edit is an ordinary generation whose source is queryable."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    _install_ace(
        monkeypatch,
        _box(release={"task_id": "task-edit", "status": "queued", "queue_position": 0}),
    )
    r = auth_client.post(
        "/api/generator/edit",
        json={"file": TAKE_1_FILE, "mode": "cover", "genre_folder": "techno"},
    )
    assert r.status_code == 200, r.text

    feed = _feed(auth_client)
    assert [g["id"] for g in feed] == ["task-edit", "task-1"]
    edit = feed[0]
    assert edit["status"] == "pending"
    assert edit["request"]["task_type"] == "cover"
    assert edit["request"]["src_audio_path"] == TAKE_1_FILE
    assert edit["request"]["audio_cover_strength"] == generator.DEFAULT_COVER_STRENGTH
    assert edit["request"]["genre_folder"] == "techno"


# ══ Pagination ═══════════════════════════════════════════════════════


def test_pagination_walks_the_feed_newest_first(auth_client, ace_on, monkeypatch):
    for task_id in ("task-1", "task-2", "task-3"):
        _release(auth_client, monkeypatch, task_id=task_id)

    assert [g["id"] for g in _feed(auth_client, limit=2)] == ["task-3", "task-2"]
    assert [g["id"] for g in _feed(auth_client, limit=2, offset=2)] == ["task-1"]
    assert _feed(auth_client, limit=2, offset=9) == []


def test_the_default_page_is_twenty(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)

    assert db.DEFAULT_GENERATIONS_LIMIT == 20
    # The default is the store's constant, not a literal in the handler.
    assert len(_feed(auth_client)) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},        # a page of nothing is a mistake, not a query
        {"limit": 101},      # over the ceiling
        {"limit": -1},
        {"offset": -1},
        {"limit": "many"},
    ],
)
def test_pagination_bounds_are_enforced(auth_client, params):
    assert auth_client.get(
        "/api/generator/generations", params=params
    ).status_code == 422


def test_the_maximum_page_is_accepted(auth_client):
    assert auth_client.get(
        "/api/generator/generations", params={"limit": db.MAX_GENERATIONS_LIMIT}
    ).status_code == 200


# ══ Hook idempotency ═════════════════════════════════════════════════


def test_two_done_polls_persist_one_set_of_takes(auth_client, ace_on, monkeypatch):
    """The wizard polls on a timer; the second answer must change nothing."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    feed = _feed(auth_client)

    assert len(feed) == 1
    assert [t["index"] for t in feed[0]["takes"]] == [0, 1]


def test_a_re_poll_does_not_resurrect_a_discarded_take(
    auth_client, ace_on, monkeypatch
):
    """``state`` is the user's, not ACE's — an upsert may not walk it back."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)
    auth_client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "discarded"}
    )

    _poll(auth_client, monkeypatch)

    assert _feed(auth_client)[0]["takes"][0]["state"] == "discarded"


def test_a_re_poll_does_not_unpublish_a_published_take(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)
    db.mark_take_published(_user_id(), TAKE_1_FILE, "techno--neon-rain")

    _poll(auth_client, monkeypatch)

    take = _feed(auth_client)[0]["takes"][1]
    assert take["state"] == "published"
    assert take["published_track_id"] == "techno--neon-rain"


def test_a_re_release_of_the_same_task_id_keeps_the_first_row(
    auth_client, ace_on, monkeypatch
):
    """ACE's task id is the primary key — a collision is "already recorded"."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)
    _release(auth_client, monkeypatch)

    feed = _feed(auth_client)

    assert len(feed) == 1
    assert feed[0]["status"] == "done"      # not reset to pending
    assert len(feed[0]["takes"]) == 2


def test_polling_a_task_the_store_never_saw_is_a_normal_poll(
    auth_client, ace_on, monkeypatch
):
    """Tasks released before G6 (or from another box) still poll fine."""
    body = _poll(auth_client, monkeypatch, task_id="task-unknown")

    assert body["status"] == "done"
    assert len(body["takes"]) == 2
    assert _feed(auth_client) == []   # nothing invented for an unknown id


# ══ A store failure never breaks the endpoint it hangs off ═══════════


def test_a_store_failure_does_not_break_the_poll(auth_client, ace_on, monkeypatch):
    """The oldest rule in the module: a poll must not 5xx. SQLite included."""
    _release(auth_client, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "save_generation_takes", boom)

    body = _poll(auth_client, monkeypatch)

    assert body["status"] == "done"
    assert len(body["takes"]) == 2
    assert body["degraded"] is False   # ACE was fine; the store was not


def test_a_store_failure_does_not_break_the_release(auth_client, ace_on, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk is full")

    monkeypatch.setattr(db, "record_generation", boom)
    _install_ace(monkeypatch, _box())

    r = auth_client.post("/api/generator/tasks", json=_BODY)

    assert r.status_code == 200
    assert r.json()["task_id"] == "task-1"


def test_a_store_failure_does_not_break_a_refresh(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "set_generation_status", boom)
    _install_ace(monkeypatch, _box(results=[]))

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 200
    assert r.json()["degraded"] is False


# ══ Publish marks the take by its decoded path ═══════════════════════


def test_publish_marks_the_matching_take_published(
    auth_client, ace_on, monkeypatch, tmp_catalog, long_wav
):
    """Zero contract change: the path the page already sends is the key."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    _install_ace(monkeypatch, _box(audio=long_wav))
    r = auth_client.post("/api/generator/publish", json=_publish_body())
    assert r.status_code == 200, r.text
    track_id = r.json()["track_id"]

    takes = _feed(auth_client)[0]["takes"]
    assert takes[1]["state"] == "published"
    assert takes[1]["published_track_id"] == track_id == "techno--neon-rain"
    # The sibling take of the same batch is untouched.
    assert takes[0]["state"] == "fresh"
    assert takes[0]["published_track_id"] is None


def test_publishing_a_take_the_store_never_saw_still_succeeds(
    auth_client, ace_on, monkeypatch, tmp_catalog, long_wav
):
    """An unmatched path is a log line, not a refusal."""
    _install_ace(monkeypatch, _box(audio=long_wav))

    r = auth_client.post("/api/generator/publish", json=_publish_body())

    assert r.status_code == 200, r.text
    assert _feed(auth_client) == []


def test_publish_marks_only_the_publishers_own_take(
    auth_client, other_client, ace_on, monkeypatch, tmp_catalog, long_wav
):
    """u2 publishing the same path may not touch u1's row."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    _install_ace(monkeypatch, _box(audio=long_wav))
    r = other_client.post("/api/generator/publish", json=_publish_body())
    assert r.status_code == 200, r.text

    assert _feed(auth_client)[0]["takes"][1]["state"] == "fresh"


# ══ Strict user scoping ══════════════════════════════════════════════


def test_a_user_never_sees_another_users_generations(
    auth_client, other_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    assert len(_feed(auth_client)) == 1
    assert _feed(other_client) == []


def test_a_foreign_done_poll_does_not_rewrite_the_owners_row(
    auth_client, other_client, ace_on, monkeypatch
):
    """Task ids come from the browser — the scoping lives in the SQL."""
    _release(auth_client, monkeypatch)

    body = _poll(other_client, monkeypatch)   # u2 polls u1's task id
    assert body["status"] == "done"           # the poll itself is unaffected

    gen = _feed(auth_client)[0]
    assert gen["status"] == "pending"
    assert gen["takes"] == []
    assert _feed(other_client) == []


def test_a_user_cannot_patch_another_users_take(
    auth_client, other_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    r = other_client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "discarded"}
    )

    assert r.status_code == 404   # indistinguishable from "no such generation"
    assert _feed(auth_client)[0]["takes"][0]["state"] == "fresh"


def test_a_user_cannot_refresh_another_users_generation(
    auth_client, other_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    calls = _install_ace(monkeypatch, _box(results=[_entry()]))

    r = other_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 404
    assert calls == []            # refused before ACE was asked anything
    assert _feed(auth_client)[0]["status"] == "pending"


# ══ PATCH .../takes/{idx} ════════════════════════════════════════════


def test_patch_requires_auth(client):
    assert client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "discarded"}
    ).status_code == 401


def test_patch_discards_and_restores_a_take(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    r = auth_client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "discarded"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "discarded"
    assert r.json()["index"] == 0
    assert _feed(auth_client)[0]["takes"][0]["state"] == "discarded"
    # The sibling is untouched.
    assert _feed(auth_client)[0]["takes"][1]["state"] == "fresh"

    back = auth_client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "fresh"}
    )
    assert back.status_code == 200
    assert back.json()["state"] == "fresh"


def test_patch_returns_the_take_in_the_feed_shape(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    body = auth_client.patch(
        "/api/generator/generations/task-1/takes/1", json={"state": "discarded"}
    ).json()

    assert set(body) == {
        "index", "file", "prompt", "lyrics", "metas", "seed_value",
        "decoded_path", "state", "published_track_id",
    }
    assert body["decoded_path"] == TAKE_1_FILE


@pytest.mark.parametrize(
    "body",
    [
        {"state": "published"},   # published is earned by publishing, never set
        {"state": "PUBLISHED"},
        {"state": "nonsense"},
        {"state": ""},
        {"state": None},
        {},                       # state is required
        {"state": "fresh", "published_track_id": "techno--sneaky"},  # extra=forbid
    ],
)
def test_patch_422_on_an_invalid_state(auth_client, ace_on, monkeypatch, body):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    r = auth_client.patch("/api/generator/generations/task-1/takes/0", json=body)

    assert r.status_code == 422, body
    assert _feed(auth_client)[0]["takes"][0]["state"] == "fresh"


def test_patch_404_on_an_unknown_generation(auth_client):
    r = auth_client.patch(
        "/api/generator/generations/nope/takes/0", json={"state": "discarded"}
    )
    assert r.status_code == 404


def test_patch_404_on_an_unknown_take_index(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)

    r = auth_client.patch(
        "/api/generator/generations/task-1/takes/7", json={"state": "discarded"}
    )

    assert r.status_code == 404
    assert "take 7" in r.json()["detail"]


def test_patch_404_on_a_pending_generation_with_no_takes_yet(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)

    r = auth_client.patch(
        "/api/generator/generations/task-1/takes/0", json={"state": "discarded"}
    )

    assert r.status_code == 404


def test_discarding_a_published_take_keeps_its_track_id(
    auth_client, ace_on, monkeypatch
):
    """The catalog entry is a fact about the past, not a state the feed owns."""
    _release(auth_client, monkeypatch)
    _poll(auth_client, monkeypatch)
    db.mark_take_published(_user_id(), TAKE_1_FILE, "techno--neon-rain")

    body = auth_client.patch(
        "/api/generator/generations/task-1/takes/1", json={"state": "discarded"}
    ).json()

    assert body["state"] == "discarded"
    assert body["published_track_id"] == "techno--neon-rain"


# ══ POST .../refresh — stale is not degraded ═════════════════════════


def test_refresh_requires_auth(client):
    assert client.post(
        "/api/generator/generations/task-1/refresh"
    ).status_code == 401


def test_refresh_404_on_an_unknown_generation(auth_client, ace_on, monkeypatch):
    calls = _install_ace(monkeypatch, _box(results=[_entry()]))

    r = auth_client.post("/api/generator/generations/nope/refresh")

    assert r.status_code == 404
    assert calls == []


def test_refresh_finishes_a_pending_generation(auth_client, ace_on, monkeypatch):
    """The resume lane: the tab died, the takes land anyway."""
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[_entry()]))

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done"
    assert body["degraded"] is False
    assert [t["index"] for t in body["takes"]] == [0, 1]
    assert body["takes"][0]["decoded_path"] == TAKE_0_FILE
    assert _feed(auth_client)[0]["status"] == "done"


def test_refresh_marks_failed_when_ace_reports_a_failure(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[_entry(status=2, takes=())]))

    body = auth_client.post("/api/generator/generations/task-1/refresh").json()

    assert body["status"] == "failed"
    assert body["degraded"] is False


def test_refresh_keeps_a_still_running_generation_pending(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[_entry(status=0, takes=())]))

    body = auth_client.post("/api/generator/generations/task-1/refresh").json()

    assert body["status"] == "pending"
    assert body["degraded"] is False   # ACE answered; the job is simply running


def test_refresh_marks_stale_when_ace_answers_without_the_task(
    auth_client, ace_on, monkeypatch
):
    """ACE ANSWERED and does not know it: the 24 h record window closed."""
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[]))

    body = auth_client.post("/api/generator/generations/task-1/refresh").json()

    assert body["status"] == "stale"
    assert body["degraded"] is False   # nothing is degraded — this is the answer
    assert _feed(auth_client)[0]["status"] == "stale"


def test_refresh_marks_stale_when_the_batch_holds_someone_elses_task(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[_entry(task_id="another-task")]))

    body = auth_client.post("/api/generator/generations/task-1/refresh").json()

    assert body["status"] == "stale"
    assert body["takes"] == []


def test_refresh_degrades_on_a_transport_blip(auth_client, ace_on, monkeypatch):
    """The box being off says NOTHING about the job — still pending."""
    _release(auth_client, monkeypatch)

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_ace(monkeypatch, refuse)

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["degraded"] is True
    assert _feed(auth_client)[0]["status"] == "pending"   # never marked stale


def test_refresh_degrades_on_an_ace_500(auth_client, ace_on, monkeypatch):
    _release(auth_client, monkeypatch)

    def boom(request):
        if request.url.path == "/query_result":
            return httpx.Response(500, json=_envelope(None, code=500, error="boom"))
        return _box()(request)

    _install_ace(monkeypatch, boom)

    body = auth_client.post("/api/generator/generations/task-1/refresh").json()

    assert body["status"] == "pending" and body["degraded"] is True


def test_refresh_502_on_a_bad_api_key(auth_client, ace_on, monkeypatch):
    """A misconfiguration is not a blip — retrying will never fix it."""
    _release(auth_client, monkeypatch)

    def unauthorized(request):
        if request.url.path == "/query_result":
            return httpx.Response(401, json=_envelope(None, code=401, error="nope"))
        return _box()(request)

    _install_ace(monkeypatch, unauthorized)

    assert auth_client.post(
        "/api/generator/generations/task-1/refresh"
    ).status_code == 502


def test_refresh_503_when_the_generator_is_disabled(
    auth_client, ace_on, monkeypatch
):
    _release(auth_client, monkeypatch)
    monkeypatch.delenv(ac.ENV_BASE_URL, raising=False)
    calls = _install_ace(monkeypatch, _box(results=[_entry()]))

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 503
    assert calls == []


@pytest.mark.parametrize("terminal", ["done", "failed", "stale"])
def test_refresh_409_on_a_terminal_generation(
    auth_client, ace_on, monkeypatch, terminal
):
    """All three terminal states have nothing left to poll for."""
    _release(auth_client, monkeypatch)
    db.set_generation_status("task-1", _user_id(), terminal)
    calls = _install_ace(monkeypatch, _box(results=[_entry()]))

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 409
    assert terminal in r.json()["detail"]
    assert calls == []   # no pointless round trip to the box


def test_refresh_is_allowed_during_a_live_session(
    auth_client, ace_on, monkeypatch, clean_live_registry
):
    """Only RELEASING work touches the GPU; a re-poll costs no VRAM."""
    _release(auth_client, monkeypatch)
    _install_ace(monkeypatch, _box(results=[_entry()]))
    clean_live_registry[("sess-live", generator.LIVE_WS_CHANNEL)] = _FakeWS()

    r = auth_client.post("/api/generator/generations/task-1/refresh")

    assert r.status_code == 200
    assert r.json()["status"] == "done"


# ══ The store, directly (the invariants the SQL owns) ════════════════


def test_record_generation_is_idempotent_on_the_task_id(tmp_db):
    assert db.record_generation("t", 1, {"prompt": "a"}) is True
    assert db.record_generation("t", 1, {"prompt": "b"}) is False
    assert db.get_generation("t")["request"] == {"prompt": "a"}


def test_save_generation_takes_refuses_an_unknown_generation(tmp_db):
    assert db.save_generation_takes("nope", 1, [{"index": 0}]) is False
    assert db.get_generation("nope") is None


def test_set_generation_status_is_user_scoped(tmp_db):
    db.record_generation("t", 1, {})

    assert db.set_generation_status("t", 2, "done") is False
    assert db.get_generation("t")["status"] == "pending"
    assert db.set_generation_status("t", 1, "done") is True


def test_mark_take_published_reports_no_match(tmp_db):
    assert db.mark_take_published(1, "/nowhere/take.wav", "techno--x") == 0


def test_a_corrupt_json_column_degrades_to_an_empty_object(tmp_db):
    """The feed must not die on one bad row (the poll endpoint's rule)."""
    db.record_generation("t", 1, {})
    with db._conn() as c:
        c.execute("UPDATE generations SET request_json = '{not json' WHERE id = 't'")
        c.execute(
            "INSERT INTO generation_takes (generation_id, idx, metas_json, state) "
            "VALUES ('t', 0, 'nope', 'fresh')"
        )
        c.commit()

    gen = db.get_generation("t")

    assert gen["request"] == {}
    assert gen["takes"][0]["metas"] == {}
