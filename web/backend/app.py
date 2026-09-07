"""
ApolloAgents Web Backend — FastAPI application (v2.0)

Startup:
    uvicorn backend.app:app --reload --port 4020 --app-dir web
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from . import covers, db, auth, pipeline, youtube_auth
from .generator import router as generator_router
from .render import router as render_router
from .models import (
    CreateSessionRequest,
    EditorCommandRequest,
    GenreIntentRequest,
    LoginRequest,
    PlaylistAddTracks,
    PlaylistCreate,
    PlaylistRename,
    PlaylistReorder,
    RatingRequest,
    RatingUpdate,
    RegisterRequest,
    SessionEditorCommand,
    SessionTrackInsert,
    SessionTracksReorder,
    TokenResponse,
)
from .session_store import store
from .ws_manager import ws_manager


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="ApolloAgents API", version="2.0.0", lifespan=lifespan)

_DEFAULT_ORIGINS = "http://localhost:4010,http://127.0.0.1:4010"
_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("APOLLO_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v2.6.0 — async render endpoints + downloads. Lives in its own module
# because the SSE generator + in-memory _jobs registry are sizeable.
app.include_router(render_router)

# G0 — ACE-Step generator feature flag (+ the VRAM guard G1 enforces).
app.include_router(generator_router)


# --- beatmatch feedback loop (W1) ------------------------------------------
# Where the browser-streamed per-transition measurements land. The
# beatmatch-learn loop reads this file to learn per-profile timing offsets.
# Project root = parent of the backend package's parent (web/backend → repo).
_BEATMATCH_MEASUREMENTS_PATH = (
    Path(pipeline._PROJECT_DIR) / "loops" / "beatmatch-learn" / "measurements.jsonl"
)

# Required fields a measurement must carry to be persisted (W1 acceptance).
_BEATMATCH_REQUIRED = ("profile", "offset_ms")


def record_beatmatch_measurement(
    msg: dict, *, path: Path | None = None
) -> bool:
    """Append one validated beatmatch measurement as a JSON line.

    Returns True if a line was written, False if the message was dropped as
    malformed (missing required field, non-numeric offset). Creates the parent
    directory and file on first write. Raises only on unexpected I/O errors
    (the caller logs and continues — a bad measurement must never kill the WS).

    A valid measurement carries at least ``profile`` (str) and ``offset_ms``
    (number); ``pitch_bend_ms``, ``key_pair``, ``bpm_bucket``, ``ts`` are
    optional. See ``beatmatch-learn-knowledge`` for the schema + sign rubric.
    """
    target = path or _BEATMATCH_MEASUREMENTS_PATH

    # Validation — drop (return False), do not raise, on malformed input.
    for field in _BEATMATCH_REQUIRED:
        if field not in msg or msg[field] is None:
            return False
    if not isinstance(msg.get("profile"), str) or not msg["profile"]:
        return False
    try:
        offset_ms = float(msg["offset_ms"])
    except (TypeError, ValueError):
        return False

    record: dict = {"profile": msg["profile"], "offset_ms": offset_ms}
    # Optional fields, copied through only when present + well-typed.
    if msg.get("pitch_bend_ms") is not None:
        try:
            record["pitch_bend_ms"] = float(msg["pitch_bend_ms"])
        except (TypeError, ValueError):
            return False
    for opt in ("key_pair", "bpm_bucket", "ts"):
        if isinstance(msg.get(opt), str) and msg[opt]:
            record[opt] = msg[opt]

    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Single write() of one line is atomic enough for concurrent appenders on
    # a local fs (POSIX append semantics); keeps W1 dependency-free.
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    return True


# Genre Guard cap: max user turns before we treat the conversation as
# stuck and emit the "Could not confirm genre" error. See issue #23.
MAX_GENRE_TURNS = 8


def _should_emit_genre_error(history: list[dict]) -> bool:
    """Return True when the genre-guard turn ended without a confirmation
    AND the agent gave up (empty/whitespace assistant response, OR the
    user has hit MAX_GENRE_TURNS without confirming).

    Returns False on the normal "still asking, awaiting user reply" turn,
    so the handler should leave ``s.phase`` at ``"genre"`` instead of
    surfacing a misleading red error banner. See issue #23.
    """
    last_assistant = next(
        (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
        "",
    )
    if not (last_assistant or "").strip():
        return True
    user_turns = sum(1 for m in history if m.get("role") == "user")
    return user_turns >= MAX_GENRE_TURNS


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    if db.get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    user_id = db.create_user(req.username, req.email, auth.hash_password(req.password))
    token = auth.create_access_token({"sub": str(user_id)})
    return TokenResponse(
        access_token=token,
        user={"id": user_id, "username": req.username, "email": req.email},
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = db.get_user_by_username(req.username)
    if not user or not auth.verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = auth.create_access_token({"sub": str(user["id"])})
    return TokenResponse(
        access_token=token,
        user={"id": user["id"], "username": user["username"], "email": user["email"]},
    )


@app.get("/api/auth/me")
async def me(current_user: dict = Depends(auth.get_current_user)):
    return {"id": current_user["id"], "username": current_user["username"], "email": current_user["email"]}


# ---------------------------------------------------------------------------
# YouTube OAuth (v2.7) — read-only access to the operator's own live chat.
# All four routes 404 cleanly when GOOGLE_CLIENT_ID / SECRET / REDIRECT_URI
# aren't in the env, so unconfigured installs behave exactly like v2.6.x.
# ---------------------------------------------------------------------------


def _yt_guard() -> None:
    if not youtube_auth.enabled():
        raise HTTPException(status_code=404, detail="YouTube integration not configured")


@app.get("/api/youtube/oauth/start")
async def youtube_oauth_start(token: str = Query(...)):
    """Issue a 302 to Google's consent screen with a signed state token.

    The operator's browser follows the redirect, consents, and Google
    bounces back to ``/api/youtube/oauth/callback`` with ``?code=&state=``.

    Auth is via a query-string JWT (``?token=``) rather than the usual
    Authorization header because the frontend opens this URL with
    ``window.open()``, which can't attach custom headers. Same pattern
    we already use for WS upgrades and render SSE — see
    ``auth.user_from_query_token`` and ``test_auth_query_token``.
    """
    _yt_guard()
    user = auth.user_from_query_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    url = youtube_auth.start_oauth_flow(user["id"])
    return RedirectResponse(url=url, status_code=302)


@app.get("/api/youtube/oauth/callback")
async def youtube_oauth_callback(code: str = Query(...), state: str = Query(...)):
    """Complete the OAuth flow + redirect the operator back to /live.

    Note: this endpoint is hit by the operator's browser (via the
    Google redirect) — there's no Authorization header, so we
    authenticate via the HMAC-signed ``state`` token rather than the
    usual Bearer JWT.
    """
    _yt_guard()
    try:
        summary = await youtube_auth.handle_callback(code=code, state=state)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Redirect operator to /live with a hint flag the UI can render.
    front = os.environ.get("APOLLO_FRONTEND_URL", "http://localhost:4040")
    return RedirectResponse(
        url=f"{front}/live?yt_connected={summary['channel_id']}",
        status_code=302,
    )


@app.get("/api/youtube/status")
async def youtube_status(current_user: dict = Depends(auth.get_current_user)):
    """Return whether the current user has linked YouTube + which channel."""
    _yt_guard()
    summary = youtube_auth.channel_summary(current_user["id"])
    if summary is None:
        return {"connected": False}
    return {
        "connected": True,
        "channel_id": summary["channel_id"],
        "channel_title": summary["channel_title"],
    }


@app.post("/api/youtube/disconnect")
async def youtube_disconnect(current_user: dict = Depends(auth.get_current_user)):
    _yt_guard()
    removed = youtube_auth.disconnect(current_user["id"])
    return {"disconnected": removed}


# ---------------------------------------------------------------------------
# Catalog (read-only browse of tracks.json)
# ---------------------------------------------------------------------------

@app.get("/api/catalog")
async def get_catalog(
    genre: str | None = Query(None, description="Filter by genre_folder (case-insensitive)"),
    current_user: dict = Depends(auth.get_current_user),
):
    try:
        tracks, genres = await asyncio.to_thread(pipeline.load_catalog, genre)
    except pipeline.CatalogUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Hydrate each track with the calling user's own rating (or null). This is
    # always scoped to current_user.id — ratings from other users never leak.
    ratings = await asyncio.to_thread(db.get_user_ratings, current_user["id"])
    # ``cover_url`` is the LOCAL cover, present only for tracks Apollo
    # generated itself. Suno-imported tracks keep their remote
    # ``suno.cover_url`` untouched; the frontend prefers the local one
    # when both exist, so a republished take can supersede a stale
    # remote image without the catalog having to rewrite tracks.json.
    enriched = [
        {
            **t,
            "user_rating": ratings.get(t.get("id")),
            "cover_url": covers.cover_url_for(t.get("id") or ""),
        }
        for t in tracks
    ]
    return {"tracks": enriched, "genres": genres}


# ---------------------------------------------------------------------------
# Audio streaming — single endpoint that browser <audio> can hit directly.
# JWT travels in the query string (browsers can't set Authorization on
# <audio>), mirroring the WebSocket pattern above.
# ---------------------------------------------------------------------------

_STREAM_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def _is_within(path: Path, root: Path) -> bool:
    """True if `path` is the same as or a descendant of `root`. Both paths
    must already be absolute and resolved."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _stream_authorize(token: str) -> dict:
    """Decode the JWT from the query string and load the user, or raise 401."""
    user = auth.user_from_query_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


def _resolve_track_path(track_id: str) -> Path:
    """Look up `track_id` in the catalog and return a safe absolute path.

    Raises 404 for unknown ids or any path that escapes the allowed roots
    (`tracks/` for real catalog entries, `.tmp/` for the mock-pipeline
    fixture used in E2E), and 415 for extensions outside `.wav` / `.mp3`.
    """
    try:
        tracks, _ = pipeline.load_catalog(None)
    except pipeline.CatalogUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    track = next((t for t in tracks if t.get("id") == track_id), None)
    if not track or not track.get("file"):
        raise HTTPException(status_code=404, detail="Track not found")

    project_root = Path(pipeline._PROJECT_DIR).resolve()
    # Path-traversal defence: the resolved file must live under one of the
    # allowed roots. `tracks/` covers real catalog entries; `.tmp/` is the
    # mock-pipeline fixture root used during E2E (see #13 — keeps the mock
    # silence file out of tracks/ so --build-catalog never picks it up).
    allowed_roots = [
        (project_root / "tracks").resolve(),
        (project_root / ".tmp").resolve(),
    ]

    # Prefer the pre-encoded MP3 sibling when ``--build-catalog`` produced
    # one. The browser plays a 192 kbps MP3 in a single Range request
    # instead of paginating an 80 MB WAV — which is what was causing the
    # mid-track stalls (see ``main._ensure_mp3_for``). Fall back to the
    # original WAV only when the MP3 is absent or unresolvable.
    raw_path: Path | None = None
    mp3_rel = track.get("mp3_file")
    if mp3_rel:
        candidate = (project_root / mp3_rel).resolve()
        if (
            any(_is_within(candidate, root) for root in allowed_roots)
            and candidate.exists()
            and candidate.is_file()
        ):
            raw_path = candidate
    if raw_path is None:
        raw_path = (project_root / track["file"]).resolve()

    if not any(_is_within(raw_path, root) for root in allowed_roots):
        raise HTTPException(status_code=404, detail="Track not found")

    if not raw_path.exists() or not raw_path.is_file():
        raise HTTPException(status_code=404, detail="Track file missing")

    suffix = raw_path.suffix.lower()
    if suffix not in _STREAM_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio format: {suffix}")

    return raw_path


@app.get("/api/tracks/{track_id}/stream")
async def stream_track(track_id: str, token: str = Query(...)):
    _stream_authorize(token)
    path = await asyncio.to_thread(_resolve_track_path, track_id)
    media_type = _STREAM_MEDIA_TYPES[path.suffix.lower()]
    # Starlette's FileResponse handles Range/206 Partial Content natively,
    # which is what makes seek + iOS playback work without extra plumbing.
    return FileResponse(str(path), media_type=media_type)


@app.get("/api/tracks/{track_id}/cover")
async def track_cover(track_id: str, token: str = Query(...)):
    """Serve a locally generated cover for a catalog track.

    The token travels in the query string for the same reason it does on
    ``/stream``: an ``<img src>`` cannot carry an Authorization header
    any more than an ``<audio src>`` can.

    404 for a track with no cover — the overwhelming majority, since 510
    of the catalog's tracks were imported from Suno and carry a remote
    ``suno.cover_url`` instead. The frontend treats a missing local
    cover as "fall back to whatever the track already had", so this must
    stay a plain 404 and not an error.
    """
    _stream_authorize(token)
    path = covers.cover_path(track_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="No cover for this track")
    return FileResponse(str(path), media_type="image/png")


# ---------------------------------------------------------------------------
# Playlists (v2.2.1) — named, ordered collections of catalog tracks. The
# tracks themselves stay in `tracks/tracks.json`; only the playlist shell and
# the ordered track_ids live in SQLite. GET hydrates each id back into a full
# Track via the catalog, so the frontend never re-resolves metadata itself.
# ---------------------------------------------------------------------------


def _own_playlist(playlist_id: int, user: dict) -> dict:
    p = db.get_playlist(playlist_id)
    if not p or p["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return p


@app.post("/api/playlists", status_code=201)
async def create_playlist(
    req: PlaylistCreate,
    current_user: dict = Depends(auth.get_current_user),
):
    return db.create_playlist(current_user["id"], req.name)


@app.get("/api/playlists")
async def list_playlists(current_user: dict = Depends(auth.get_current_user)):
    return db.list_playlists_by_user(current_user["id"])


@app.get("/api/playlists/{playlist_id}")
async def get_playlist_detail(
    playlist_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    p = _own_playlist(playlist_id, current_user)

    # Warm the catalog cache once (no-op on repeated GETs) so the per-track
    # get_track_by_id lookups below are O(1) against the pre-built index
    # instead of re-reading tracks.json + rebuilding a by_id dict per call.
    try:
        await asyncio.to_thread(pipeline.load_catalog, None)
    except pipeline.CatalogUnavailable:
        pass

    hydrated: list[dict] = []
    for tid in p["track_ids"]:
        t = pipeline.get_track_by_id(tid)
        if t is None:
            # Catalog may have been rebuilt and dropped this id — surface it
            # rather than failing so the user can clean up the playlist.
            hydrated.append({"id": tid, "display_name": tid, "missing": True})
        else:
            hydrated.append(t)

    return {
        "id": p["id"],
        "user_id": p["user_id"],
        "name": p["name"],
        "created_at": p["created_at"],
        "updated_at": p["updated_at"],
        "tracks": hydrated,
    }


@app.patch("/api/playlists/{playlist_id}")
async def rename_playlist_endpoint(
    playlist_id: int,
    req: PlaylistRename,
    current_user: dict = Depends(auth.get_current_user),
):
    _own_playlist(playlist_id, current_user)
    db.rename_playlist(playlist_id, req.name)
    refreshed = db.get_playlist(playlist_id)
    assert refreshed is not None
    return {
        "id": refreshed["id"],
        "user_id": refreshed["user_id"],
        "name": refreshed["name"],
        "created_at": refreshed["created_at"],
        "updated_at": refreshed["updated_at"],
        "track_count": len(refreshed["track_ids"]),
    }


@app.delete("/api/playlists/{playlist_id}", status_code=204)
async def delete_playlist_endpoint(
    playlist_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    _own_playlist(playlist_id, current_user)
    db.delete_playlist(playlist_id)


@app.post("/api/playlists/{playlist_id}/tracks")
async def add_tracks_endpoint(
    playlist_id: int,
    req: PlaylistAddTracks,
    current_user: dict = Depends(auth.get_current_user),
):
    _own_playlist(playlist_id, current_user)
    new_count = db.add_tracks_to_playlist(playlist_id, req.track_ids)
    return {"playlist_id": playlist_id, "track_count": new_count}


@app.delete("/api/playlists/{playlist_id}/tracks/{track_id}", status_code=204)
async def remove_track_endpoint(
    playlist_id: int,
    track_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    _own_playlist(playlist_id, current_user)
    if not db.remove_track_from_playlist(playlist_id, track_id):
        raise HTTPException(status_code=404, detail="Track not in playlist")


@app.put("/api/playlists/{playlist_id}/order")
async def reorder_endpoint(
    playlist_id: int,
    req: PlaylistReorder,
    current_user: dict = Depends(auth.get_current_user),
):
    _own_playlist(playlist_id, current_user)
    ok = db.reorder_playlist_tracks(playlist_id, req.track_ids)
    if not ok:
        raise HTTPException(
            status_code=422,
            detail="track_ids must match the playlist's current track_ids exactly",
        )
    refreshed = db.get_playlist(playlist_id)
    assert refreshed is not None
    return {
        "id": refreshed["id"],
        "track_ids": refreshed["track_ids"],
        "updated_at": refreshed["updated_at"],
    }


# ---------------------------------------------------------------------------
# Per-user track ratings (1–5). The catalog endpoint above hydrates the
# `user_rating` field for the calling user; these two routes let the user
# create / update / clear their own rating.
# ---------------------------------------------------------------------------

@app.put("/api/tracks/{track_id}/rating")
async def set_track_rating(
    track_id: str,
    body: RatingUpdate,
    current_user: dict = Depends(auth.get_current_user),
):
    await asyncio.to_thread(
        db.upsert_track_rating, current_user["id"], track_id, body.rating
    )
    return {"track_id": track_id, "rating": body.rating}


@app.delete("/api/tracks/{track_id}/rating", status_code=204)
async def clear_track_rating(
    track_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    # Idempotent: deleting a non-existent rating succeeds with 204 anyway —
    # the UI fires DELETE on every "click on filled star" without first
    # checking the row exists.
    await asyncio.to_thread(
        db.delete_track_rating, current_user["id"], track_id
    )
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Sessions — REST CRUD
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def create_session(
    req: CreateSessionRequest | None = None,
    current_user: dict = Depends(auth.get_current_user),
):
    """Create a session. v2.6.0 — when ``brief`` is provided, parse it
    synchronously and kick off planning + critique as a background task
    so the frontend can navigate to ``/curate`` immediately and watch
    progress via ``/ws/sessions/{id}``.

    The parse runs on whichever provider is configured (v3.11), so its
    cost is provider-dependent: ~300 ms on Haiku, seconds on a local
    model, bounded by ``brief_parser.TIMEOUT_SEC``.

    Legacy callers (the v2.5 ``/session/[id]`` route) post no body and
    still get an empty session back — the brief flow is purely additive.
    """
    s = store.create(current_user["id"])
    parsed: dict | None = None

    if req and req.brief and req.brief.strip():
        from . import brief_parser  # local import — keeps the SDK
        # dependency optional for tests that monkeypatch the function.
        parsed = await asyncio.to_thread(brief_parser.parse, req.brief)
        s.context_variables["brief_text"] = req.brief
        if req.environment and req.environment.strip():
            s.context_variables["environment"] = req.environment.strip()
        # Seed only non-null parsed fields — the planner falls back to
        # phase_genre_guard for anything the parser couldn't extract.
        s.context_variables.update({k: v for k, v in parsed.items() if v is not None})
        # Optimistic phase — the background task refines it precisely
        # ("genre" if guard is needed, "planning" after the planner runs).
        s.phase = "planning"
        store.save(s)

        async def _emit(data: dict) -> None:
            await ws_manager.send(s.id, data)

        async def _drive_planning() -> None:
            try:
                await pipeline.run_planning_from_brief(s, _emit)
            except Exception as exc:  # noqa: BLE001 — surface to the UI
                await _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                store.save(s)

        asyncio.create_task(_drive_planning())

    response = s.to_dict()
    if parsed is not None:
        response["parsed"] = parsed
    return response


@app.get("/api/sessions")
async def list_sessions(current_user: dict = Depends(auth.get_current_user)):
    return [s.to_dict() for s in store.get_user_sessions(current_user["id"])]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, current_user: dict = Depends(auth.get_current_user)):
    return _own(session_id, current_user).to_dict()


# ---------------------------------------------------------------------------
# Critic notes — Apply / Ignore (v2.6.0)
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/notes/{note_id}/apply")
async def apply_note(
    session_id: str,
    note_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    """Run a bounded editor agent turn that addresses one critic note.

    The agent receives a synthetic prompt naming the note's position and
    body and is told to fire exactly one editing tool. After it returns
    we mark the note as ``applied`` and recompute set-health; the note
    text + position stay in ``structured_problems`` so re-fetching shows
    it greyed out instead of disappearing.
    """
    from .notes import note_id as derive_note_id  # noqa: PLC0415

    s = _own(session_id, current_user)
    if s.phase == "performing":
        raise HTTPException(status_code=409, detail="Live session in progress")

    target = None
    for p in s.structured_problems or []:
        if derive_note_id(p) == note_id:
            target = p
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Note not found")

    # Idempotent: a second apply is a no-op (returning current state). The
    # frontend disables the button while pending, but a stale tab might
    # still fire — don't penalise it with a 409.
    if s.handled_notes.get(note_id) == "applied":
        return s.to_dict()

    # Auto-promote phase to "editing" so phase_editor's history bucket is
    # populated and the WS dispatcher's editing-phase guards apply. Mirrors
    # the "checkpoint2 → editing" jump in `_handle_ws_message`.
    if s.phase in {"critique", "checkpoint2"}:
        s.phase = "editing"
        s.messages.setdefault("editor", [])

    pos = target.get("pos_from") or target.get("pos_to") or 1
    msg = (
        f"Address only this critic note (and nothing else): "
        f"{target.get('text', '')}. "
        f"Position: {pos}. Use exactly one tool call "
        f"(swap_track / insert_track / remove_track / move_track) and "
        f"report the result briefly."
    )

    async def _emit(data: dict) -> None:
        await ws_manager.send(s.id, data)

    history = s.messages.setdefault("editor", [])
    try:
        await pipeline.phase_editor(msg, history, s.context_variables, _emit)
    except Exception as exc:  # noqa: BLE001 — surface to caller as 422
        raise HTTPException(
            status_code=422,
            detail=f"apply failed: {type(exc).__name__}: {exc}",
        ) from exc

    s.handled_notes[note_id] = "applied"
    s.set_health = pipeline.compute_set_health(s.structured_problems)
    store.save(s)
    # Tell subscribed tabs to refresh — they listen for phase_complete on
    # `/ws/sessions/{id}` and merge the payload into local state.
    await _emit({"type": "phase_complete", "phase": "editor_turn", "data": s.to_dict()})
    return s.to_dict()


@app.post("/api/sessions/{session_id}/notes/{note_id}/ignore")
async def ignore_note(
    session_id: str,
    note_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    """Mark a critic note as ignored. Idempotent — re-firing is a no-op.

    No agent turn, no playlist mutation; the note text + position stay in
    ``structured_problems`` so the frontend can render it greyed-out.
    """
    s = _own(session_id, current_user)
    if s.handled_notes.get(note_id) != "ignored":
        s.handled_notes[note_id] = "ignored"
        store.save(s)
        await ws_manager.send(
            s.id,
            {"type": "phase_complete", "phase": "editor_turn", "data": s.to_dict()},
        )
    return {"handled": list(s.handled_notes.keys())}


# ---------------------------------------------------------------------------
# Session tracks — deterministic UI gestures (v2.6.0)
#
# Drag-reorder, trash, and TrackPicker insert call these endpoints. The LLM
# editor command is exposed below via SSE so the agent can use the same
# tools the WS dispatcher already does — these are the *deterministic*
# counterparts that bypass an LLM turn for tiny gestures.
# ---------------------------------------------------------------------------

def _editor_phase_guard(s) -> None:
    """Reject during a live broadcast; auto-promote critique→editing.

    Mirrors the WS dispatcher's behaviour (`_handle_ws_message` at
    ``"editor_command"``): edits during ``critique`` / ``checkpoint2``
    promote the session phase so subsequent gestures land in the editor
    history. Live sessions reject with 409.
    """
    if s.phase == "performing":
        raise HTTPException(status_code=409, detail="Live session in progress")
    if s.phase in {"critique", "checkpoint2"}:
        s.phase = "editing"
        s.messages.setdefault("editor", [])


async def _broadcast_editor_turn(s) -> None:
    """Emit a `phase_complete editor_turn` so other tabs sync."""
    await ws_manager.send(
        s.id,
        {"type": "phase_complete", "phase": "editor_turn", "data": s.to_dict()},
    )


@app.post("/api/sessions/{session_id}/tracks/reorder")
async def reorder_session_tracks(
    session_id: str,
    req: SessionTracksReorder,
    current_user: dict = Depends(auth.get_current_user),
):
    s = _own(session_id, current_user)
    _editor_phase_guard(s)
    playlist = list(s.context_variables.get("playlist") or [])
    n = len(playlist)
    if sorted(req.order) != list(range(n)):
        raise HTTPException(
            status_code=422,
            detail=f"order must be a permutation of [0..{n})",
        )
    s.context_variables["playlist"] = [playlist[i] for i in req.order]
    s.set_health = pipeline.compute_set_health(s.structured_problems)
    store.save(s)
    await _broadcast_editor_turn(s)
    return s.to_dict()


@app.delete("/api/sessions/{session_id}/tracks/{position}")
async def delete_session_track(
    session_id: str,
    position: int,
    current_user: dict = Depends(auth.get_current_user),
):
    s = _own(session_id, current_user)
    _editor_phase_guard(s)
    playlist = list(s.context_variables.get("playlist") or [])
    if position < 0 or position >= len(playlist):
        raise HTTPException(
            status_code=404,
            detail=f"Track index {position} out of range",
        )
    playlist.pop(position)
    s.context_variables["playlist"] = playlist
    s.set_health = pipeline.compute_set_health(s.structured_problems)
    store.save(s)
    await _broadcast_editor_turn(s)
    return s.to_dict()


@app.post("/api/sessions/{session_id}/tracks/insert")
async def insert_session_track(
    session_id: str,
    req: SessionTrackInsert,
    current_user: dict = Depends(auth.get_current_user),
):
    s = _own(session_id, current_user)
    _editor_phase_guard(s)
    track = pipeline.get_track_by_id(req.track_id)
    if track is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown track id: {req.track_id}",
        )
    playlist = list(s.context_variables.get("playlist") or [])
    at = max(0, min(req.at, len(playlist)))
    playlist.insert(at, track)
    s.context_variables["playlist"] = playlist
    s.set_health = pipeline.compute_set_health(s.structured_problems)
    store.save(s)
    await _broadcast_editor_turn(s)
    return s.to_dict()


@app.post("/api/sessions/{session_id}/editor_command")
async def editor_command(
    session_id: str,
    req: SessionEditorCommand,
    current_user: dict = Depends(auth.get_current_user),
):
    """SSE wrapper around ``pipeline.phase_editor``.

    The editor agent emits ``text_delta``/``tool_call``/``tool_progress``
    events; each becomes an SSE ``data:`` frame. After the turn ends, a
    terminal ``phase_complete`` frame carries the updated session and the
    stream closes with an ``event: done`` line.
    """
    s = _own(session_id, current_user)
    _editor_phase_guard(s)

    queue: asyncio.Queue = asyncio.Queue()

    async def _emit(data: dict) -> None:
        # `phase_editor` calls `await emit(...)`; we just funnel into the
        # SSE queue and return immediately (no awaiting).
        queue.put_nowait(data)

    async def _drive_turn() -> None:
        try:
            history = s.messages.setdefault("editor", [])
            await pipeline.phase_editor(req.text, history, s.context_variables, _emit)

            # If the editor's turn produced a build (via `build_session`
            # tool), chain validate just like the WS dispatcher does. This
            # keeps the SSE flow feature-parity with the legacy WS path.
            last_build = s.context_variables.get("last_build")
            if last_build:
                s.session_name = last_build
                s.phase = "validating"
                await _emit({"type": "phase_start", "phase": "validating"})
                v_status, v_issues = await pipeline.phase_validate(
                    last_build, s.context_variables, _emit
                )
                s.validator_status = v_status
                s.validator_issues = v_issues
                s.phase = "rating"
                await _emit({
                    "type": "phase_complete",
                    "phase": "validating",
                    "data": s.to_dict(),
                })
            else:
                # Recompute set_health after a non-build editor turn since
                # the agent may have mutated the playlist (swap/insert/etc).
                s.set_health = pipeline.compute_set_health(s.structured_problems)
                await _emit({
                    "type": "phase_complete",
                    "phase": "editor_turn",
                    "data": s.to_dict(),
                })
            queue.put_nowait({"_status": "done"})
        except Exception as exc:  # noqa: BLE001 — surface to caller
            queue.put_nowait({
                "_status": "error",
                "_message": f"{type(exc).__name__}: {exc}",
            })

    task = asyncio.create_task(_drive_turn())

    async def event_stream():
        try:
            while True:
                data = await queue.get()
                status = data.get("_status")
                if status == "done":
                    yield "event: done\ndata: {}\n\n"
                    break
                if status == "error":
                    payload = json.dumps({"message": data["_message"]})
                    yield f"event: error\ndata: {payload}\n\n"
                    break
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            # Persist the session whether the stream finished normally,
            # erred, or the client closed the EventSource early.
            if not task.done():
                task.cancel()
            store.save(s)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable Nginx response buffering for SSE — otherwise long
            # turns sit on the proxy until the first 8 KB or close.
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, current_user: dict = Depends(auth.get_current_user)):
    _own(session_id, current_user)
    store.delete(session_id)


# ---------------------------------------------------------------------------
# Rating (REST — called after the session completes)
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/rate")
async def rate_session(
    session_id: str,
    req: RatingRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    s = _own(session_id, current_user)
    ctx = s.context_variables
    result = await asyncio.to_thread(
        pipeline.write_session_record,
        session_name=s.session_name or ctx.get("last_build", "unnamed"),
        genre=ctx.get("genre", ""),
        duration_min=ctx.get("duration_min", 0),
        mood=ctx.get("mood", ""),
        rating=req.rating,
        notes=req.notes or "",
        critic_verdict=s.critic_verdict or "",
        critic_problems_json=json.dumps(s.critic_problems),
        validator_status=s.validator_status or "",
        validator_issues_json=json.dumps(s.validator_issues),
        tracks_swapped_json=json.dumps([]),
        final_playlist_json=json.dumps([t.get("display_name") for t in ctx.get("playlist", [])]),
        transition_ratings_json=json.dumps(req.transition_ratings or []),
        structured_problems_json=json.dumps(s.structured_problems),
        context_variables=ctx,
    )
    s.phase = "complete"
    store.save(s)
    return {"ok": True, "result": result}


# ---------------------------------------------------------------------------
# WebSocket — streaming pipeline channel
# ---------------------------------------------------------------------------

@app.websocket("/ws/sessions/{session_id}")
async def session_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    user = auth.user_from_query_token(token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    s = store.get(session_id)
    if not s or s.user_id != user["id"]:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws_manager.connect(session_id, websocket)

    async def emit(data: dict) -> None:
        await ws_manager.send(session_id, data)

    try:
        # Send current state on connect
        await emit({"type": "state", "data": s.to_dict()})

        while True:
            msg = await ws_manager.receive(session_id)
            if msg is None:
                break

            msg_type = msg.get("type")
            content = msg.get("content", "")

            try:
                await _handle_ws_message(s, msg_type, content, emit)
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # noqa: BLE001 — surface any phase failure as a UI banner
                await emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                # Persist after every message so a backend restart resumes at
                # the last committed phase/playlist/chat-history.
                store.save(s)

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id)


# ---------------------------------------------------------------------------
# WebSocket — live performance channel (v2.5.1)
# ---------------------------------------------------------------------------

@app.websocket("/ws/live/{session_id}")
@app.websocket("/api/sessions/{session_id}/live/stream")
async def live_session_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """Bridge between the browser-driven audio player and ``LiveEngineBrowser``.

    Available at two paths during the v2.6.0 cutover:
      - ``/ws/live/{session_id}`` (v2.5 legacy, kept as a deprecated alias).
      - ``/api/sessions/{session_id}/live/stream`` (v2.6 canonical).

    The live channel is a second websocket over the same ``session_id``,
    distinct from ``/ws/sessions/{id}`` (which carries planning events).
    On connect we resolve the session's last approved playlist, construct
    a :class:`LiveEngineBrowser` whose emitter forwards each event back
    over this websocket, and spawn the live DJ loop. The browser drives
    audio playback and pings ``playback_pos`` every ~250 ms so the engine
    can detect crossfade thresholds without a background thread.
    """
    user = auth.user_from_query_token(token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    s = store.get(session_id)
    if not s or s.user_id != user["id"]:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    playlist = list(s.context_variables.get("playlist") or [])
    if not playlist:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # v3.6.2 — drop playlist entries whose id no longer exists in the
    # catalog. Sessions created before a ``--build-catalog`` rebuild
    # keep the OLD track ids in their stored playlist; the browser's
    # buffer fetch then 404s on the first stale track and the whole
    # live session sits in silence with no recovery path (found live
    # 2026-07-11: a May session opened for a stream referenced deleted
    # ``LoFi-*`` files). Filtering here keeps the engine, the frontend
    # deck, and the ``/api/tracks/{id}/stream`` endpoint consistent —
    # they all resolve against the same ``pipeline.load_catalog``.
    try:
        catalog_tracks, _ = await asyncio.to_thread(pipeline.load_catalog, None)
        catalog_ids = {t.get("id") for t in catalog_tracks if t.get("id")}
    except pipeline.CatalogUnavailable:
        catalog_ids = None  # no catalog to validate against — let it ride
    if catalog_ids is not None:
        fresh = [t for t in playlist if t.get("id") in catalog_ids]
        dropped = len(playlist) - len(fresh)
        if dropped:
            print(
                f"[live-ws {session_id}] dropped {dropped} stale playlist "
                f"track(s) missing from the current catalog "
                f"({len(fresh)} remain)",
                flush=True,
            )
            playlist = fresh
    if not playlist:
        print(
            f"[live-ws {session_id}] every playlist track is stale — "
            "closing; regenerate the session against the current catalog",
            flush=True,
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # v2.7.2 — clean displacement of a prior primary on the same slot.
    # Without this, a refresh (or a second tab landing on /live without
    # ?viewer=1) silently overwrites the dict entry and the existing
    # handler keeps reading from the new socket, racing with the new
    # handler — the failure mode the viewer-WS split was meant to fix
    # for OBS-flagged URLs but which still affected plain /live.
    displaced = await ws_manager.displace_existing(
        session_id,
        code=4001,
        reason="replaced by new connection",
        channel="live",
    )
    if displaced:
        print(
            f"[live-ws {session_id}] displaced previous primary "
            f"(close code 4001)",
            flush=True,
        )

    await ws_manager.connect(session_id, websocket, channel="live")

    # Inject session identity into context_variables so the live tools can
    # see the current user (mirrors the planning WS handler).
    if "user_id" not in s.context_variables:
        s.context_variables["user_id"] = s.user_id
        user_row = db.get_user_by_id(s.user_id)
        if user_row:
            s.context_variables["username"] = user_row["username"]

    loop = asyncio.get_running_loop()

    from . import live_runtime  # noqa: PLC0415 — module-level import would
    # create a circular reference during the initial app.py import.

    async def emit(data: dict) -> None:
        # Send to this primary WS …
        await ws_manager.send(session_id, data, channel="live")
        # … and fan out to any read-only viewers attached to the
        # session's engine bus (OBS Browser Source via the /viewer
        # endpoint, debug consoles, etc.). The bus is created lazily
        # on first publish, so the no-viewer case is essentially a
        # dict lookup.
        try:
            await live_runtime.publish(user["id"], session_id, data)
        except Exception as exc:  # noqa: BLE001 — never let pub/sub kill the primary
            print(
                f"[live-ws {session_id}] live_runtime.publish failed: {exc}",
                flush=True,
            )

    # Engine events arrive on the engine's emitter (called from arbitrary
    # threads in the local engine, sync in the browser engine). Funnel
    # everything through the same asyncio.Queue the live_dj loop drains
    # so the agent sees one ordered stream of events + user commands.
    command_queue: asyncio.Queue = asyncio.Queue()

    # v3.7 — resume support. The engine is built PER WEBSOCKET, so a
    # browser reload previously restarted the set at track 0 (observed
    # live 2026-08-17: a tab drop at track 10 went back to track 0).
    #
    # We remember the TRACK ID rather than the index. The index is only
    # meaningful against the exact playlist that produced it, and the
    # playlist can be reordered, edited or replaced between connections —
    # a stale index would then resume on the wrong track, silently. An id
    # either resolves to a real position in the current playlist or it
    # does not, in which case we start over honestly.
    from agent.live_engine import resolve_resume_index  # noqa: PLC0415

    _playlist_ids = [t.get("id") for t in playlist]
    _resume_id = s.context_variables.get("_live_track_id")
    s.context_variables["_live_idx"] = resolve_resume_index(
        _playlist_ids, _resume_id
    )
    if s.context_variables["_live_idx"]:
        print(
            f"[live-ws {session_id}] resuming at #"
            f"{s.context_variables['_live_idx'] + 1} ({_resume_id})",
            flush=True,
        )

    def engine_emitter(event: dict) -> None:
        etype = event.get("type")
        if etype == "track_started":
            tid = (event.get("track") or {}).get("id")
            if tid:
                s.context_variables["_live_track_id"] = tid
        elif etype == "session_ended":
            # The set finished on its own terms; the next play() is a new
            # set and must start at the top, not resume the old ending.
            s.context_variables.pop("_live_track_id", None)
            s.context_variables["_live_idx"] = 0
        # Push into both the WS (so the browser sees it) and the agent
        # queue (so the live_dj loop reacts). The browser engine emits
        # synchronously from this same loop thread (no background watchdog),
        # so we schedule via the loop rather than blocking on
        # ``run_coroutine_threadsafe`` (which deadlocks when called from
        # the loop thread). Both are wrapped in try/except so flaky UI
        # plumbing never kills the engine.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            try:
                loop.create_task(emit(event))
            except Exception:
                pass
            try:
                command_queue.put_nowait(dict(event))
            except Exception:
                pass
        else:
            try:
                asyncio.run_coroutine_threadsafe(emit(event), loop)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(command_queue.put_nowait, dict(event))
            except Exception:
                pass

    from agent.live_engine import LiveEngineBrowser

    engine = LiveEngineBrowser(emitter=engine_emitter)

    # Spawn the live DJ loop. We supervise it ourselves so a crash in the
    # agent surfaces as a WS error event instead of silently dropping the
    # connection.
    async def run_phase() -> None:
        try:
            await pipeline.phase_live(
                playlist, s.context_variables, engine, emit, command_queue
            )
            print(f"[live-ws {session_id}] phase_live returned cleanly", flush=True)
        except asyncio.CancelledError:
            print(f"[live-ws {session_id}] phase_live cancelled", flush=True)
            raise
        except Exception as exc:  # noqa: BLE001 — surface to UI banner
            print(
                f"[live-ws {session_id}] phase_live crashed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            await emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    phase_task = asyncio.create_task(run_phase())

    # ── v3.6.3: server-side stall watchdog ─────────────────────────────
    # The browser engine is ping-driven; if the operator tab freezes
    # (Chrome Energy Saver) or a deck never starts, no ping ever crosses
    # a threshold again and the session wedges in silence (observed live
    # 2026-07-11). Poll the engine's stall check so the backend can
    # synthesise the missing track_ended and keep the set moving — any
    # attached OBS viewer follows the resulting load command and the
    # stream stays audible. Module attribute lookups (not from-imports)
    # so tests can monkeypatch the cadence/margin.
    import agent.live_engine as live_engine_mod  # noqa: PLC0415

    async def run_stall_watchdog() -> None:
        while True:
            await asyncio.sleep(live_engine_mod.LIVE_STALL_CHECK_SEC)
            try:
                # to_thread: a forced advance can run the endless
                # fallback, which reads tracks.json from disk.
                forced = await asyncio.to_thread(engine.check_stall)
            except Exception as exc:  # noqa: BLE001 — watchdog must outlive hiccups
                print(
                    f"[live-ws {session_id}] check_stall failed: {exc}",
                    flush=True,
                )
                continue
            if forced:
                print(
                    f"[live-ws {session_id}] stall watchdog forced advance "
                    f"past {forced!r}",
                    flush=True,
                )

    stall_task = asyncio.create_task(run_stall_watchdog())

    # ── v2.7.2: YouTube Live Chat ingest (session-scoped) ──────────────
    # The poller now lives in ``youtube_runtime`` keyed by
    # (user_id, session_id) — one poller per session regardless of how
    # many WS connections are attached. The WS handler subscribes to
    # the runtime, receives events via the two callbacks below, and
    # detaches in the finally block. A 30-s grace window survives
    # Chrome refreshes and OBS Browser Source reconnects without
    # respawning the poller / re-fetching chat backlog.
    from . import youtube_runtime  # noqa: PLC0415 — optional dep lazy

    from .chat_names import build_greeting_event, sanitize_display_name

    async def _on_yt_message(author: str, text: str, ts_ms: int, is_first: bool) -> None:
        """Per-WS callback: emit dj_chat visibility + enqueue for agent.

        v3.7.0 — ``is_first`` (author's first message this stream,
        computed once in youtube_runtime) drives the greeting overlay:
        a ``chat_greeting`` event fans out to the operator page and
        every OBS viewer, and the DJ's copy of the message is tagged so
        the LLM greets by name via emit_chat.
        """
        # Diagnostic — surfaces in .tmp/backend.log so we can confirm
        # the poller is firing when audience messages don't appear in
        # the UI. Cheap enough to keep in prod for now.
        print(
            f"[live-ws {session_id}] YT msg from @{author!r}: {text[:80]!r}"
            f"{' [first]' if is_first else ''}",
            flush=True,
        )
        greeting = build_greeting_event(author, is_first)
        if greeting is not None:
            try:
                await emit(greeting)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[live-ws {session_id}] chat_greeting emit failed: {exc}",
                    flush=True,
                )
        # Sanitized name in the DJ/panel framing too — a crafted
        # username must not be able to fake the [YT @...] envelope.
        safe_author = sanitize_display_name(author) or "viewer"
        first_tag = " - first message this stream" if greeting is not None else ""
        try:
            await emit({"type": "dj_chat", "text": f"[YT @{safe_author}] {text}"})
        except Exception as exc:  # noqa: BLE001
            print(f"[live-ws {session_id}] yt dj_chat emit failed: {exc}", flush=True)
        payload = {
            "type": "user_msg",
            "text": f"[YT @{safe_author}{first_tag}] {text}",
            "timestamp_ms": ts_ms or None,
        }
        try:
            await command_queue.put(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[live-ws {session_id}] yt enqueue failed: {exc}", flush=True)

    async def _on_yt_status(status_event: dict) -> None:
        """Per-WS callback: forward poller status frames to this WS."""
        await emit({"type": "youtube_status", **status_event})

    yt_subscription = None
    try:
        yt_subscription, yt_state = await youtube_runtime.attach(
            user_id=user["id"],
            session_id=session_id,
            on_message=_on_yt_message,
            on_status=_on_yt_status,
        )
        # Emit the current state snapshot immediately so this WS sees
        # whatever the existing runtime knows (or, on first attach,
        # the freshly-discovered state). State "off" means YT is
        # disabled server-side — skip the emit so the pill stays
        # hidden, matching v2.6.x behaviour.
        if yt_state.get("state") and yt_state["state"] != "off":
            await emit({"type": "youtube_status", **yt_state})
    except Exception as exc:  # noqa: BLE001 — never let YT setup take down the live session
        print(f"[live-ws {session_id}] yt attach failed: {exc}", flush=True)

    try:
        # Initial state event so the client knows what's loaded before any
        # engine events arrive.
        await emit(
            {
                "type": "live_state",
                "data": {
                    "session_id": session_id,
                    "playlist": playlist,
                    "engine_state": engine.get_state(),
                },
            }
        )

        while True:
            msg = await ws_manager.receive(session_id, channel="live")
            if msg is None:
                # ws_manager.receive swallows every exception (including
                # WebSocketDisconnect AND json parse errors) — log so we
                # can tell a clean disconnect from a malformed frame that
                # would otherwise silently tear down the session.
                print(
                    f"[live-ws {session_id}] receive returned None — "
                    "loop exiting (disconnect or parse error)",
                    flush=True,
                )
                break

            msg_type = msg.get("type")
            if msg_type == "playback_pos":
                tid = str(msg.get("track_id", ""))
                try:
                    current_time = float(msg.get("currentTime", 0.0))
                except (TypeError, ValueError):
                    current_time = 0.0
                # v3.9.7 — the deck's own rate/offset, when the frontend
                # sends them. The engine cannot infer these: the reported
                # position is DERIVED from them, so a deck at the wrong
                # rate still reports a self-consistent position.
                def _opt(name: str) -> float | None:
                    raw = msg.get(name)
                    if raw is None:
                        return None
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return None

                # report_playback_pos is sync but small (no I/O), so we can
                # call it directly on the event loop without to_thread.
                engine.report_playback_pos(
                    tid,
                    current_time,
                    deck_rate=_opt("deck_rate"),
                    deck_offset=_opt("deck_offset"),
                )
            elif msg_type == "track_ended":
                # v2.5.0.1 — the browser fires ``ended`` on its <audio>
                # element when natural playback finishes and the
                # ``setInterval`` polling stops advancing ``currentTime``.
                # The hook forwards that as a synthetic ``track_ended``
                # message so the engine can advance even if the
                # ``playback_pos`` watchdog never crossed the threshold.
                # v3.9.3 — ``source="client"`` (the default, passed
                # explicitly because this is the only caller that can be
                # a skip) makes the engine's ``[engine track_ended]``
                # diagnostic attribute the advance to the browser. The
                # frontend also sends this message when a decode fails
                # (``reportLoadFailure``), so a premature position here
                # is the signature of a skip-on-load-failure rather than
                # a track finishing.
                tid = str(msg.get("track_id", ""))
                engine.report_track_ended(tid, source="client")
            elif msg_type in {"user_msg", "command"}:
                text = msg.get("text") or msg.get("content") or ""
                await command_queue.put(
                    {
                        "type": "user_msg",
                        "text": str(text),
                        "timestamp_ms": msg.get("timestamp_ms"),
                    }
                )
            elif msg_type == "perception":
                # v2.5.2 — aggregated mic metric from the browser. Raw audio
                # never reaches the backend; only RMS / onset density / VAD
                # likelihood. The phase_live consumer maintains a ring buffer
                # and emits synthetic environment_changed events to the
                # agent when the window mean shifts.
                try:
                    rms_db = float(msg.get("rms_db", 0.0))
                except (TypeError, ValueError):
                    rms_db = 0.0
                try:
                    onset_density_hz = float(msg.get("onset_density_hz", 0.0))
                except (TypeError, ValueError):
                    onset_density_hz = 0.0
                vl_raw = msg.get("voice_likelihood")
                if vl_raw is None:
                    voice_likelihood: float | None = None
                else:
                    try:
                        voice_likelihood = float(vl_raw)
                    except (TypeError, ValueError):
                        voice_likelihood = None
                await command_queue.put(
                    {
                        "type": "perception_sample",
                        "rms_db": rms_db,
                        "onset_density_hz": onset_density_hz,
                        "voice_likelihood": voice_likelihood,
                        "timestamp_ms": msg.get("timestamp_ms"),
                    }
                )
            elif msg_type == "beatmatch_measurement":
                # W1 (beatmatch feedback loop) — one transition's measured
                # residual offset + optional human pitch-bend correction,
                # streamed from the browser (lib/live.ts crossfadeToNext).
                # Appended as one JSON line to measurements.jsonl, which the
                # beatmatch-learn loop consumes to learn per-profile offsets.
                # Malformed messages are dropped (logged), never fatal.
                try:
                    record_beatmatch_measurement(msg)
                except Exception as exc:  # noqa: BLE001 — never kill the WS loop
                    print(
                        f"[live-ws {session_id}] beatmatch_measurement dropped: {exc}",
                        flush=True,
                    )
            elif msg_type == "quit":
                await command_queue.put({"type": "quit"})
                break
            elif msg_type == "get_state":
                await emit(
                    {
                        "type": "live_state",
                        "data": {
                            "session_id": session_id,
                            "playlist": list(engine.playlist),
                            "engine_state": engine.get_state(),
                        },
                    }
                )
            elif msg_type == "set_endless_mode":
                # v2.6.0 — opt-in endless / improvisation mode. Persists
                # on the session so reconnects keep the flag, and flips
                # the live engine so the watchdog (Local) /
                # report_playback_pos (Browser) gate picks it up on the
                # next tick. Server echoes confirmation so the frontend
                # state mirror matches reality.
                enabled = bool(msg.get("enabled", False))
                s.context_variables["endless_mode"] = enabled
                engine._endless_mode = enabled
                # v2.7.2 — persist the ctx mutation so a backend restart
                # between toggle and the actual end-of-set still has the
                # flag. Without this the in-memory toggle is lost on
                # reload and ``phase_live`` syncs ``False`` to a fresh
                # engine while the frontend pill still reads ON.
                try:
                    store.save(s)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[live-ws {session_id}] set_endless_mode store.save failed: {exc}",
                        flush=True,
                    )
                print(
                    f"[live-ws {session_id}] set_endless_mode enabled={enabled} "
                    f"(persisted to session ctx)",
                    flush=True,
                )
                await emit({"type": "endless_mode", "enabled": enabled})

    except WebSocketDisconnect:
        print(f"[live-ws {session_id}] WebSocketDisconnect raised", flush=True)
    finally:
        print(
            f"[live-ws {session_id}] entering finally — calling engine.stop()",
            flush=True,
        )
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass
        if not phase_task.done():
            phase_task.cancel()
            try:
                await phase_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if not stall_task.done():
            stall_task.cancel()
            try:
                await stall_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # v2.7.2: detach from the session-scoped YT runtime. If we were
        # the last subscriber, the runtime schedules a 30-s grace
        # teardown — a Chrome refresh or OBS Browser Source reconnect
        # within that window re-attaches without re-discovering the
        # broadcast or losing the poller's state.
        if yt_subscription is not None:
            try:
                await yt_subscription.detach()
            except Exception as exc:  # noqa: BLE001
                print(f"[live-ws {session_id}] yt detach failed: {exc}", flush=True)
        # v2.7.2: drop the engine-event bus so a fresh primary on the
        # same session starts with no stale cache. Viewers still attached
        # will see their next publish silently no-op; their handlers
        # exit when the user navigates away.
        try:
            await live_runtime.drop_bus(user["id"], session_id)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[live-ws {session_id}] live_runtime.drop_bus failed: {exc}",
                flush=True,
            )
        ws_manager.disconnect(session_id, channel="live")


@app.websocket("/api/sessions/{session_id}/live/viewer")
async def live_viewer_ws(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
):
    """Read-only viewer WS for OBS Browser Source / embed pages (v2.7.2).

    Subscribes to the session's engine pub/sub via
    :mod:`live_runtime`. Receives every event the primary live WS emits
    (``live_state``, ``engine_command``, ``track_started`` …) and
    forwards them verbatim. Does NOT instantiate a ``LiveEngineBrowser``
    and does NOT touch :mod:`ws_manager` — multiple viewers can attach
    to the same session without contending with the primary or with
    each other.

    Auth + ownership: same checks as the primary live WS; the viewer is
    bound to the session's owner so a shared embed URL still requires
    the operator's JWT (carried as ``?token=`` for OBS Browser Source).
    """
    user = auth.user_from_query_token(token)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    s = store.get(session_id)
    if not s or s.user_id != user["id"]:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    from . import live_runtime  # noqa: PLC0415 — match primary handler.

    async def on_event(event: dict) -> None:
        try:
            await websocket.send_json(event)
        except Exception:
            # Caller will catch the WS close on the next iteration of
            # the read loop below; nothing actionable here.
            pass

    subscription = await live_runtime.subscribe_viewer(
        user_id=user["id"], session_id=session_id, on_event=on_event,
    )

    try:
        # Viewers don't drive the engine, but we still need a read loop
        # so the handler stays alive (and we notice when the client
        # disconnects). Any frames the client sends are ignored — they
        # have no path to the agent in viewer mode by design.
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[live-viewer {session_id}] receive error: {exc}",
                    flush=True,
                )
                break
    finally:
        try:
            await subscription.detach()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[live-viewer {session_id}] detach failed: {exc}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# WS message dispatcher (separated so the outer loop can catch phase failures
# and emit a graceful `error` event instead of dropping the connection)
# ---------------------------------------------------------------------------

async def _handle_ws_message(s, msg_type: str | None, content: str, emit) -> None:
    # Inject session identity into context_variables so agent tools can
    # consult per-user data (playlists, ratings) via SQLite. Idempotent —
    # only populated on the first ws message; subsequent messages reuse
    # the cached values, and a defensive `.get()` everywhere downstream
    # means previous phases keep working when these keys are absent.
    if "user_id" not in s.context_variables:
        s.context_variables["user_id"] = s.user_id
        user_row = db.get_user_by_id(s.user_id)
        if user_row:
            s.context_variables["username"] = user_row["username"]

    # ── Genre Guard ──────────────────────────────────────────────
    if msg_type == "genre_intent":
        s.phase = "genre"
        history = s.messages.setdefault("genre", [])
        try:
            await asyncio.to_thread(pipeline.check_catalog)
        except pipeline.CatalogUnavailable as exc:
            await emit({"type": "error", "message": str(exc)})
            s.phase = "init"
            return
        confirmed = await pipeline.phase_genre_guard(content, history, s.context_variables, emit)

        if confirmed:
            s.context_variables.update(confirmed)
            # Always emit s.to_dict() for phase_complete so the frontend can
            # safely setSession(event.data) without losing fields like playlist.
            await emit({"type": "phase_complete", "phase": "genre", "data": s.to_dict()})

            # Verify the confirmed genre actually has tracks before invoking
            # the Planner — otherwise the LLM ends up calling get_catalog and
            # producing a vague "issue accessing the catalog" message.
            try:
                await asyncio.to_thread(pipeline.check_catalog, confirmed["genre"])
            except pipeline.CatalogUnavailable as exc:
                await emit({"type": "error", "message": str(exc)})
                s.phase = "init"
                return

            # ── Planner (auto-starts after genre confirmation) ──
            s.phase = "planning"
            await emit({"type": "phase_start", "phase": "planning"})
            memory = await pipeline.load_memory(confirmed["genre"], s.context_variables)
            await pipeline.phase_plan(s.context_variables, emit, memory)

            s.phase = "checkpoint1"
            await emit({"type": "phase_complete", "phase": "planning", "data": s.to_dict()})
        else:
            # Distinguish "agent is still asking" from "agent gave up". On
            # in-progress turns the LLM is politely asking "Is this correct?"
            # and the user just needs to reply — emitting an error banner
            # next to that question is purely cosmetic noise. See issue #23.
            if _should_emit_genre_error(history):
                await emit({"type": "error", "message": "Could not confirm genre — please start a new session."})
                s.phase = "init"
            # Otherwise, the agent is still asking a question — keep
            # s.phase = "genre" so the user's next message routes through
            # this handler again. No error event.

    # ── Checkpoint 1 — user approves playlist → run Critic ──────
    elif msg_type == "checkpoint_approve" and s.phase == "checkpoint1":
        try:
            await asyncio.to_thread(pipeline.check_catalog, s.context_variables.get("genre"))
        except pipeline.CatalogUnavailable as exc:
            await emit({"type": "error", "message": str(exc)})
            return
        s.phase = "critique"
        await emit({"type": "phase_start", "phase": "critique"})
        memory = await pipeline.load_memory(s.context_variables.get("genre", ""), s.context_variables)
        verdict, problems, structured = await pipeline.phase_critique(s.context_variables, emit, memory)

        s.critic_verdict = verdict
        s.critic_problems = problems
        s.structured_problems = structured
        # v2.6.0 — populate set_health so Editor + Curate can render it
        # without re-running their own client-side formula.
        s.set_health = pipeline.compute_set_health(structured)
        s.phase = "checkpoint2"
        await emit({"type": "phase_complete", "phase": "critique", "data": s.to_dict()})

    # ── Checkpoint 2 — user proceeds to Editor ───────────────────
    elif msg_type == "checkpoint2_approve" and s.phase == "checkpoint2":
        s.phase = "editing"
        s.messages.setdefault("editor", [])
        await emit({"type": "phase_start", "phase": "editing"})
        await emit({"type": "phase_complete", "phase": "checkpoint2", "data": s.to_dict()})

    # ── Editor command ────────────────────────────────────────────
    elif msg_type == "editor_command" and s.phase == "editing":
        try:
            await asyncio.to_thread(pipeline.check_catalog, s.context_variables.get("genre"))
        except pipeline.CatalogUnavailable as exc:
            await emit({"type": "error", "message": str(exc)})
            return
        history = s.messages.setdefault("editor", [])
        await pipeline.phase_editor(content, history, s.context_variables, emit)

        last_build = s.context_variables.get("last_build")
        if last_build:
            s.session_name = last_build
            s.phase = "validating"
            await emit({"type": "phase_start", "phase": "validating"})
            v_status, v_issues = await pipeline.phase_validate(last_build, s.context_variables, emit)
            s.validator_status = v_status
            s.validator_issues = v_issues
            s.phase = "rating"
            await emit({"type": "phase_complete", "phase": "validating", "data": s.to_dict()})
        else:
            await emit({"type": "phase_complete", "phase": "editor_turn", "data": s.to_dict()})

    # ── State sync ────────────────────────────────────────────────
    elif msg_type == "get_state":
        await emit({"type": "state", "data": s.to_dict()})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _own(session_id: str, user: dict):
    s = store.get(session_id)
    if not s or s.user_id != user["id"]:
        raise HTTPException(status_code=404, detail="Session not found")
    return s
