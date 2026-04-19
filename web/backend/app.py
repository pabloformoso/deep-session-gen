"""
ApolloAgents Web Backend — FastAPI application (v2.0)

Startup:
    cd web && uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from . import db, auth, pipeline
from .models import (
    EditorCommandRequest,
    GenreIntentRequest,
    LoginRequest,
    RatingRequest,
    RegisterRequest,
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

_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
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
# Sessions — REST CRUD
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def create_session(current_user: dict = Depends(auth.get_current_user)):
    s = store.create(current_user["id"])
    return s.to_dict()


@app.get("/api/sessions")
async def list_sessions(current_user: dict = Depends(auth.get_current_user)):
    return [s.to_dict() for s in store.get_user_sessions(current_user["id"])]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, current_user: dict = Depends(auth.get_current_user)):
    return _own(session_id, current_user).to_dict()


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
    payload = auth.decode_token(token)
    if not payload:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    user = db.get_user_by_id(int(payload.get("sub", 0)))
    if not user:
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

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id)


# ---------------------------------------------------------------------------
# WS message dispatcher (separated so the outer loop can catch phase failures
# and emit a graceful `error` event instead of dropping the connection)
# ---------------------------------------------------------------------------

async def _handle_ws_message(s, msg_type: str | None, content: str, emit) -> None:
    # ── Genre Guard ──────────────────────────────────────────────
    if msg_type == "genre_intent":
        s.phase = "genre"
        history = s.messages.setdefault("genre", [])
        confirmed = await pipeline.phase_genre_guard(content, history, s.context_variables, emit)

        if confirmed:
            s.context_variables.update(confirmed)
            # Always emit s.to_dict() for phase_complete so the frontend can
            # safely setSession(event.data) without losing fields like playlist.
            await emit({"type": "phase_complete", "phase": "genre", "data": s.to_dict()})

            # ── Planner (auto-starts after genre confirmation) ──
            s.phase = "planning"
            await emit({"type": "phase_start", "phase": "planning"})
            memory = await pipeline.load_memory(confirmed["genre"], s.context_variables)
            await pipeline.phase_plan(s.context_variables, emit, memory)

            s.phase = "checkpoint1"
            await emit({"type": "phase_complete", "phase": "planning", "data": s.to_dict()})
        else:
            await emit({"type": "error", "message": "Could not confirm genre — please try again."})
            s.phase = "init"

    # ── Checkpoint 1 — user approves playlist → run Critic ──────
    elif msg_type == "checkpoint_approve" and s.phase == "checkpoint1":
        s.phase = "critique"
        await emit({"type": "phase_start", "phase": "critique"})
        memory = await pipeline.load_memory(s.context_variables.get("genre", ""), s.context_variables)
        verdict, problems, structured = await pipeline.phase_critique(s.context_variables, emit, memory)

        s.critic_verdict = verdict
        s.critic_problems = problems
        s.structured_problems = structured
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
