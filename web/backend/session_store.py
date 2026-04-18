"""In-memory session state — one Session per active pipeline run."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional


class Session:
    def __init__(self, session_id: str, user_id: int) -> None:
        self.id = session_id
        self.user_id = user_id
        # Phase tracks where we are in the 7-step pipeline
        self.phase: str = "init"
        # LLM message histories keyed by phase
        self.messages: dict[str, list[dict]] = {}
        # Shared mutable state passed through every tool call
        self.context_variables: dict = {}
        # Results from critic and validator phases
        self.critic_verdict: Optional[str] = None
        self.critic_problems: list[str] = []
        self.structured_problems: list[dict] = []
        self.validator_status: Optional[str] = None
        self.validator_issues: list[str] = []
        # Set after build_session succeeds
        self.session_name: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        ctx = self.context_variables
        playlist = ctx.get("playlist", [])
        safe_playlist = [
            {
                "id": t.get("id", ""),
                "display_name": t.get("display_name", ""),
                "bpm": t.get("bpm"),
                "camelot_key": t.get("camelot_key"),
                "duration_sec": t.get("duration_sec"),
                "genre": t.get("genre"),
            }
            for t in playlist
        ]
        return {
            "id": self.id,
            "user_id": self.user_id,
            "phase": self.phase,
            "genre": ctx.get("genre"),
            "duration_min": ctx.get("duration_min"),
            "mood": ctx.get("mood"),
            "playlist": safe_playlist,
            "session_name": self.session_name or ctx.get("last_build"),
            "critic_verdict": self.critic_verdict,
            "critic_problems": self.critic_problems,
            "structured_problems": self.structured_problems,
            "validator_status": self.validator_status,
            "validator_issues": self.validator_issues,
            "created_at": self.created_at,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._by_user: dict[int, list[str]] = {}

    def create(self, user_id: int) -> Session:
        sid = str(uuid.uuid4())
        s = Session(sid, user_id)
        self._sessions[sid] = s
        self._by_user.setdefault(user_id, []).append(sid)
        return s

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_user_sessions(self, user_id: int) -> list[Session]:
        return [self._sessions[i] for i in self._by_user.get(user_id, []) if i in self._sessions]

    def delete(self, session_id: str) -> None:
        s = self._sessions.pop(session_id, None)
        if s:
            ids = self._by_user.get(s.user_id, [])
            if session_id in ids:
                ids.remove(session_id)


store = SessionStore()
