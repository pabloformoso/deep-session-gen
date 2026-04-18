"""WebSocket connection manager — one connection per active session."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import WebSocket


class WSManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[session_id] = ws

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    async def send(self, session_id: str, data: dict) -> None:
        ws = self._connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(session_id)

    async def receive(self, session_id: str) -> Optional[dict]:
        ws = self._connections.get(session_id)
        if not ws:
            return None
        try:
            text = await ws.receive_text()
            return json.loads(text)
        except Exception:
            return None

    def is_connected(self, session_id: str) -> bool:
        return session_id in self._connections


ws_manager = WSManager()
