"""Unit tests for web/backend/ws_manager.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from web.backend.ws_manager import WSManager


@pytest.mark.asyncio
async def test_connect_accepts_and_stores():
    mgr = WSManager()
    ws = MagicMock()
    ws.accept = AsyncMock()
    await mgr.connect("sid", ws)
    ws.accept.assert_awaited_once()
    assert mgr.is_connected("sid")


@pytest.mark.asyncio
async def test_send_to_missing_session_is_silent():
    mgr = WSManager()
    await mgr.send("missing", {"type": "noop"})  # must not raise


@pytest.mark.asyncio
async def test_send_forwards_json():
    mgr = WSManager()
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    await mgr.connect("sid", ws)
    await mgr.send("sid", {"type": "hello"})
    ws.send_json.assert_awaited_once_with({"type": "hello"})


@pytest.mark.asyncio
async def test_send_disconnects_on_error():
    mgr = WSManager()
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock(side_effect=RuntimeError("broken"))
    await mgr.connect("sid", ws)
    await mgr.send("sid", {"type": "x"})
    assert not mgr.is_connected("sid")


def test_disconnect_removes_session():
    mgr = WSManager()
    mgr._connections["sid"] = MagicMock()  # type: ignore[assignment]
    mgr.disconnect("sid")
    assert not mgr.is_connected("sid")
    mgr.disconnect("sid")  # idempotent
