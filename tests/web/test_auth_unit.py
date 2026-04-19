"""Unit tests for web/backend/auth.py primitives."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from jose import jwt

from web.backend import auth


def test_hash_password_is_not_plaintext():
    hashed = auth.hash_password("supersecret")
    assert hashed != "supersecret"
    assert hashed.startswith("$2b$")  # bcrypt identifier


def test_verify_password_round_trip():
    hashed = auth.hash_password("pw12345")
    assert auth.verify_password("pw12345", hashed) is True
    assert auth.verify_password("wrong", hashed) is False


def test_verify_password_handles_invalid_hash():
    # Should not raise — just return False
    assert auth.verify_password("anything", "not-a-bcrypt-hash") is False


def test_hash_handles_unicode_and_long_passwords():
    # bcrypt's 72-byte limit was the actual 500 in the field; cover it
    hashed_long = auth.hash_password("x" * 200)
    assert auth.verify_password("x" * 200, hashed_long) is True
    hashed_unicode = auth.hash_password("héllo-wörld-🎵")
    assert auth.verify_password("héllo-wörld-🎵", hashed_unicode) is True


def test_create_access_token_is_decodable():
    token = auth.create_access_token({"sub": "42"})
    payload = auth.decode_token(token)
    assert payload is not None
    assert payload["sub"] == "42"
    assert "exp" in payload


def test_decode_token_returns_none_on_garbage():
    assert auth.decode_token("garbage") is None
    assert auth.decode_token("") is None


def test_decode_token_returns_none_on_expired():
    expired_payload = {
        "sub": "1",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    token = jwt.encode(expired_payload, auth.SECRET_KEY, algorithm=auth.ALGORITHM)
    assert auth.decode_token(token) is None


@pytest.mark.asyncio
async def test_get_current_user_raises_on_bad_token(tmp_db):
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(token="bogus")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_raises_on_missing_user(tmp_db):
    token = auth.create_access_token({"sub": "99999"})  # no such user
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(token=token)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_returns_user_on_valid_token(tmp_db):
    from web.backend import db

    uid = db.create_user("alice", "a@t.io", auth.hash_password("pw"))
    token = auth.create_access_token({"sub": str(uid)})
    user = await auth.get_current_user(token=token)
    assert user["username"] == "alice"
