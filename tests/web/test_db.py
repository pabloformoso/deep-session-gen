"""Unit tests for web/backend/db.py SQLite user store."""
from __future__ import annotations

import sqlite3

import pytest

from web.backend import db


def test_init_db_is_idempotent(tmp_db):
    # Calling init_db twice must not raise
    db.init_db()
    db.init_db()


def test_create_user_returns_valid_int(tmp_db):
    """Regression: lastrowid must not be None; user_id must be a positive int."""
    uid = db.create_user("alice", "a@t.io", "hashed")
    assert isinstance(uid, int)
    assert uid > 0


def test_create_user_returns_incrementing_ids(tmp_db):
    uid1 = db.create_user("alice", "a@t.io", "h")
    uid2 = db.create_user("bob", "b@t.io", "h")
    assert uid2 > uid1


def test_create_user_duplicate_username_raises(tmp_db):
    db.create_user("alice", "a@t.io", "h")
    with pytest.raises(sqlite3.IntegrityError):
        db.create_user("alice", "other@t.io", "h")


def test_get_user_by_username(tmp_db):
    db.create_user("alice", "a@t.io", "h")
    user = db.get_user_by_username("alice")
    assert user is not None
    assert user["username"] == "alice"
    assert user["email"] == "a@t.io"
    assert db.get_user_by_username("nobody") is None


def test_get_user_by_id(tmp_db):
    uid = db.create_user("alice", "a@t.io", "h")
    user = db.get_user_by_id(uid)
    assert user is not None
    assert user["id"] == uid
    assert db.get_user_by_id(99999) is None
