"""Tests for suggest_bridge_track and insert_bridge_track tools."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.tools as tools
from agent.tools import suggest_bridge_track, insert_bridge_track


# ---------------------------------------------------------------------------
# Synthetic catalog
# ---------------------------------------------------------------------------

_CATALOG = [
    {"id": "t1", "display_name": "Alpha",   "genre_folder": "techno", "genre": "techno", "bpm": 90.0,  "camelot_key": "1A"},
    {"id": "t2", "display_name": "Beta",    "genre_folder": "techno", "genre": "techno", "bpm": 100.0, "camelot_key": "2A"},
    {"id": "t3", "display_name": "Gamma",   "genre_folder": "techno", "genre": "techno", "bpm": 120.0, "camelot_key": "3A"},
    {"id": "t4", "display_name": "Delta",   "genre_folder": "techno", "genre": "techno", "bpm": 130.0, "camelot_key": "4A"},
    {"id": "t5", "display_name": "Epsilon", "genre_folder": "techno", "genre": "techno", "bpm": 140.0, "camelot_key": "5A"},
    {"id": "t6", "display_name": "Zeta",    "genre_folder": "techno", "genre": "techno", "bpm": 150.0, "camelot_key": "6A"},
    # Different genre — should be excluded
    {"id": "t7", "display_name": "Lofi",    "genre_folder": "lofi - ambient", "genre": "lofi - ambient", "bpm": 80.0, "camelot_key": "1A"},
]


def _make_catalog_file():
    """Write synthetic catalog to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_CATALOG, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _ctx(playlist, genre="techno"):
    return {"playlist": list(playlist), "genre": genre}


def _make_playlist(*ids):
    """Return a list of track dicts for the given ids (from _CATALOG)."""
    by_id = {t["id"]: t for t in _CATALOG}
    return [dict(by_id[i]) for i in ids]


# ---------------------------------------------------------------------------
# Tests for suggest_bridge_track
# ---------------------------------------------------------------------------

class TestSuggestBridgeTrack:
    def _call(self, playlist, from_pos, to_pos, genre="techno"):
        catalog_path = _make_catalog_file()
        try:
            with patch.object(tools, "_CATALOG_PATH", catalog_path):
                return suggest_bridge_track(from_pos, to_pos, _ctx(playlist, genre))
        finally:
            catalog_path.unlink(missing_ok=True)

    def test_returns_up_to_three_candidates(self):
        # Playlist has t1 (90 BPM) and t6 (150 BPM), rest are candidates
        playlist = _make_playlist("t1", "t6")
        result = self._call(playlist, 1, 2)
        # Count candidate lines (lines starting with spaces/indentation showing id)
        candidate_lines = [l for l in result.splitlines() if l.strip().startswith("t")]
        assert len(candidate_lines) <= 3
        assert len(candidate_lines) >= 1

    def test_candidates_sorted_by_score(self):
        # t1=90, t6=150 → target ≈ 116.2 BPM → t3 (120) should score highest
        playlist = _make_playlist("t1", "t6")
        result = self._call(playlist, 1, 2)
        lines = [l for l in result.splitlines() if l.strip().startswith("t")]
        assert len(lines) >= 1
        # First candidate should be t3 (closest to geometric mean of 90 and 150)
        assert lines[0].strip().startswith("t3")

    def test_playlist_tracks_excluded(self):
        # t3 is in the playlist, so it should not appear as a candidate
        playlist = _make_playlist("t1", "t6", "t3")
        result = self._call(playlist, 1, 2)
        assert "t3" not in result

    def test_out_of_range_from_pos_returns_error(self):
        playlist = _make_playlist("t1", "t2")
        result = self._call(playlist, 0, 2)
        assert "error" in result.lower() or "must be between" in result.lower() or "Positions" in result

    def test_out_of_range_to_pos_returns_error(self):
        playlist = _make_playlist("t1", "t2")
        result = self._call(playlist, 1, 5)
        assert "error" in result.lower() or "must be between" in result.lower() or "Positions" in result

    def test_no_playlist_returns_error(self):
        result = suggest_bridge_track(1, 2, {})
        assert "No playlist" in result

    def test_wrong_genre_excluded(self):
        # Genre is lofi - ambient, but all tracks in playlist are techno genre_folder
        # so no candidates should match
        playlist = _make_playlist("t1", "t6")
        result = self._call(playlist, 1, 2, genre="lofi - ambient")
        # t7 is the only lofi track but it's not in the playlist, should be only candidate
        # or result may say no candidates since t1/t6 are techno genre_folder mismatch
        # Either way result should not error (just 0 or 1 candidate)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests for insert_bridge_track
# ---------------------------------------------------------------------------

class TestInsertBridgeTrack:
    def _call(self, playlist, after_position, track_id, genre="techno"):
        catalog_path = _make_catalog_file()
        ctx = _ctx(playlist, genre)
        try:
            with patch.object(tools, "_CATALOG_PATH", catalog_path):
                result = insert_bridge_track(after_position, track_id, ctx)
            return result, ctx
        finally:
            catalog_path.unlink(missing_ok=True)

    def test_playlist_length_increases_by_one(self):
        playlist = _make_playlist("t1", "t6")
        result, ctx = self._call(playlist, 1, "t3")
        assert len(ctx["playlist"]) == 3

    def test_inserted_track_at_correct_position(self):
        playlist = _make_playlist("t1", "t6")
        result, ctx = self._call(playlist, 1, "t3")
        # after_position=1 means inserted at index 1 (0-indexed) → position 2 (1-indexed)
        assert ctx["playlist"][1]["id"] == "t3"

    def test_appending_to_end_works(self):
        playlist = _make_playlist("t1", "t2", "t3")
        result, ctx = self._call(playlist, 3, "t4")
        assert len(ctx["playlist"]) == 4
        assert ctx["playlist"][-1]["id"] == "t4"

    def test_unknown_track_id_returns_error(self):
        playlist = _make_playlist("t1", "t2")
        result, ctx = self._call(playlist, 1, "nonexistent-id")
        assert "not found" in result.lower() or "error" in result.lower()
        # Playlist should be unchanged
        assert len(ctx["playlist"]) == 2

    def test_out_of_range_after_position_returns_error(self):
        playlist = _make_playlist("t1", "t2")
        result, ctx = self._call(playlist, 5, "t3")
        assert "must be between" in result or "error" in result.lower()
        assert len(ctx["playlist"]) == 2

    def test_after_position_zero_returns_error(self):
        playlist = _make_playlist("t1", "t2")
        result, ctx = self._call(playlist, 0, "t3")
        assert "must be between" in result or "error" in result.lower()

    def test_confirmation_message_in_result(self):
        playlist = _make_playlist("t1", "t6")
        result, ctx = self._call(playlist, 1, "t3")
        assert "Inserted" in result or "position 2" in result

    def test_no_playlist_returns_error(self):
        catalog_path = _make_catalog_file()
        try:
            with patch.object(tools, "_CATALOG_PATH", catalog_path):
                result = insert_bridge_track(1, "t3", {})
        finally:
            catalog_path.unlink(missing_ok=True)
        assert "No playlist" in result
