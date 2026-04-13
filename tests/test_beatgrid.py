"""
Tests for beatgrid auto-generation.

Covers:
  - detect_beatgrid() — output shape, types, edge cases (silent, no hint, empty beats)
  - build_catalog() — backfill of existing entries missing beatgrid; idempotency
  - fix_incomplete_catalog() — beatgrid counted as a missing field
  - generate_beatgrid_catalog() — genre filter, idempotency, missing file skip
  - generate_beatgrid tool — subprocess args
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

import main as main_module
from main import detect_beatgrid, fix_incomplete_catalog, generate_beatgrid_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent_wav(duration_sec: float = 1.0, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Return (silent mono audio, sample_rate)."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32), sr


def _beat_wav(bpm: float = 128.0, duration_sec: float = 4.0, sr: int = 22050):
    """Return (audio with impulse beats, sample_rate, expected_first_beat_sec)."""
    n = int(duration_sec * sr)
    y = np.zeros(n, dtype=np.float32)
    beat_interval = int(sr * 60.0 / bpm)
    first_beat = int(0.1 * sr)  # 100 ms offset
    for i in range(first_beat, n, beat_interval):
        y[i] = 1.0  # click
    return y, sr, first_beat / sr


def _make_catalog_file(tracks: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="catalog_"
    )
    json.dump({"tracks": tracks}, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _fake_wav_file() -> Path:
    """Create a tiny real WAV (silence) that soundfile/wave can open."""
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, np.zeros(22050, dtype=np.float32), 22050)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# detect_beatgrid()
# ---------------------------------------------------------------------------

class TestDetectBeatgrid:

    def _call(self, y, sr, bpm=None):
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track") as mock_bt, \
             patch("main.librosa.frames_to_time") as mock_ft:
            # Simulate beat_track returning (tempo, frames)
            tempo = bpm or 128.0
            beat_frames = np.array([int(sr * 0.1), int(sr * 0.1 + sr * 60 / tempo)])
            mock_bt.return_value = (np.array(tempo), beat_frames)
            # frames_to_time converts frames → seconds
            beat_times = beat_frames / sr
            mock_ft.return_value = beat_times
            return detect_beatgrid("fake.wav", bpm)

    def test_returns_dict_with_required_keys(self):
        y, sr = _silent_wav()
        result = self._call(y, sr, bpm=128.0)
        assert "bpm" in result
        assert "first_beat_sec" in result

    def test_values_are_python_floats_not_numpy(self):
        """Ensure JSON-serialisable types (not numpy scalars)."""
        y, sr = _silent_wav()
        result = self._call(y, sr, bpm=128.0)
        assert isinstance(result["bpm"], float)
        assert isinstance(result["first_beat_sec"], float)

    def test_first_beat_sec_rounded_to_3dp(self):
        y, sr = _silent_wav()
        result = self._call(y, sr, bpm=128.0)
        # Should have at most 3 decimal places
        assert result["first_beat_sec"] == round(result["first_beat_sec"], 3)

    def test_bpm_rounded_to_3dp(self):
        y, sr = _silent_wav()
        result = self._call(y, sr, bpm=128.0)
        assert result["bpm"] == round(result["bpm"], 3)

    def test_uses_bpm_hint_as_start_bpm(self):
        """beat_track() must receive the bpm hint so phase is anchored correctly."""
        y, sr = _silent_wav()
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track") as mock_bt, \
             patch("main.librosa.frames_to_time", return_value=np.array([0.1])):
            mock_bt.return_value = (np.array(130.0), np.array([2205]))
            detect_beatgrid("fake.wav", bpm=130.0)
            _, kwargs = mock_bt.call_args
            assert kwargs.get("start_bpm") == 130.0

    def test_no_bpm_hint_defaults_to_120(self):
        """When bpm=None, start_bpm should default to 120.0."""
        y, sr = _silent_wav()
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track") as mock_bt, \
             patch("main.librosa.frames_to_time", return_value=np.array([0.0])):
            mock_bt.return_value = (np.array(120.0), np.array([0]))
            detect_beatgrid("fake.wav", bpm=None)
            _, kwargs = mock_bt.call_args
            assert kwargs.get("start_bpm") == 120.0

    def test_empty_beat_frames_returns_zero(self):
        """If librosa finds no beats, first_beat_sec must be 0.0 — not an error."""
        y, sr = _silent_wav()
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track", return_value=(np.array(120.0), np.array([]))), \
             patch("main.librosa.frames_to_time", return_value=np.array([])):
            result = detect_beatgrid("fake.wav", bpm=120.0)
        assert result["first_beat_sec"] == 0.0

    def test_silent_audio_does_not_raise(self):
        """Silent audio produces no beats — should return gracefully."""
        y, sr = _silent_wav(duration_sec=2.0)
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track", return_value=(np.array(120.0), np.array([]))), \
             patch("main.librosa.frames_to_time", return_value=np.array([])):
            result = detect_beatgrid("silent.wav")
        assert result["first_beat_sec"] == 0.0
        assert result["bpm"] > 0

    def test_first_beat_is_positive(self):
        """first_beat_sec should be >= 0."""
        y, sr = _silent_wav()
        result = self._call(y, sr, bpm=128.0)
        assert result["first_beat_sec"] >= 0.0

    def test_bpm_consistent_with_hint(self):
        """Returned bpm should equal the beat_track tempo output (not the hint)."""
        y, sr = _silent_wav()
        # beat_track returns 130, even though we hint 128
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track", return_value=(np.array(130.0), np.array([100]))), \
             patch("main.librosa.frames_to_time", return_value=np.array([100 / sr])):
            result = detect_beatgrid("fake.wav", bpm=128.0)
        assert result["bpm"] == 130.0


# ---------------------------------------------------------------------------
# build_catalog() — beatgrid backfill
# ---------------------------------------------------------------------------

class TestBuildCatalogBeatgridBackfill:
    """Test that build_catalog() backfills beatgrid for existing entries."""

    def _run_build(self, catalog_tracks: list[dict], wav_files: list[tuple]):
        """
        Run build_catalog() with:
          - a temp catalog containing catalog_tracks
          - mocked scan_genre_folders returning wav_files [(genre_folder, wav_path)]
          - mocked audio analysis functions
        """
        cat_file = _make_catalog_file(catalog_tracks)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.TRACKS_BASE_DIR", str(cat_file.parent)), \
                 patch("main.scan_genre_folders", return_value=wav_files), \
                 patch("main.detect_bpm", return_value=128.0), \
                 patch("main.detect_beatgrid", return_value={"bpm": 128.0, "first_beat_sec": 0.021}), \
                 patch("main.detect_camelot_key", return_value="8A"), \
                 patch("main._wav_duration_sec", return_value=240.0), \
                 patch("main.load_existing_session_jsons", return_value={}), \
                 patch("main._make_track_id", return_value="genre--track"):
                main_module.build_catalog()

            with open(cat_file) as f:
                return json.load(f)["tracks"]
        finally:
            cat_file.unlink(missing_ok=True)

    def test_existing_entry_without_beatgrid_gets_backfilled(self):
        existing = [{
            "id": "t1", "display_name": "Track", "file": "tracks/techno/track.wav",
            "genre_folder": "techno", "genre": "techno",
            "bpm": 128.0, "camelot_key": "8A", "duration_sec": 240.0,
            # no beatgrid
        }]
        tracks = self._run_build(
            existing,
            [("techno", str(Path("tracks/techno/track.wav").absolute()))]
        )
        entry = next(t for t in tracks if t["id"] == "t1")
        assert "beatgrid" in entry
        assert entry["beatgrid"]["first_beat_sec"] == 0.021

    def test_existing_entry_with_beatgrid_not_reprocessed(self):
        """detect_beatgrid must NOT be called for entries that already have it."""
        existing = [{
            "id": "t1", "display_name": "Track", "file": "tracks/techno/track.wav",
            "genre_folder": "techno", "genre": "techno",
            "bpm": 128.0, "camelot_key": "8A", "duration_sec": 240.0,
            "beatgrid": {"bpm": 128.0, "first_beat_sec": 0.5},
        }]
        cat_file = _make_catalog_file(existing)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.TRACKS_BASE_DIR", str(cat_file.parent)), \
                 patch("main.scan_genre_folders", return_value=[("techno", "tracks/techno/track.wav")]), \
                 patch("main.detect_beatgrid") as mock_bg, \
                 patch("main._wav_duration_sec", return_value=240.0):
                main_module.build_catalog()
                mock_bg.assert_not_called()

            with open(cat_file) as f:
                tracks = json.load(f)["tracks"]
            entry = next(t for t in tracks if t["id"] == "t1")
            # Original beatgrid preserved
            assert entry["beatgrid"]["first_beat_sec"] == 0.5
        finally:
            cat_file.unlink(missing_ok=True)

    def test_new_entry_gets_beatgrid(self):
        tracks = self._run_build(
            [],  # empty catalog
            [("techno", str(Path("/tmp/new_track.wav")))]
        )
        assert len(tracks) == 1
        assert "beatgrid" in tracks[0]
        assert tracks[0]["beatgrid"]["first_beat_sec"] == 0.021

    def test_entry_missing_both_duration_and_beatgrid_gets_both(self):
        existing = [{
            "id": "t1", "display_name": "Track", "file": "tracks/techno/track.wav",
            "genre_folder": "techno", "genre": "techno",
            "bpm": 128.0, "camelot_key": "8A",
            # missing duration_sec AND beatgrid
        }]
        tracks = self._run_build(
            existing,
            [("techno", str(Path("tracks/techno/track.wav").absolute()))]
        )
        entry = next(t for t in tracks if t["id"] == "t1")
        assert entry.get("duration_sec") == 240.0
        assert "beatgrid" in entry


# ---------------------------------------------------------------------------
# fix_incomplete_catalog() — beatgrid as missing field
# ---------------------------------------------------------------------------

class TestFixIncompleteBeatgrid:

    def _run_fix(self, catalog_tracks: list[dict], wav_exists: bool = True):
        cat_file = _make_catalog_file(catalog_tracks)
        fake_wav = _fake_wav_file()
        try:
            # Point all file paths to a real file (or not)
            for t in catalog_tracks:
                t["file"] = str(fake_wav) if wav_exists else "/nonexistent/track.wav"

            cat_file.write_text(json.dumps({"tracks": catalog_tracks}))

            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_bpm", return_value=128.0), \
                 patch("main.detect_beatgrid", return_value={"bpm": 128.0, "first_beat_sec": 0.01}), \
                 patch("main.detect_camelot_key", return_value="8A"), \
                 patch("main._wav_duration_sec", return_value=240.0), \
                 patch("main._SCRIPT_DIR", ""):
                fix_incomplete_catalog()

            with open(cat_file) as f:
                return json.load(f)["tracks"]
        finally:
            cat_file.unlink(missing_ok=True)
            fake_wav.unlink(missing_ok=True)

    def test_entry_missing_beatgrid_gets_fixed(self):
        tracks = self._run_fix([{
            "id": "t1", "display_name": "T", "file": "fake.wav",
            "genre_folder": "techno", "genre": "techno",
            "bpm": 128.0, "camelot_key": "8A", "duration_sec": 240.0,
            # no beatgrid
        }])
        assert "beatgrid" in tracks[0]

    def test_entry_with_beatgrid_not_in_incomplete_list(self):
        """Entry with all fields including beatgrid should not appear as incomplete."""
        entry = {
            "id": "t1", "display_name": "T", "file": "fake.wav",
            "genre_folder": "techno", "genre": "techno",
            "bpm": 128.0, "camelot_key": "8A", "duration_sec": 240.0,
            "beatgrid": {"bpm": 128.0, "first_beat_sec": 0.1},
        }
        cat_file = _make_catalog_file([entry])
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid") as mock_bg:
                fix_incomplete_catalog()
                mock_bg.assert_not_called()
        finally:
            cat_file.unlink(missing_ok=True)

    def test_entry_missing_bpm_skips_beatgrid(self):
        """Can't generate beatgrid without bpm — should not crash."""
        tracks = self._run_fix([{
            "id": "t1", "display_name": "T", "file": "fake.wav",
            "genre_folder": "techno", "genre": "techno",
            "camelot_key": "8A", "duration_sec": 240.0,
            # no bpm, no beatgrid
        }])
        # BPM gets detected (detect_bpm mocked to 128), then beatgrid should also run
        # Because fix_incomplete calls detect_bpm first, then beatgrid is no longer blocked
        entry = tracks[0]
        assert entry.get("bpm") == 128.0


# ---------------------------------------------------------------------------
# generate_beatgrid_catalog() — standalone function
# ---------------------------------------------------------------------------

class TestGenerateBeatgridCatalog:

    def _run(self, catalog_tracks: list[dict], genre_filter=None, wav_exists=True):
        cat_file = _make_catalog_file(catalog_tracks)
        fake_wav = _fake_wav_file() if wav_exists else None
        try:
            if wav_exists and fake_wav:
                for t in catalog_tracks:
                    t["file"] = str(fake_wav)
                cat_file.write_text(json.dumps({"tracks": catalog_tracks}))

            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid", return_value={"bpm": 128.0, "first_beat_sec": 0.021}), \
                 patch("main._SCRIPT_DIR", ""), \
                 patch("os.path.exists", return_value=wav_exists):
                generate_beatgrid_catalog(genre_filter=genre_filter)

            with open(cat_file) as f:
                return json.load(f)["tracks"]
        finally:
            cat_file.unlink(missing_ok=True)
            if fake_wav:
                fake_wav.unlink(missing_ok=True)

    def test_missing_beatgrid_gets_generated(self):
        tracks = self._run([{
            "id": "t1", "display_name": "T", "file": "t.wav",
            "genre_folder": "techno", "genre": "techno", "bpm": 128.0,
        }])
        assert tracks[0].get("beatgrid") is not None

    def test_existing_beatgrid_not_overwritten(self):
        """Idempotent — entries already having beatgrid must be skipped."""
        original_bg = {"bpm": 130.0, "first_beat_sec": 0.5}
        cat = [{
            "id": "t1", "display_name": "T", "file": "t.wav",
            "genre_folder": "techno", "genre": "techno", "bpm": 128.0,
            "beatgrid": original_bg,
        }]
        cat_file = _make_catalog_file(cat)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid") as mock_bg:
                generate_beatgrid_catalog()
                mock_bg.assert_not_called()
        finally:
            cat_file.unlink(missing_ok=True)

    def test_genre_filter_only_processes_matching_genre(self):
        cat = [
            {"id": "t1", "display_name": "A", "file": "a.wav",
             "genre_folder": "techno", "bpm": 128.0},
            {"id": "t2", "display_name": "B", "file": "b.wav",
             "genre_folder": "lofi - ambient", "bpm": 80.0},
        ]
        cat_file = _make_catalog_file(cat)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid", return_value={"bpm": 128.0, "first_beat_sec": 0.01}) as mock_bg, \
                 patch("os.path.exists", return_value=True), \
                 patch("main._SCRIPT_DIR", ""):
                generate_beatgrid_catalog(genre_filter="techno")
                # Only called once — for techno track
                assert mock_bg.call_count == 1
        finally:
            cat_file.unlink(missing_ok=True)

    def test_missing_wav_file_skipped_gracefully(self):
        cat = [{
            "id": "t1", "display_name": "T", "file": "/nonexistent/track.wav",
            "genre_folder": "techno", "bpm": 128.0,
        }]
        cat_file = _make_catalog_file(cat)
        try:
            # Return True for CATALOG_PATH (temp file exists) but False for the WAV.
            def _exists(path):
                return str(path) == str(cat_file)

            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid") as mock_bg, \
                 patch("main.os.path.exists", side_effect=_exists), \
                 patch("main._SCRIPT_DIR", ""):
                generate_beatgrid_catalog()  # must not raise
                mock_bg.assert_not_called()
        finally:
            cat_file.unlink(missing_ok=True)

    def test_entry_without_bpm_skipped(self):
        """Can't generate beatgrid without a BPM value."""
        cat = [{"id": "t1", "display_name": "T", "file": "t.wav",
                "genre_folder": "techno"}]  # no bpm
        cat_file = _make_catalog_file(cat)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)), \
                 patch("main.detect_beatgrid") as mock_bg, \
                 patch("os.path.exists", return_value=True), \
                 patch("main._SCRIPT_DIR", ""):
                generate_beatgrid_catalog()
                mock_bg.assert_not_called()
        finally:
            cat_file.unlink(missing_ok=True)

    def test_no_pending_entries_exits_early(self, capsys):
        """If all tracks have beatgrid, prints 'nothing to do' message."""
        cat = [{
            "id": "t1", "display_name": "T", "file": "t.wav",
            "genre_folder": "techno", "bpm": 128.0,
            "beatgrid": {"bpm": 128.0, "first_beat_sec": 0.1},
        }]
        cat_file = _make_catalog_file(cat)
        try:
            with patch("main.CATALOG_PATH", str(cat_file)):
                generate_beatgrid_catalog()
            out = capsys.readouterr().out
            assert "Nothing to do" in out or "nothing to do" in out.lower()
        finally:
            cat_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# generate_beatgrid tool (agent/tools.py)
# ---------------------------------------------------------------------------

class TestGenerateBeatgridTool:
    def _call(self, genre, returncode=0):
        import agent.tools as tools_module
        with patch("agent.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=returncode)
            result = tools_module.generate_beatgrid(genre, context_variables={})
        return result, mock_run

    def test_all_genres_no_genre_flag(self):
        _, mock_run = self._call("all")
        cmd = mock_run.call_args[0][0]
        assert "--generate-beatgrid" in cmd
        assert "--genre" not in cmd

    def test_specific_genre_passes_genre_flag(self):
        _, mock_run = self._call("techno")
        cmd = mock_run.call_args[0][0]
        assert "--generate-beatgrid" in cmd
        assert "--genre" in cmd
        assert "techno" in cmd

    def test_success_message_on_returncode_0(self):
        result, _ = self._call("all", returncode=0)
        assert "Beatgrid generated" in result

    def test_failure_message_on_nonzero_returncode(self):
        result, _ = self._call("techno", returncode=1)
        assert "failed" in result.lower()

    def test_case_insensitive_all(self):
        """'ALL' and 'All' should behave the same as 'all'."""
        for variant in ("ALL", "All"):
            _, mock_run = self._call(variant)
            cmd = mock_run.call_args[0][0]
            assert "--genre" not in cmd
