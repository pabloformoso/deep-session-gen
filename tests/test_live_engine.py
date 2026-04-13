"""
Unit tests for agent/live_engine.py.

sounddevice and audio I/O are mocked throughout — no hardware required.
Audio buffers are tiny numpy arrays (1–2 seconds of silence) for speed.
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.live_engine import (
    APPROACHING_CF,
    CROSSFADE_FINISHED,
    CROSSFADE_TRIGGERED,
    SESSION_ENDED,
    TRACK_ENDED,
    TRACK_STARTED,
    LiveEngine,
    _SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_SR = _SAMPLE_RATE  # use real SR so time math works
TRACK_DUR = 3  # very short test tracks (3 s)


def _silent_audio(duration_sec: float = TRACK_DUR) -> np.ndarray:
    """Return a silent stereo float32 array of the given duration."""
    n = int(duration_sec * TINY_SR)
    return np.zeros((n, 2), dtype=np.float32)


def _make_playlist(n: int = 2, bpm: float = 120.0) -> list[dict]:
    return [
        {
            "id": f"track-{i}",
            "display_name": f"Track {i}",
            "file": f"tracks/test/track{i}.wav",
            "bpm": bpm,
            "camelot_key": "8A",
            "duration_sec": float(TRACK_DUR),
        }
        for i in range(1, n + 1)
    ]


@pytest.fixture
def mock_sd():
    """Patch sounddevice so no audio hardware is needed."""
    class _FakeStream:
        def __init__(self, *a, **kw):
            self._cb = kw.get("callback")

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = MagicMock()
    fake_sd.OutputStream.side_effect = _FakeStream

    with patch("agent.live_engine.sd", fake_sd), \
         patch("agent.live_engine._SD_AVAILABLE", True):
        yield fake_sd


@pytest.fixture
def mock_load_audio():
    """Patch LiveEngine._load_audio to return silent numpy arrays."""
    with patch.object(
        LiveEngine, "_load_audio", return_value=_silent_audio()
    ) as m:
        yield m


@pytest.fixture
def mock_prestretch():
    """Patch LiveEngine._time_stretch to be a no-op."""
    with patch.object(
        LiveEngine, "_time_stretch", side_effect=lambda audio, *a, **kw: audio
    ) as m:
        yield m


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def test_engine_initialises_idle(mock_sd, mock_load_audio):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    assert engine.get_state()["state"] == "idle"
    assert engine._audio is None  # no audio loaded until play() is called


# ---------------------------------------------------------------------------
# play() — TRACK_STARTED event
# ---------------------------------------------------------------------------

def test_play_emits_track_started(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()

    ev = q.get(timeout=1)
    assert ev["type"] == TRACK_STARTED
    assert ev["track"]["display_name"] == "Track 1"

    engine.stop()


def test_play_empty_playlist_emits_session_ended(mock_sd, mock_load_audio):
    q = Queue()
    engine = LiveEngine([], q)
    engine.play()

    ev = q.get(timeout=1)
    assert ev["type"] == SESSION_ENDED


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

def test_get_state_after_play(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)  # consume TRACK_STARTED

    state = engine.get_state()
    assert state["state"] == "playing"
    assert state["current_track"]["display_name"] == "Track 1"
    assert state["next_track"]["display_name"] == "Track 2"
    assert state["playlist_remaining"] == 1

    engine.stop()


# ---------------------------------------------------------------------------
# extend_track()
# ---------------------------------------------------------------------------

def test_extend_track_increments_extend_samples(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)  # TRACK_STARTED

    result = engine.extend_track(10)
    assert "10s" in result
    assert engine._extend_samples == 10 * _SAMPLE_RATE

    engine.stop()


def test_extend_track_accumulates(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    engine.extend_track(5)
    engine.extend_track(3)
    assert engine._extend_samples == 8 * _SAMPLE_RATE

    engine.stop()


# ---------------------------------------------------------------------------
# skip_track()
# ---------------------------------------------------------------------------

def test_skip_track_advances_idx(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(3), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    # drain TRACK_STARTED
    q.get(timeout=1)

    result = engine.skip_track()
    assert "Track 2" in result
    assert engine._idx == 1

    engine.stop()


def test_skip_track_on_last_returns_message(mock_sd, mock_load_audio, mock_prestretch):
    playlist = _make_playlist(1)
    q = Queue()
    engine = LiveEngine(playlist, q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    result = engine.skip_track()
    assert "No next track" in result

    engine.stop()


# ---------------------------------------------------------------------------
# queue_swap()
# ---------------------------------------------------------------------------

def test_queue_swap_rejects_past_position(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(3), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    result = engine.queue_swap(1, "some-id")  # position 1 is current
    assert "not a future slot" in result

    engine.stop()


def test_queue_swap_rejects_unknown_track(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(3), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    with patch("agent.live_engine._load_catalog", return_value=[]):
        result = engine.queue_swap(3, "nonexistent-id")
    assert "not found" in result

    engine.stop()


def test_queue_swap_replaces_slot(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(3), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    new_track = {"id": "bridge", "display_name": "Bridge", "file": "t.wav", "bpm": 125.0}
    with patch("agent.live_engine._load_catalog", return_value=[new_track]):
        result = engine.queue_swap(3, "bridge")
    assert "Bridge" in result
    assert engine.playlist[2]["display_name"] == "Bridge"

    engine.stop()


# ---------------------------------------------------------------------------
# crossfade_now() — state guard
# ---------------------------------------------------------------------------

def test_crossfade_now_requires_playing_state(mock_sd, mock_load_audio):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    # engine is idle
    result = engine.crossfade_now()
    assert "Cannot crossfade" in result


def test_crossfade_now_returns_error_on_last_track(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(1), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)

    result = engine.crossfade_now()
    assert "No next track" in result

    engine.stop()


# ---------------------------------------------------------------------------
# _cf_point_samples() — hot cue OUT integration
# ---------------------------------------------------------------------------

def test_cf_point_uses_hot_cue_out():
    q = Queue()
    track = {
        "display_name": "T",
        "file": "t.wav",
        "bpm": 120.0,
        "duration_sec": 60.0,
        "hot_cues": [{"type": "out", "position_sec": 45.0, "label": "OUT"}],
    }
    engine = LiveEngine([track], q)
    engine._audio = _silent_audio(60)
    engine._extend_samples = 0
    samples = engine._cf_point_samples(track)
    assert samples == int(45.0 * _SAMPLE_RATE)


def test_cf_point_defaults_without_hot_cue():
    q = Queue()
    track = {
        "display_name": "T",
        "file": "t.wav",
        "bpm": 120.0,
        "duration_sec": 60.0,
    }
    engine = LiveEngine([track], q, crossfade_sec=12)
    engine._audio = _silent_audio(60)
    engine._extend_samples = 0
    samples = engine._cf_point_samples(track)
    expected = int((60.0 - 12 - 5) * _SAMPLE_RATE)
    assert samples == expected


# ---------------------------------------------------------------------------
# _in_point_of() — hot cue IN
# ---------------------------------------------------------------------------

def test_in_point_uses_hot_cue_in():
    track = {
        "hot_cues": [{"type": "in", "position_sec": 4.2, "label": "IN"}],
    }
    result = LiveEngine._in_point_of(track)
    assert result == int(4.2 * _SAMPLE_RATE)


def test_in_point_defaults_to_zero():
    result = LiveEngine._in_point_of({"hot_cues": []})
    assert result == 0


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

def test_stop_sets_idle(mock_sd, mock_load_audio, mock_prestretch):
    q = Queue()
    engine = LiveEngine(_make_playlist(2), q, crossfade_sec=1, approach_warn_sec=1)
    engine.play()
    q.get(timeout=1)
    engine.stop()
    assert engine._state == "idle"
