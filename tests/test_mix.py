"""
Tests for build_mix() soft-fade branch + change_tempo() crispness handling.

Covers:
  - Soft-fade triggered when stretch_ratio > SOFT_FADE_RATIO_THRESHOLD
  - Meet-in-middle preserved when ratio is within threshold
  - Soft-fade falls back to meet-in-middle when incoming track is too short
  - change_tempo() passes -c crispness arg to pyrubberband
  - Genre-based crispness override (techno → 5, default → 4)
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from pydub import AudioSegment

import main as main_module
from main import (
    SOFT_FADE_CROSSFADE_SEC,
    SOFT_FADE_RATIO_THRESHOLD,
    CROSSFADE_SEC,
    RUBBERBAND_CRISPNESS_DEFAULT,
    _crispness_for_genre,
    build_mix,
    change_tempo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silent_segment(duration_sec: float, sr: int = 22050) -> AudioSegment:
    """Build a real (silent) AudioSegment of the requested duration."""
    return AudioSegment.silent(duration=int(duration_sec * 1000), frame_rate=sr)


def _track(path: str, bpm: float, duration_sec: float, genre: str | None = None,
           camelot_key: str | None = None):
    """Build a track dict matching what analyze_tracks() produces."""
    sr = 22050
    # Synthetic beats every 60/bpm seconds across the duration.
    beat_interval = 60.0 / bpm
    beats = np.arange(0.0, duration_sec, beat_interval)
    return {
        "path": path,
        "display_name": f"Track at {bpm}",
        "bpm": bpm,
        "beats": beats,
        "camelot_key": camelot_key,
        "genre": genre,
        "_duration_sec": duration_sec,  # consumed by the from_file mock
    }


def _run_build_mix(tracks):
    """Run build_mix with audio I/O & DSP mocked.

    AudioSegment.from_file returns a silent segment matching each track's
    declared duration. _normalize_loudness / _apply_bus_limiter / change_speed
    pass through unchanged so the soft-fade branch logic is what we observe.
    """
    path_to_duration = {t["path"]: t["_duration_sec"] for t in tracks}

    def _fake_from_file(path, *_args, **_kwargs):
        return _silent_segment(path_to_duration[path])

    def _passthrough_normalize(segment, _target=None):
        return segment, 0.0

    def _passthrough_limiter(segment):
        return segment

    def _passthrough_change_speed(segment, factor, crispness=None):
        # Don't actually run pyrubberband on silent segments — just return as-is.
        return segment

    with patch("main.AudioSegment.from_file", side_effect=_fake_from_file), \
         patch("main._normalize_loudness", side_effect=_passthrough_normalize), \
         patch("main._apply_bus_limiter", side_effect=_passthrough_limiter), \
         patch("main.change_speed", side_effect=_passthrough_change_speed), \
         patch("main.change_tempo", side_effect=_passthrough_change_speed):
        return build_mix(tracks, target_duration_sec=None)


# ---------------------------------------------------------------------------
# Soft-fade branch
# ---------------------------------------------------------------------------

class TestSoftFadeBranch:

    def test_soft_fade_triggered_when_ratio_exceeds_threshold(self, capsys):
        """76 → 152 BPM: ratio 2.0 > 1.4, must hit soft-fade branch."""
        tracks = [
            _track("a.wav", bpm=76.0, duration_sec=120.0, genre="lofi - ambient"),
            _track("b.wav", bpm=152.0, duration_sec=120.0, genre="lofi - ambient"),
        ]
        _mix, transitions = _run_build_mix(tracks)
        out = capsys.readouterr().out
        assert f"soft-fade @{SOFT_FADE_CROSSFADE_SEC}s" in out
        # Recorded ratio reflects the original BPMs
        assert transitions[1]["stretch_ratio"] == pytest.approx(2.0, rel=0.01)

    def test_meet_in_middle_when_ratio_within_threshold(self, capsys):
        """120 → 130 BPM: ratio ≈ 1.08, must take the classic meet-in-middle path."""
        tracks = [
            _track("a.wav", bpm=120.0, duration_sec=120.0, genre="techno"),
            _track("b.wav", bpm=130.0, duration_sec=120.0, genre="techno"),
        ]
        _run_build_mix(tracks)
        out = capsys.readouterr().out
        assert "meet-in-middle" in out
        assert "soft-fade @" not in out

    def test_soft_fade_falls_back_when_track_too_short(self, capsys):
        """Ratio > 1.4 but incoming duration < SOFT_FADE_CROSSFADE_SEC + CROSSFADE_SEC
        must fall back to meet-in-middle to avoid a corrupt body slice."""
        # 76 → 152 forces soft-fade, but 20s incoming < 24+12=36s required.
        tracks = [
            _track("a.wav", bpm=76.0, duration_sec=120.0, genre="lofi - ambient"),
            _track("b.wav", bpm=152.0, duration_sec=20.0, genre="lofi - ambient"),
        ]
        _run_build_mix(tracks)
        out = capsys.readouterr().out
        assert "[soft-fade skipped]" in out
        assert "meet-in-middle" in out

    def test_stretch_warning_marks_soft_fade(self, capsys):
        """Big-jump transitions still emit [STRETCH WARNING] in soft-fade,
        with a (soft-faded) suffix so the post-mortem log shows the strategy."""
        tracks = [
            _track("a.wav", bpm=76.0, duration_sec=120.0, genre="lofi - ambient"),
            _track("b.wav", bpm=152.0, duration_sec=120.0, genre="lofi - ambient"),
        ]
        _run_build_mix(tracks)
        out = capsys.readouterr().out
        assert "[STRETCH WARNING]" in out
        assert "(soft-faded)" in out


# ---------------------------------------------------------------------------
# change_tempo() crispness
# ---------------------------------------------------------------------------

class TestChangeTempoCrispness:

    def _run_change_tempo(self, factor: float, crispness: int | None = None):
        sr = 22050
        segment = _silent_segment(1.0, sr=sr)
        with patch("main.pyrb.time_stretch") as mock_stretch:
            # Return a stretched-shape array so _numpy_to_segment can rebuild.
            mock_stretch.return_value = np.zeros((int(sr / factor), 1), dtype=np.float32)
            if crispness is None:
                change_tempo(segment, factor)
            else:
                change_tempo(segment, factor, crispness=crispness)
            return mock_stretch

    def test_default_crispness_is_4(self):
        mock = self._run_change_tempo(factor=1.1)
        _, kwargs = mock.call_args
        assert kwargs["rbargs"] == {"-c": "4"}

    def test_explicit_crispness_propagated(self):
        mock = self._run_change_tempo(factor=1.1, crispness=5)
        _, kwargs = mock.call_args
        assert kwargs["rbargs"] == {"-c": "5"}

    def test_no_op_factor_skips_pyrb(self):
        """factor ≈ 1.0: change_tempo returns the segment without invoking pyrb."""
        with patch("main.pyrb.time_stretch") as mock_stretch:
            segment = _silent_segment(1.0)
            result = change_tempo(segment, 1.0)
            mock_stretch.assert_not_called()
            assert result is segment


class TestCrispnessForGenre:

    def test_techno_overrides_to_5(self):
        assert _crispness_for_genre("techno") == 5

    def test_cyberpunk_overrides_to_5(self):
        assert _crispness_for_genre("cyberpunk") == 5

    def test_lofi_uses_default(self):
        assert _crispness_for_genre("lofi - ambient") == RUBBERBAND_CRISPNESS_DEFAULT

    def test_unknown_genre_uses_default(self):
        assert _crispness_for_genre("experimental") == RUBBERBAND_CRISPNESS_DEFAULT

    def test_none_uses_default(self):
        assert _crispness_for_genre(None) == RUBBERBAND_CRISPNESS_DEFAULT

    def test_case_insensitive_lookup(self):
        assert _crispness_for_genre("Techno") == 5
        assert _crispness_for_genre("TECHNO") == 5
