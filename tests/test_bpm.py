"""
Tests for detect_bpm() octave ladder.

Covers:
  - Doubled / quartered detections corrected via genre midpoint
  - In-range raw values left untouched (no spurious half-pick)
  - Outliers surfaced as their real value (no silent clamp)
  - Unknown genre passes raw_bpm through
  - librosa.beat_track called WITHOUT start_bpm bias
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from main import detect_bpm


def _patched_detect(raw_bpm: float, genre_folder: str = "") -> float:
    """Run detect_bpm() with librosa mocked to return a fixed raw_bpm."""
    sr = 22050
    y = np.zeros(sr, dtype=np.float32)
    with patch("main.librosa.load", return_value=(y, sr)), \
         patch("main.librosa.beat.beat_track") as mock_bt:
        mock_bt.return_value = (np.array(raw_bpm), np.array([]))
        return detect_bpm("fake.wav", genre_folder)


class TestOctaveLadder:

    def test_doubled_detection_picks_half(self):
        """Lofi range (60, 110): librosa returns 152 → should pick 76."""
        assert _patched_detect(152.0, "lofi - ambient") == 76.0

    def test_quartered_detection_picks_quarter(self):
        """Lofi range (60, 110): librosa returns 304 → should pick 76."""
        assert _patched_detect(304.0, "lofi - ambient") == 76.0

    def test_in_range_value_unchanged(self):
        """Techno range (120, 160): librosa returns 128 → keep 128, don't half-pick to 64."""
        assert _patched_detect(128.0, "techno") == 128.0

    def test_deep_house_in_range(self):
        """Deep house range (115, 135): 124 stays 124."""
        assert _patched_detect(124.0, "deep house") == 124.0

    def test_outlier_no_clamp(self):
        """Techno range (120, 160): librosa returns 174 → keep 174 (not clamped to 160).

        174 is the real BPM and ½×174=87 / 2×174=348 also fall outside the range.
        Previously this got clamped silently to 160; now it must surface as 174.
        """
        assert _patched_detect(174.0, "techno") == 174.0

    def test_outlier_logs_out_of_range(self, capsys):
        _patched_detect(174.0, "techno")
        out = capsys.readouterr().out
        assert "[BPM out-of-range]" in out

    def test_unknown_genre_passthrough(self, capsys):
        """Genre not in BPM_GENRE_RANGES → return raw_bpm without ladder."""
        result = _patched_detect(95.7, "experimental")
        assert result == round(95.7, 1)
        out = capsys.readouterr().out
        assert "[BPM no-genre-range]" in out

    def test_empty_genre_treated_as_unknown(self):
        """Empty genre_folder must not crash — passthrough."""
        result = _patched_detect(123.4, "")
        assert result == round(123.4, 1)

    def test_midpoint_not_used_as_start_bpm(self):
        """beat_track must NOT receive start_bpm — it would bias raw_bpm.

        Regression guard: previously detect_bpm passed start_bpm=midpoint,
        which dragged librosa's output toward the genre center and weakened
        the octave ladder's effectiveness.
        """
        sr = 22050
        y = np.zeros(sr, dtype=np.float32)
        with patch("main.librosa.load", return_value=(y, sr)), \
             patch("main.librosa.beat.beat_track") as mock_bt:
            mock_bt.return_value = (np.array(128.0), np.array([]))
            detect_bpm("fake.wav", "techno")
            _, kwargs = mock_bt.call_args
            assert "start_bpm" not in kwargs or kwargs.get("start_bpm") is None

    def test_octave_correction_logs_change(self, capsys):
        """When the picked octave differs from raw, a [BPM octave] line is printed."""
        _patched_detect(152.0, "lofi - ambient")
        out = capsys.readouterr().out
        assert "[BPM octave]" in out

    def test_no_log_when_already_in_range(self, capsys):
        """In-range raw value: no [BPM octave] log (no change to announce)."""
        _patched_detect(128.0, "techno")
        out = capsys.readouterr().out
        assert "[BPM octave]" not in out
