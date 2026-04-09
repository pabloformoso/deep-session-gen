"""Tests for the EQ matching at crossfade feature (_apply_crossfade_eq)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydub import AudioSegment
from main import _apply_crossfade_eq


def _make_mono_segment(freq_hz: float, duration_sec: float = 1.0, sample_rate: int = 44100) -> AudioSegment:
    """Create a mono AudioSegment containing a pure sine wave at the given frequency."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = np.sin(2.0 * np.pi * freq_hz * t)
    int_data = (wave * 32767.0).astype(np.int16)
    return AudioSegment(
        data=int_data.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )


class TestApplyCrossfadeEq:
    def test_no_filter_when_distance_zero(self):
        seg = _make_mono_segment(1000.0)
        result = _apply_crossfade_eq(seg, "outgoing", 0)
        assert result.raw_data == seg.raw_data

    def test_outgoing_attenuates_high_freq(self):
        seg = _make_mono_segment(10000.0)
        result = _apply_crossfade_eq(seg, "outgoing", 3)
        input_samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
        output_samples = np.frombuffer(result.raw_data, dtype=np.int16).astype(np.float32)
        input_rms = np.sqrt(np.mean(input_samples ** 2))
        output_rms = np.sqrt(np.mean(output_samples ** 2))
        assert output_rms < input_rms

    def test_incoming_attenuates_low_freq(self):
        seg = _make_mono_segment(60.0)
        result = _apply_crossfade_eq(seg, "incoming", 3)
        input_samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
        output_samples = np.frombuffer(result.raw_data, dtype=np.int16).astype(np.float32)
        input_rms = np.sqrt(np.mean(input_samples ** 2))
        output_rms = np.sqrt(np.mean(output_samples ** 2))
        assert output_rms < input_rms

    def test_strength_scales_with_distance(self):
        seg = _make_mono_segment(10000.0)
        result_d1 = _apply_crossfade_eq(seg, "outgoing", 1)
        result_d3 = _apply_crossfade_eq(seg, "outgoing", 3)
        samples_d1 = np.frombuffer(result_d1.raw_data, dtype=np.int16).astype(np.float32)
        samples_d3 = np.frombuffer(result_d3.raw_data, dtype=np.int16).astype(np.float32)
        rms_d1 = np.sqrt(np.mean(samples_d1 ** 2))
        rms_d3 = np.sqrt(np.mean(samples_d3 ** 2))
        assert rms_d1 > rms_d3

    def test_full_strength_capped(self):
        seg = _make_mono_segment(10000.0)
        result_d3 = _apply_crossfade_eq(seg, "outgoing", 3)
        result_d6 = _apply_crossfade_eq(seg, "outgoing", 6)
        samples_d3 = np.frombuffer(result_d3.raw_data, dtype=np.int16).astype(np.float32)
        samples_d6 = np.frombuffer(result_d6.raw_data, dtype=np.int16).astype(np.float32)
        rms_d3 = np.sqrt(np.mean(samples_d3 ** 2))
        rms_d6 = np.sqrt(np.mean(samples_d6 ** 2))
        assert abs(rms_d3 - rms_d6) / max(rms_d3, 1e-9) < 0.01

    def test_output_length_preserved(self):
        seg = _make_mono_segment(440.0)
        result = _apply_crossfade_eq(seg, "incoming", 2)
        assert result.duration_seconds == pytest.approx(seg.duration_seconds, abs=0.001)

    def test_stereo_input_no_error(self):
        t = np.linspace(0, 1.0, 44100, endpoint=False)
        wave = np.sin(2.0 * np.pi * 440.0 * t)
        int_data = (wave * 32767.0).astype(np.int16)
        stereo_data = np.stack([int_data, int_data], axis=1)
        seg = AudioSegment(
            data=stereo_data.tobytes(),
            sample_width=2,
            frame_rate=44100,
            channels=2,
        )
        # Should complete without exception
        result = _apply_crossfade_eq(seg, "outgoing", 3)
        assert result is not None
