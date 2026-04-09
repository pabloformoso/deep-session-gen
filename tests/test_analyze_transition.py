"""Tests for analyze_transition() BPM stretch safety bounds."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_catalog(tracks):
    return {"tracks": tracks}


def _make_track(track_id, display_name, bpm, camelot_key="8A"):
    return {
        "id": track_id,
        "display_name": display_name,
        "bpm": bpm,
        "camelot_key": camelot_key,
    }


class TestAnalyzeTransitionStretch:
    def _call(self, catalog_data, track_a_id, track_b_id):
        import agent.tools as tools

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(catalog_data, f)
            tmp = Path(f.name)

        try:
            with patch.object(tools, "_CATALOG_PATH", tmp):
                return tools.analyze_transition(track_a_id, track_b_id, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_stretch_warning_appears_when_ratio_exceeds_1_5(self):
        """80 BPM → 130 BPM gives ratio 1.625×, should trigger stretch warning."""
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 80.0),
            _make_track("genre--track-b", "Track B", 130.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "stretch" in result.lower()
        assert "1.62" in result or "1.63" in result  # ratio 130/80 = 1.625

    def test_stretch_warning_not_present_when_bpms_close(self):
        """120 BPM → 125 BPM gives ratio ~1.04×, should not trigger stretch warning."""
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 120.0),
            _make_track("genre--track-b", "Track B", 125.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "stretch" not in result.lower()

    def test_stretch_warning_exactly_at_boundary(self):
        """Ratio of exactly 1.5 should NOT trigger the warning (strictly >1.5)."""
        # 100 → 150 BPM gives ratio = 1.5 exactly
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 100.0),
            _make_track("genre--track-b", "Track B", 150.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "stretch" not in result.lower()

    def test_stretch_warning_just_above_boundary(self):
        """Ratio just above 1.5 should trigger the warning."""
        # 100 → 151 BPM gives ratio = 1.51
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 100.0),
            _make_track("genre--track-b", "Track B", 151.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "stretch" in result.lower()

    def test_stretch_warning_direction_independent(self):
        """High → Low BPM (130 → 80) should also trigger stretch warning."""
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 130.0),
            _make_track("genre--track-b", "Track B", 80.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "stretch" in result.lower()

    def test_stretch_warning_contains_bridge_recommendation(self):
        """Stretch warning should recommend a bridge track."""
        catalog = _make_catalog([
            _make_track("genre--track-a", "Track A", 80.0),
            _make_track("genre--track-b", "Track B", 130.0),
        ])
        result = self._call(catalog, "genre--track-a", "genre--track-b")
        assert "bridge" in result.lower()
