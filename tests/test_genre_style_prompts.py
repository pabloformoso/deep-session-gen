"""The genre must reach ACE as SOUND, not just as a folder name.

Before 2026-09-07 ``genre_folder`` only picked the default BPM and the
destination folder: ``_release_payload`` sent ACE the user's free text
and nothing else. Asking for "a calm track" under `healing` told the
model "a calm track" and a tempo — nothing about what healing sounds
like — so takes came back off-genre and could not be promoted into the
catalog, which is the entire point of generating them.

Also guards the two genres that had NO BPM window at all (`aural`, 179
tracks; `synthware`, 86): a genre with no window is off the generator's
allow-list AND has its raw detection stored verbatim, which is exactly
the poisoning ``BPM_GENRE_RANGES`` exists to prevent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "web") not in sys.path:
    sys.path.insert(0, str(ROOT / "web"))

import main  # noqa: E402
from agent.tools import (  # noqa: E402
    _BPM_GENRE_RANGES,
    GENRE_STYLE_PROMPTS,
    genre_style_prompt,
)

ORPHANS = ["aural", "synthware"]

#: tracks.json is NOT in git (worktrees and CI have no ``tracks/``), so
#: the catalog-grounded checks skip rather than fail there. They still
#: bite in the main checkout, which is where a genre folder is actually
#: added.
CATALOG = ROOT / "tracks" / "tracks.json"
needs_catalog = pytest.mark.skipif(
    not CATALOG.exists(), reason="tracks.json is not in git — main checkout only"
)


def _catalog_genre_folders() -> set[str]:
    raw = json.loads(CATALOG.read_text())
    tracks = raw["tracks"] if isinstance(raw, dict) else raw
    return {
        (t.get("genre_folder") or "").strip().lower()
        for t in tracks
        if t.get("genre_folder")
    }


# --- the two genres that had no window -----------------------------------

@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_has_a_bpm_window(genre):
    assert genre in main.BPM_GENRE_RANGES, f"{genre} still has no BPM window"


@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_window_is_exactly_one_octave(genre):
    """CLAUDE.md: hi == 2*lo so only one rung of the octave ladder qualifies."""
    lo, hi = main.BPM_GENRE_RANGES[genre]
    assert hi == 2 * lo, f"{genre} window {lo}-{hi} is not one octave"


@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_window_is_mirrored_into_agent_tools(genre):
    """The two copies have drifted before; a mismatch degrades silently."""
    assert _BPM_GENRE_RANGES[genre] == main.BPM_GENRE_RANGES[genre]


@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_has_a_theme(genre):
    assert genre in main.GENRE_THEMES


@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_theme_points_at_a_real_artwork_style(genre):
    """An unknown artwork_style falls back to 'abstract' SILENTLY."""
    style = main.GENRE_THEMES[genre]["artwork_style"]
    assert style in main.ARTWORK_PROMPTS, f"{genre} -> unknown style {style!r}"


@needs_catalog
@pytest.mark.parametrize("genre", ORPHANS)
def test_orphan_genre_window_covers_its_own_catalog_mass(genre):
    """The window is useless if the genre's real tempos sit outside it."""
    raw = json.loads(CATALOG.read_text())
    tracks = raw["tracks"] if isinstance(raw, dict) else raw
    bpms = sorted(
        float(t["bpm"])
        for t in tracks
        if (t.get("genre_folder") or "").strip().lower() == genre and t.get("bpm")
    )
    if not bpms:
        pytest.skip(f"no {genre} tracks in this checkout's catalog")
    lo, hi = main.BPM_GENRE_RANGES[genre]
    median = bpms[len(bpms) // 2]
    assert lo <= median <= hi, f"{genre} median {median} outside {lo}-{hi}"


# --- every catalog genre must be generatable ------------------------------

@needs_catalog
def test_every_catalog_genre_has_a_style_descriptor():
    """A folder with no descriptor generates off-genre and can't be promoted."""
    missing = sorted(_catalog_genre_folders() - set(GENRE_STYLE_PROMPTS))
    assert not missing, f"catalog genres with no style descriptor: {missing}"


def test_style_lookup_is_case_and_whitespace_insensitive():
    assert genre_style_prompt("  HEALING ") == GENRE_STYLE_PROMPTS["healing"]


def test_style_lookup_returns_none_for_unknown_genre():
    assert genre_style_prompt("not-a-genre") is None


# --- composition ----------------------------------------------------------

def _compose():
    from backend.generator import _compose_prompt

    return _compose_prompt


def test_style_is_composed_ahead_of_the_user_prompt():
    """Genre frames, the user's words specialise — so order matters."""
    out = _compose()("a calm track", "healing")
    assert out.startswith(GENRE_STYLE_PROMPTS["healing"])
    assert out.endswith("a calm track")


def test_unknown_genre_degrades_to_the_bare_prompt():
    """Old behaviour preserved: an unlisted folder still generates."""
    assert _compose()("a calm track", "not-a-genre") == "a calm track"


def test_empty_genre_degrades_to_the_bare_prompt():
    assert _compose()("a calm track", "") == "a calm track"


def test_empty_user_prompt_still_carries_the_genre():
    assert _compose()("", "synthware") == GENRE_STYLE_PROMPTS["synthware"]


def test_composed_prompt_is_trimmed_not_rejected():
    """A long user prompt must never turn into a 422."""
    assert len(_compose()("x" * 5000, "healing")) == 4000
