"""Unit tests for the v2.6.0 endless / improvisation mode plumbing.

Covers the engine API surface (``append_track``, ``_maybe_end_or_extend``
gate, ``playlist_running_low`` emission, ``_autoplay_pick`` ranking).
Uses ``LiveEngineBrowser`` for the integration tests because it has the
simpler state machine (no audio device, no pre-stretch) and exercises
exactly the same code paths that the web/YouTube use case hits.
"""
from __future__ import annotations

import time

import pytest

from agent.live_engine import (
    ENDLESS_APPEND_CAP,
    ENDLESS_GRACE_SEC,
    ENDLESS_NO_REPEAT_WINDOW,
    ENDLESS_WARNING,
    PLAYLIST_RUNNING_LOW,
    SESSION_ENDED,
    TRACK_ENDED,
    TRACK_STARTED,
    LiveEngineBrowser,
    _autoplay_pick,
    _camelot_distance,
    _endless_pick,
    _recent_window_ids,
)


def _track(
    track_id: str,
    *,
    # v3.9.1 — default above MIN_TRACK_DURATION_SEC so tests that are
    # not about the eligibility screen don't trip it. Tests that poke
    # positions near a 60 s end keep an explicit duration_sec=60 for
    # their PLAYLIST tracks (playlists are not screened — only picks).
    duration_sec: float = 240.0,
    bpm: float = 120.0,
    camelot_key: str = "8A",
    genre_folder: str = "lofi - ambient",
) -> dict:
    return {
        "id": track_id,
        "display_name": f"Track {track_id}",
        "bpm": bpm,
        "camelot_key": camelot_key,
        "duration_sec": duration_sec,
        "genre_folder": genre_folder,
        "genre": genre_folder,
        "hot_cues": [],
    }


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.get("type") for e in self.events]


# ---------------------------------------------------------------------------
# _camelot_distance — pure function unit tests
# ---------------------------------------------------------------------------

def test_camelot_distance_identical_keys_is_zero():
    assert _camelot_distance("8A", "8A") == 0.0


def test_camelot_distance_adjacent_number_same_letter_is_one():
    assert _camelot_distance("8A", "9A") == 1.0
    assert _camelot_distance("8A", "7A") == 1.0


def test_camelot_distance_wheel_wraps_around():
    # 1A and 12A are adjacent on the wheel.
    assert _camelot_distance("1A", "12A") == 1.0


def test_camelot_distance_letter_flip_adds_half():
    # Same number, A vs B = the "energy boost" move.
    assert _camelot_distance("8A", "8B") == 0.5
    assert _camelot_distance("8A", "9B") == 1.5


def test_camelot_distance_unknown_or_malformed_returns_max():
    assert _camelot_distance(None, "8A") == 6.0
    assert _camelot_distance("8A", None) == 6.0
    assert _camelot_distance("garbage", "8A") == 6.0
    assert _camelot_distance("13A", "8A") == 6.0  # out of range
    assert _camelot_distance("8C", "8A") == 6.0   # bad letter


# ---------------------------------------------------------------------------
# _autoplay_pick — pure function unit tests
# ---------------------------------------------------------------------------

def test_autoplay_pick_filters_by_genre_folder():
    catalog = [
        _track("a", genre_folder="techno"),
        _track("b", genre_folder="lofi - ambient", bpm=76),
        _track("c", genre_folder="lofi - ambient", bpm=80),
    ]
    current = _track("playing", bpm=76, genre_folder="lofi - ambient")
    pick = _autoplay_pick(current, catalog, "lofi - ambient", set())
    assert pick is not None
    assert pick["genre_folder"] == "lofi - ambient"


def test_autoplay_pick_drops_already_played_ids():
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=76),
        _track("b", genre_folder="lofi - ambient", bpm=80),
    ]
    current = _track("playing", bpm=78, genre_folder="lofi - ambient")
    pick = _autoplay_pick(current, catalog, "lofi - ambient", {"a"})
    assert pick is not None and pick["id"] == "b"


def test_autoplay_pick_returns_none_when_no_in_genre_candidates():
    catalog = [_track("a", genre_folder="techno", bpm=130)]
    current = _track("playing", bpm=76, genre_folder="lofi - ambient")
    assert _autoplay_pick(current, catalog, "lofi - ambient", set()) is None


def test_autoplay_pick_ranks_closest_bpm_first():
    catalog = [
        _track("far", genre_folder="lofi - ambient", bpm=120),
        _track("near", genre_folder="lofi - ambient", bpm=78),
        _track("medium", genre_folder="lofi - ambient", bpm=90),
    ]
    current = _track("playing", bpm=76, genre_folder="lofi - ambient")
    pick = _autoplay_pick(current, catalog, "lofi - ambient", set())
    assert pick is not None and pick["id"] == "near"


def test_autoplay_pick_breaks_bpm_ties_with_camelot_distance():
    # Two candidates equidistant in BPM — pick the closer Camelot.
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=80, camelot_key="3A"),
        _track("b", genre_folder="lofi - ambient", bpm=80, camelot_key="8A"),
    ]
    current = _track("playing", bpm=82, genre_folder="lofi - ambient", camelot_key="8A")
    pick = _autoplay_pick(current, catalog, "lofi - ambient", set())
    assert pick is not None and pick["id"] == "b"


def test_autoplay_pick_returns_none_on_empty_catalog():
    current = _track("playing")
    assert _autoplay_pick(current, [], "lofi - ambient", set()) is None


# ---------------------------------------------------------------------------
# _autoplay_pick — allow_repeats fallback (24/7 streaming)
# ---------------------------------------------------------------------------

def test_autoplay_pick_returns_none_when_all_in_genre_excluded_without_allow_repeats():
    """Default behaviour: exclude_ids covers every in-genre track →
    no candidate. This is what kills a session without endless mode's
    recycle fallback."""
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=76),
        _track("b", genre_folder="lofi - ambient", bpm=80),
    ]
    current = _track("playing", bpm=78, genre_folder="lofi - ambient")
    pick = _autoplay_pick(current, catalog, "lofi - ambient", {"a", "b"})
    assert pick is None


def test_autoplay_pick_with_allow_repeats_recycles_excluded_tracks():
    """allow_repeats: when the exclude filter eats the whole pool, fall
    back to the full in-genre catalog so the stream keeps going."""
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=76),
        _track("b", genre_folder="lofi - ambient", bpm=80),
    ]
    current = _track("playing", bpm=78, genre_folder="lofi - ambient")
    pick = _autoplay_pick(
        current, catalog, "lofi - ambient", {"a", "b"}, allow_repeats=True
    )
    # Either 'a' or 'b' is fine — both are equidistant in BPM. The
    # important assertion is that we got SOMETHING back.
    assert pick is not None
    assert pick["id"] in {"a", "b"}


def test_autoplay_pick_with_allow_repeats_avoids_current_track():
    """Even on a recycle, never pick the track that just finished —
    back-to-back repeats sound broken."""
    catalog = [
        _track("currently-playing", genre_folder="lofi - ambient", bpm=76),
        _track("other", genre_folder="lofi - ambient", bpm=120),
    ]
    current = _track("currently-playing", bpm=76, genre_folder="lofi - ambient")
    # Both ids excluded → recycle path. Should pick the non-current one
    # even though 'currently-playing' is a much better BPM match.
    pick = _autoplay_pick(
        current,
        catalog,
        "lofi - ambient",
        {"currently-playing", "other"},
        allow_repeats=True,
    )
    assert pick is not None and pick["id"] == "other"


def test_autoplay_pick_with_allow_repeats_returns_none_when_only_current_in_genre():
    """If the only in-genre track is the one currently playing, the
    recycle can't pick anything safe — return None and let the engine
    end the session cleanly."""
    catalog = [
        _track("currently-playing", genre_folder="lofi - ambient", bpm=76),
        _track("wrong-genre", genre_folder="techno", bpm=130),
    ]
    current = _track("currently-playing", bpm=76, genre_folder="lofi - ambient")
    pick = _autoplay_pick(
        current,
        catalog,
        "lofi - ambient",
        {"currently-playing"},
        allow_repeats=True,
    )
    assert pick is None


def test_autoplay_pick_allow_repeats_noop_when_fresh_candidate_exists():
    """allow_repeats must not change behaviour when a non-excluded
    in-genre track is still available — the fresh pick wins."""
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=76),
        _track("b", genre_folder="lofi - ambient", bpm=80),
    ]
    current = _track("playing", bpm=78, genre_folder="lofi - ambient")
    pick = _autoplay_pick(
        current, catalog, "lofi - ambient", {"a"}, allow_repeats=True
    )
    assert pick is not None and pick["id"] == "b"


# ---------------------------------------------------------------------------
# append_track — Browser engine
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _endless_pick — the tier ladder that keeps a 24/7 set out of one genre
#
# Regression cover for the 2026-09-07 live finding: an 18 h endless set
# played 352 tracks drawn from exactly the 48 'Healing' entries while
# 465 other catalog tracks were never touched, because every recycle
# tier inside _autoplay_pick is in_genre and the genre is re-read from
# the track the picker itself chose on the previous lap.
# ---------------------------------------------------------------------------

def test_endless_pick_prefers_unheard_in_genre():
    catalog = [
        _track("a", genre_folder="healing", bpm=70),
        _track("b", genre_folder="techno", bpm=71),
    ]
    current = _track("playing", bpm=70, genre_folder="healing")
    pick, tier = _endless_pick(current, catalog, "healing", set())
    assert tier == "in_genre"
    assert pick is not None and pick["id"] == "a"


def test_endless_pick_widens_out_of_genre_before_recycling():
    """The bug: this used to recycle 'a' instead of ever reaching 'b'."""
    catalog = [
        _track("a", genre_folder="healing", bpm=70),
        _track("b", genre_folder="lofi - ambient", bpm=72),
    ]
    current = _track("playing", bpm=70, genre_folder="healing")
    # The whole healing genre has been heard; the rest of the catalog has not.
    pick, tier = _endless_pick(current, catalog, "healing", {"a"})
    assert tier == "widened"
    assert pick is not None and pick["id"] == "b"


def test_endless_pick_widened_tier_still_ranks_by_musical_distance():
    """Widening must cross to the nearest neighbour, not to an arbitrary track."""
    catalog = [
        _track("heard", genre_folder="healing", bpm=70),
        _track("far", genre_folder="techno", bpm=140, camelot_key="3B"),
        _track("near", genre_folder="lofi - ambient", bpm=72, camelot_key="8A"),
    ]
    current = _track("playing", bpm=70, genre_folder="healing", camelot_key="8A")
    pick, tier = _endless_pick(current, catalog, "healing", {"heard"})
    assert tier == "widened"
    assert pick is not None and pick["id"] == "near"


def test_endless_pick_recycles_only_once_whole_catalog_is_heard():
    catalog = [
        _track("a", genre_folder="healing", bpm=70),
        _track("b", genre_folder="lofi - ambient", bpm=72),
    ]
    current = _track("playing", bpm=70, genre_folder="healing")
    pick, tier = _endless_pick(current, catalog, "healing", {"a", "b"})
    assert tier == "recycled"
    assert pick is not None and pick["id"] in {"a", "b"}


def test_endless_pick_recycle_tier_is_not_locked_to_the_genre():
    """Even the last-resort tier draws on the whole catalog, not 1/10th of it."""
    catalog = [_track("only", genre_folder="lofi - ambient", bpm=72)]
    current = _track("playing", bpm=70, genre_folder="healing")
    pick, tier = _endless_pick(current, catalog, "healing", {"only"})
    assert tier == "recycled"
    assert pick is not None and pick["id"] == "only"


def test_endless_pick_reports_none_on_empty_catalog():
    current = _track("playing", bpm=70, genre_folder="healing")
    pick, tier = _endless_pick(current, [], "healing", set())
    assert pick is None and tier == "none"


def test_endless_pick_never_ids_survive_every_tier():
    """A queued track must not be picked, not even by the recycle tier."""
    catalog = [_track("queued", genre_folder="healing", bpm=70)]
    current = _track("playing", bpm=70, genre_folder="healing")
    pick, tier = _endless_pick(
        current, catalog, "healing", {"queued"}, never_ids={"queued"}
    )
    assert pick is None and tier == "none"


def test_endless_pick_escapes_a_fully_played_genre_across_many_laps():
    """The live symptom, in miniature: a small genre must not trap the set."""
    catalog = [_track(f"h{i}", genre_folder="healing", bpm=70) for i in range(3)]
    catalog += [_track(f"o{i}", genre_folder="lofi - ambient", bpm=71) for i in range(3)]
    current = _track("playing", bpm=70, genre_folder="healing")
    played: set[str] = set()
    seen_genres = set()
    for _ in range(6):
        # Faithful to the caller: the genre is re-read from whatever is
        # playing, which is exactly what welded the door shut in prod.
        pick, tier = _endless_pick(
            current, catalog, current.get("genre_folder"), played
        )
        assert pick is not None, "the ladder must always find a continuation"
        played.add(pick["id"])
        seen_genres.add(pick["genre_folder"])
        current = pick
    # All six tracks consumed exactly once — no repeats, both genres reached.
    assert len(played) == 6
    assert seen_genres == {"healing", "lofi - ambient"}


def test_append_track_adds_to_playlist_and_returns_position():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    rec.events.clear()

    msg = engine.append_track(_track("b"))
    assert "Appended" in msg and "position 2" in msg
    assert [t["id"] for t in engine.playlist] == ["a", "b"]


def test_append_track_rejects_track_without_id():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    msg = engine.append_track({"display_name": "no-id"})
    assert "id" in msg.lower()
    assert len(engine.playlist) == 1


def test_append_track_enforces_session_cap():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    # Burn through the cap.
    engine._endless_appended = ENDLESS_APPEND_CAP
    rec.events.clear()
    msg = engine.append_track(_track("b"))
    assert "cap reached" in msg.lower()
    # No new tracks landed.
    assert [t["id"] for t in engine.playlist] == ["a"]
    # Warning event surfaced.
    assert any(
        e.get("type") == ENDLESS_WARNING and e.get("reason") == "cap_reached"
        for e in rec.events
    )


def test_append_track_resets_low_water_guard():
    """A fresh append re-arms the playlist_running_low edge so the
    engine can fire it again once the new tail becomes the last
    track."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    engine._low_water_fired = True
    engine._low_water_at = time.monotonic()
    engine.append_track(_track("b"))
    assert engine._low_water_fired is False
    assert engine._low_water_at is None


# ---------------------------------------------------------------------------
# _maybe_end_or_extend — gating semantics on the Browser engine
# ---------------------------------------------------------------------------

def test_endless_off_emits_session_ended_immediately():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    # endless mode default is False — _maybe_end_or_extend should
    # behave exactly like the legacy engine.
    assert engine._maybe_end_or_extend(_track("a")) is True
    assert SESSION_ENDED in rec.types()


def test_endless_on_with_successor_returns_false_and_no_session_ended():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a"), _track("b")])
    engine._endless_mode = True
    rec.events.clear()
    assert engine._maybe_end_or_extend(_track("a")) is False
    assert SESSION_ENDED not in rec.types()


def test_endless_on_grace_window_blocks_session_ended():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    engine._endless_mode = True
    rec.events.clear()
    # First entry starts the grace clock; no SESSION_ENDED yet.
    assert engine._maybe_end_or_extend(_track("a")) is False
    assert SESSION_ENDED not in rec.types()
    # Re-poll inside the grace window still defers.
    assert engine._maybe_end_or_extend(_track("a")) is False
    assert SESSION_ENDED not in rec.types()


def test_endless_on_grace_elapsed_with_no_candidates_ends_session(monkeypatch):
    """Browser engine reads catalog fresh in _maybe_end_or_extend — when
    that returns no in-genre candidates, an endless_warning event is
    emitted and the session ends normally (no infinite loop)."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a", genre_folder="lofi - ambient")])
    engine._endless_mode = True
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    # Empty catalog → no candidates.
    monkeypatch.setattr("agent.live_engine._load_catalog", lambda: [])
    rec.events.clear()
    assert engine._maybe_end_or_extend(_track("a")) is True
    types = rec.types()
    assert ENDLESS_WARNING in types
    assert SESSION_ENDED in types
    # Warning carries the no_candidates reason for the frontend banner.
    warn = next(e for e in rec.events if e.get("type") == ENDLESS_WARNING)
    assert warn.get("reason") == "no_candidates"


def test_endless_on_grace_elapsed_with_candidate_appends_and_continues(monkeypatch):
    """When the grace window expires with no LLM append, the engine
    auto-picks an in-genre track from the catalog and appends it
    itself — no SESSION_ENDED, no warning."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a", genre_folder="lofi - ambient", bpm=76)])
    engine._endless_mode = True
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        lambda: [_track("auto-pick", genre_folder="lofi - ambient", bpm=78)],
    )
    rec.events.clear()
    assert engine._maybe_end_or_extend(
        _track("a", genre_folder="lofi - ambient", bpm=76)
    ) is False
    # The new track is now the tail.
    assert engine.playlist[-1]["id"] == "auto-pick"
    # No SESSION_ENDED, no warning.
    types = rec.types()
    assert SESSION_ENDED not in types
    assert ENDLESS_WARNING not in types


def test_endless_on_exhausted_in_genre_recycles_and_keeps_streaming(monkeypatch):
    """The 24/7 streaming case: every in-genre track in the catalog is
    already in the playlist, so the LLM has nothing fresh to append.
    Endless mode must still continue by recycling a previously-played
    track instead of emitting ENDLESS_WARNING + SESSION_ENDED."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([
        _track("a", genre_folder="lofi - ambient", bpm=76),
        _track("b", genre_folder="lofi - ambient", bpm=80),
    ])
    engine._endless_mode = True
    # Advance to the last track so remaining_after == 0 and the
    # fallback path actually fires.
    engine._idx = 1
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    # The catalog matches the playlist exactly — no fresh tracks left.
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        lambda: [
            _track("a", genre_folder="lofi - ambient", bpm=76),
            _track("b", genre_folder="lofi - ambient", bpm=80),
        ],
    )
    rec.events.clear()
    # Pretend 'b' just finished — recycle must avoid 'b' and pick 'a'.
    current = _track("b", genre_folder="lofi - ambient", bpm=80)
    assert engine._maybe_end_or_extend(current) is False
    assert engine.playlist[-1]["id"] == "a"
    types = rec.types()
    assert SESSION_ENDED not in types
    assert ENDLESS_WARNING not in types


# ---------------------------------------------------------------------------
# ENDLESS_APPEND_CAP — env override
# ---------------------------------------------------------------------------

def test_endless_append_cap_default_supports_long_streams():
    """Default cap must comfortably exceed a multi-day stream. At
    ~1 min/track, 10000 covers about a week."""
    from agent.live_engine import ENDLESS_APPEND_CAP
    assert ENDLESS_APPEND_CAP >= 10000


def test_endless_append_cap_overridable_via_env_var(monkeypatch):
    """Operators can tighten or loosen the runaway guard without code
    changes via APOLLO_ENDLESS_APPEND_CAP."""
    monkeypatch.setenv("APOLLO_ENDLESS_APPEND_CAP", "42")
    import importlib

    import agent.live_engine as engine_mod
    importlib.reload(engine_mod)
    try:
        assert engine_mod.ENDLESS_APPEND_CAP == 42
    finally:
        # Reset the module so other tests see the default again.
        monkeypatch.delenv("APOLLO_ENDLESS_APPEND_CAP", raising=False)
        importlib.reload(engine_mod)


def test_endless_append_cap_env_var_ignores_garbage(monkeypatch):
    """A malformed env value must not blow up engine import — fall
    back to the default instead."""
    monkeypatch.setenv("APOLLO_ENDLESS_APPEND_CAP", "not-a-number")
    import importlib

    import agent.live_engine as engine_mod
    importlib.reload(engine_mod)
    try:
        assert engine_mod.ENDLESS_APPEND_CAP == 10000
    finally:
        monkeypatch.delenv("APOLLO_ENDLESS_APPEND_CAP", raising=False)
        importlib.reload(engine_mod)


# ---------------------------------------------------------------------------
# playlist_running_low edge — Browser engine
# ---------------------------------------------------------------------------

def test_playlist_running_low_fires_once_when_remaining_one_and_endless():
    """Drive the engine through APPROACHING_CF when remaining == 1
    and endless is ON. The edge should fire exactly once per
    'approaching-the-last-track' window."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60), _track("b", duration_sec=60)])
    engine._endless_mode = True
    rec.events.clear()

    # 35 s in, cf at ~48 s (60 - 12) so secs_to_cf = 13 — well under
    # approach_warn_sec. APPROACHING_CF + PLAYLIST_RUNNING_LOW should
    # both fire.
    engine.report_playback_pos(track_id="a", current_time=35.0)
    types = rec.types()
    assert PLAYLIST_RUNNING_LOW in types
    # Pinging again shouldn't re-fire the edge.
    n_first = types.count(PLAYLIST_RUNNING_LOW)
    engine.report_playback_pos(track_id="a", current_time=37.0)
    n_second = rec.types().count(PLAYLIST_RUNNING_LOW)
    assert n_second == n_first


def test_playlist_running_low_does_not_fire_when_endless_off():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60), _track("b", duration_sec=60)])
    # endless OFF — even with remaining == 1, no PLAYLIST_RUNNING_LOW.
    rec.events.clear()
    engine.report_playback_pos(track_id="a", current_time=35.0)
    assert PLAYLIST_RUNNING_LOW not in rec.types()


def test_playlist_running_low_does_not_fire_when_more_than_one_track_left():
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play(
        [_track("a", duration_sec=60), _track("b", duration_sec=60), _track("c", duration_sec=60)]
    )
    engine._endless_mode = True
    rec.events.clear()
    engine.report_playback_pos(track_id="a", current_time=35.0)
    # remaining = 2 → no running-low signal even with endless ON.
    assert PLAYLIST_RUNNING_LOW not in rec.types()


# ---------------------------------------------------------------------------
# End-to-end TRACK_ENDED path with endless mode
# ---------------------------------------------------------------------------

def test_track_ended_with_endless_and_auto_pick_advances_naturally(monkeypatch):
    """The deterministic-fallback path inside report_track_ended:
    last track ends, no append from LLM, grace expired → engine
    picks an in-genre continuation, appends it, then advances to it
    instead of emitting SESSION_ENDED."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=10, genre_folder="lofi - ambient", bpm=76)])
    engine._endless_mode = True
    # Pretend the grace window already elapsed.
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        lambda: [_track("next", genre_folder="lofi - ambient", bpm=78)],
    )
    rec.events.clear()
    engine.report_track_ended("a")
    types = rec.types()
    # TRACK_ENDED for the last track, then TRACK_STARTED for the
    # auto-picked successor — no SESSION_ENDED in between.
    assert TRACK_ENDED in types
    assert TRACK_STARTED in types
    assert SESSION_ENDED not in types
    # New track is loaded as the current one.
    assert engine.playlist[engine._idx]["id"] == "next"


# ---------------------------------------------------------------------------
# v3.6 — the 2026-07-10 live deadlock regressions
# ---------------------------------------------------------------------------

def test_running_low_fires_while_last_track_plays():
    """v2.6.0 nested the low-water poke inside the APPROACHING_CF block,
    which required a next track — so the poke could never fire while
    the LAST track played. Regression: single-track playlist, endless
    ON, ping inside the warn window → PLAYLIST_RUNNING_LOW must fire."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60)])
    engine._endless_mode = True
    rec.events.clear()
    # cf point = 60 - 12 - 5 = 43 s; 35 s in → secs_to_cf = 8 < 30.
    engine.report_playback_pos(track_id="a", current_time=35.0)
    assert PLAYLIST_RUNNING_LOW in rec.types()
    # One-shot per low-water window.
    n = rec.types().count(PLAYLIST_RUNNING_LOW)
    engine.report_playback_pos(track_id="a", current_time=36.0)
    assert rec.types().count(PLAYLIST_RUNNING_LOW) == n


def test_running_low_refires_for_appended_tail_track(monkeypatch):
    """The full 2026-07-10 death spiral: original playlist runs out, the
    fallback appends a tail track — that tail track must RE-poke when
    it approaches its own end, so extension keeps cascading instead of
    dying one track after the original set."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60, bpm=76)])
    engine._endless_mode = True
    # Poke fires for 'a' (last track, no successor).
    engine.report_playback_pos(track_id="a", current_time=35.0)
    assert PLAYLIST_RUNNING_LOW in rec.types()
    # LLM appends a continuation → low-water re-arms.
    engine.append_track(_track("b", duration_sec=60, bpm=78))
    rec.events.clear()
    # Crossfade into 'b' at the cf point.
    engine.report_playback_pos(track_id="a", current_time=43.5)
    assert engine.playlist[engine._idx]["id"] == "b"
    # 'b' (now the last track) approaches its end → poke must re-fire.
    engine.report_playback_pos(track_id="b", current_time=35.0)
    assert PLAYLIST_RUNNING_LOW in rec.types()


def test_inflight_extend_appends_at_cf_point_and_crossfades(monkeypatch):
    """Past the crossfade point on the last track with nothing queued,
    endless mode must extend WHILE audio still plays (the deck dies at
    the end and takes the ping stream with it). The next ping then
    crossfades seamlessly into the appended track."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60, genre_folder="lofi - ambient", bpm=76)])
    engine._endless_mode = True
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        lambda: [_track("auto", genre_folder="lofi - ambient", bpm=78)],
    )
    # Poke at the approach edge starts the grace clock…
    engine.report_playback_pos(track_id="a", current_time=35.0)
    # …pretend the grace window has since elapsed.
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    rec.events.clear()
    # Past the cf point (43 s) but before the end (60 s): extend now.
    engine.report_playback_pos(track_id="a", current_time=50.0)
    assert engine.playlist[-1]["id"] == "auto"
    assert SESSION_ENDED not in rec.types()
    assert TRACK_ENDED not in rec.types()  # 'a' is still playing
    # Next ping sees the successor → seamless crossfade, no silence.
    engine.report_playback_pos(track_id="a", current_time=50.3)
    types = rec.types()
    assert TRACK_STARTED in types
    assert engine.playlist[engine._idx]["id"] == "auto"


def test_inflight_extend_is_one_shot_per_track(monkeypatch):
    """A fruitless catalog scan at the cf point must not repeat at ping
    rate (~4 Hz) for the rest of the track."""
    calls = {"n": 0}

    def _counting_catalog():
        calls["n"] += 1
        return []

    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60)])
    engine._endless_mode = True
    # Fire the poke at the approach edge FIRST (it owns _low_water_at),
    # then age the clock past the grace window.
    engine.report_playback_pos(track_id="a", current_time=35.0)
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    monkeypatch.setattr("agent.live_engine._load_catalog", _counting_catalog)
    engine.report_playback_pos(track_id="a", current_time=50.0)
    engine.report_playback_pos(track_id="a", current_time=51.0)
    engine.report_playback_pos(track_id="a", current_time=52.0)
    assert calls["n"] == 1


def test_inflight_extend_recycles_played_tracks_when_catalog_exhausted(monkeypatch):
    """The new in-flight extension path must honour allow_repeats: when
    every in-genre track is already in the playlist, it recycles a
    previously-played one (never the currently-playing track) instead
    of letting the set die."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([
        _track("a", duration_sec=60, genre_folder="lofi - ambient", bpm=76),
        _track("b", duration_sec=60, genre_folder="lofi - ambient", bpm=80),
    ])
    engine._endless_mode = True
    engine._idx = 1  # 'b' is playing and is the last track
    # Catalog == playlist: nothing fresh left in the genre.
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        # v3.9.1 — catalog copies must be session-eligible (≥ the min
        # duration) or the eligibility screen empties the recycle pool;
        # the PLAYLIST keeps duration 60 for the poke math below.
        lambda: [
            _track("a", duration_sec=240, genre_folder="lofi - ambient", bpm=76),
            _track("b", duration_sec=240, genre_folder="lofi - ambient", bpm=80),
        ],
    )
    engine.report_playback_pos(track_id="b", current_time=35.0)  # poke
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    rec.events.clear()
    engine.report_playback_pos(track_id="b", current_time=50.0)
    # Recycled 'a' (not 'b' — no back-to-back repeat), session alive.
    assert engine.playlist[-1]["id"] == "a"
    assert len(engine.playlist) == 3
    assert SESSION_ENDED not in rec.types()
    assert ENDLESS_WARNING not in rec.types()


# ---------------------------------------------------------------------------
# v3.6.1 — no-repeat window on the recycle path
# ---------------------------------------------------------------------------

def test_recent_window_ids_takes_last_n_up_to_current():
    playlist = [_track(x) for x in "abcde"]
    assert _recent_window_ids(playlist, 4, 3) == ["c", "d", "e"]
    # Window wider than what's been played → everything so far.
    assert _recent_window_ids(playlist, 1, 20) == ["a", "b"]
    # Tracks after idx haven't played and must not count.
    assert _recent_window_ids(playlist, 0, 20) == ["a"]


def test_no_repeat_window_default_is_about_an_hour():
    # 20 tracks ≈ 1 h of music at typical (~3 min) track lengths.
    assert ENDLESS_NO_REPEAT_WINDOW == 20


def test_no_repeat_window_overridable_via_env_var(monkeypatch):
    monkeypatch.setenv("APOLLO_ENDLESS_NO_REPEAT_WINDOW", "5")
    import importlib

    import agent.live_engine as engine_mod
    importlib.reload(engine_mod)
    try:
        assert engine_mod.ENDLESS_NO_REPEAT_WINDOW == 5
    finally:
        monkeypatch.delenv("APOLLO_ENDLESS_NO_REPEAT_WINDOW", raising=False)
        importlib.reload(engine_mod)


def test_autoplay_pick_recycle_skips_recent_window():
    """Recycling must rotate the catalog: a track heard within the
    recent window loses to one heard longer ago, even when the recent
    one ranks better on (Δbpm, camelot)."""
    catalog = [
        _track("old", genre_folder="lofi - ambient", bpm=90),   # worse Δbpm
        _track("fresh", genre_folder="lofi - ambient", bpm=77),  # better Δbpm
    ]
    current = _track("playing", genre_folder="lofi - ambient", bpm=76)
    exclude = {"old", "fresh", "playing"}  # catalog exhausted → recycle tier
    pick = _autoplay_pick(
        current, catalog, "lofi - ambient", exclude,
        allow_repeats=True, recent_ids=["fresh", "playing"],
    )
    assert pick is not None and pick["id"] == "old"


def test_autoplay_pick_recycle_degrades_when_window_covers_catalog():
    """A window wider than the in-genre catalog must fall back to the
    v2.6.0 rule (avoid back-to-back only) instead of ending the set."""
    catalog = [
        _track("a", genre_folder="lofi - ambient", bpm=77),
        _track("b", genre_folder="lofi - ambient", bpm=90),
    ]
    current = _track("b", genre_folder="lofi - ambient", bpm=90)
    pick = _autoplay_pick(
        current, catalog, "lofi - ambient", {"a", "b"},
        allow_repeats=True, recent_ids=["a", "b"],
    )
    # Everything is "recent" → degrade: anything but the current track.
    assert pick is not None and pick["id"] == "a"


def test_engine_recycle_respects_no_repeat_window(monkeypatch):
    """Engine-level: with the window monkeypatched to 2, the in-flight
    recycle must skip the two most recently played tracks and reach
    back for the older one."""
    monkeypatch.setattr("agent.live_engine.ENDLESS_NO_REPEAT_WINDOW", 2)
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    # 'c' is playing; recent window (2) = [b, c] → must pick 'a' even
    # though 'b' is the closer BPM match to 'c'.
    engine.play([
        _track("a", duration_sec=60, genre_folder="lofi - ambient", bpm=70),
        _track("b", duration_sec=60, genre_folder="lofi - ambient", bpm=79),
        _track("c", duration_sec=60, genre_folder="lofi - ambient", bpm=80),
    ])
    engine._endless_mode = True
    engine._idx = 2
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        # v3.9.1 — see the recycle test above: catalog copies eligible,
        # playlist stays at 60 s for the poke math.
        lambda: [
            _track("a", duration_sec=240, genre_folder="lofi - ambient", bpm=70),
            _track("b", duration_sec=240, genre_folder="lofi - ambient", bpm=79),
            _track("c", duration_sec=240, genre_folder="lofi - ambient", bpm=80),
        ],
    )
    engine.report_playback_pos(track_id="c", current_time=35.0)  # poke
    engine._low_water_at = time.monotonic() - (ENDLESS_GRACE_SEC + 1)
    rec.events.clear()
    engine.report_playback_pos(track_id="c", current_time=50.0)
    assert engine.playlist[-1]["id"] == "a"
    assert SESSION_ENDED not in rec.types()


def test_track_over_bypasses_grace_and_extends_immediately(monkeypatch):
    """THE deadlock: the appended tail track ends, the gate said 'start
    grace timer and wait for the next poll' — but after a natural
    ``ended`` the browser never polls again, so the engine hung in
    silence until the user refreshed (observed live 2026-07-10). When
    the track is truly over, the gate must skip the grace and run the
    fallback immediately."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60, genre_folder="lofi - ambient", bpm=76)])
    engine._endless_mode = True
    monkeypatch.setattr(
        "agent.live_engine._load_catalog",
        lambda: [_track("rescue", genre_folder="lofi - ambient", bpm=78)],
    )
    # Note: NO low_water_at pre-set — this is the exact logged state
    # ("start grace timer (PLAYLIST_RUNNING_LOW never fired)").
    assert engine._low_water_at is None
    rec.events.clear()
    engine.report_track_ended("a")
    types = rec.types()
    assert SESSION_ENDED not in types
    assert TRACK_STARTED in types
    assert engine.playlist[engine._idx]["id"] == "rescue"


def test_track_over_with_no_candidates_ends_cleanly(monkeypatch):
    """track_over + empty catalog must still end the session cleanly
    (warning + SESSION_ENDED) rather than hanging."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec, approach_warn_sec=30)
    engine.play([_track("a", duration_sec=60)])
    engine._endless_mode = True
    monkeypatch.setattr("agent.live_engine._load_catalog", lambda: [])
    rec.events.clear()
    engine.report_track_ended("a")
    types = rec.types()
    assert ENDLESS_WARNING in types
    assert SESSION_ENDED in types


def test_play_resets_endless_bookkeeping():
    """A re-``play()`` after a WS reconnect must not inherit a
    minutes-old grace clock or a spent in-flight latch. ``_endless_mode``
    itself is owned by the WS handler and must survive."""
    rec = _Recorder()
    engine = LiveEngineBrowser(emitter=rec)
    engine.play([_track("a")])
    engine._endless_mode = True
    engine._low_water_fired = True
    engine._low_water_at = time.monotonic() - 300
    engine._endless_appended = 7
    engine._extend_attempted = True
    engine.play([_track("b")])
    assert engine._low_water_fired is False
    assert engine._low_water_at is None
    assert engine._endless_appended == 0
    assert engine._extend_attempted is False
    assert engine._endless_mode is True
