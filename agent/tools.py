"""
DJ Session Agent — Tool implementations.

Each tool follows AutoAgent's convention: all params are JSON-serialisable,
context_variables dict carries shared mutable state across turns.

The tools are registered as plain functions here; agent/run.py converts them
into the Anthropic tool-use schema automatically.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydub import AudioSegment

from agent.eligibility import (
    filter_session_eligible,
    ineligibility_reason,
)
from agent.track_identity import dedupe_takes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).parent
_PROJECT_DIR = _AGENT_DIR.parent
_CATALOG_PATH = _PROJECT_DIR / "tracks" / "tracks.json"
_MAIN_PY = _PROJECT_DIR / "main.py"
_MEMORY_PATH = _AGENT_DIR / "memory.json"

# ---------------------------------------------------------------------------
# Slug helper (mirrors main.py _slugify)
# ---------------------------------------------------------------------------
def _slugify(text: str) -> str:
    slug = text.lower()
    for ch in [" ", "/", "\\", "(", ")", ".", ","]:
        slug = slug.replace(ch, "-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Camelot helpers (duplicated from main.py to keep tools self-contained)
# ---------------------------------------------------------------------------
def _camelot_neighbors(key: str) -> set[str]:
    if not key or len(key) < 2:
        return set()
    try:
        num = int(key[:-1])
        letter = key[-1].upper()
    except (ValueError, IndexError):
        return set()
    opposite = "B" if letter == "A" else "A"
    return {
        key,
        f"{(num % 12) + 1}{letter}",
        f"{((num - 2) % 12) + 1}{letter}",
        f"{num}{opposite}",
    }


def _camelot_step_distance(key_a: str, key_b: str) -> int:
    """Return the minimum number of Camelot wheel steps between two keys (0–6+)."""
    if not key_a or not key_b:
        return 0
    if key_a == key_b:
        return 0
    visited = {key_a}
    frontier = {key_a}
    for steps in range(1, 7):
        next_frontier = set()
        for k in frontier:
            for neighbor in _camelot_neighbors(k):
                if neighbor not in visited:
                    if neighbor == key_b:
                        return steps
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return 6  # unreachable on a 24-node wheel → treat as max clash


_BPM_GENRE_RANGES = {
    "lofi - ambient": (60, 110),
    "lofi": (60, 110),
    "techno": (120, 160),
    "cyberpunk": (120, 160),
    "deep house": (115, 135),
    "cocktail house": (102, 126),
    "soul jazz": (75, 140),
    # Keep in sync with ``BPM_GENRE_RANGES`` in main.py. Only used here to
    # scale the energy curve, so a missing genre degrades to the (60, 200)
    # fallback rather than crashing — but for a 50-100 BPM genre that
    # fallback flattens every track to near-zero energy and the arranger
    # can no longer tell a drone from a build.
    "healing": (50, 100),
    "chillout": (60, 120),
    # Mirrors main.BPM_GENRE_RANGES — see the reasoning there.
    "aural": (48, 96),
    "synthware": (85, 170),
}


#: What each genre SOUNDS like, in the words ACE-Step understands.
#:
#: The gap this closes: ``genre_folder`` used to be a filing decision
#: only — it picked the default BPM and chose the destination folder,
#: and then ``_release_payload`` sent ACE the user's free text and
#: nothing else. Ask for "a calm track" under `healing` and the model
#: was told "a calm track" and a tempo; nothing carried healing's actual
#: identity. The takes came back off-genre and could not be promoted
#: into the catalog, which is the whole point of generating them
#: (reported 2026-09-07).
#:
#: Each entry is a short descriptor composed AHEAD of the user's prompt,
#: so the genre frames the request and the user's words specialise it.
#: Grounded in what each folder actually holds — measured BPM spread,
#: the spectral bands in ``agent/generative/quality_references.json``,
#: and the track names — not invented.
#:
#: Deliberately NOT mirrored into main.py: this is the generator's, and
#: a third copy of genre knowledge is exactly the drift CLAUDE.md warns
#: about. The web layer imports it from here, as it already does for
#: ``_BPM_GENRE_RANGES``.
#:
#: A genre missing from this map degrades to the bare user prompt — the
#: old behaviour — so an unlisted folder still generates.
GENRE_STYLE_PROMPTS: dict[str, str] = {
    "lofi - ambient": (
        "lo-fi ambient: warm tape saturation, soft dusty drums, mellow "
        "jazz-tinged chords, gentle vinyl crackle, unhurried and hazy"
    ),
    "lofi": (
        "lo-fi ambient: warm tape saturation, soft dusty drums, mellow "
        "jazz-tinged chords, gentle vinyl crackle, unhurried and hazy"
    ),
    "healing": (
        "healing meditation music: slow binaural drones, breathy flutes "
        "and soft chimes, long reverb tails, no percussion, deeply calm "
        "and spacious"
    ),
    "aural": (
        "ethereal beatless ambient: weightless evolving pads, submarine "
        "and cosmic textures, very long reverb, no drums, dark and "
        "spacious with slow swells"
    ),
    "synthware": (
        "retro synth electro: analog synth leads, acid bassline, crisp "
        "electro drums, glitch artefacts and tape hiss, neon and driving"
    ),
    "deep house": (
        "deep house: rolling four-on-the-floor kick, warm sub bass, "
        "smooth pad chords, subtle percussion, hypnotic late-night groove"
    ),
    "cocktail house": (
        "cocktail lounge house: laid-back nu-disco groove, brushed "
        "percussion, warm Rhodes and muted guitar, sophisticated and "
        "unhurried"
    ),
    "soul jazz": (
        "soul jazz: live drums with brushes, upright bass walking lines, "
        "Rhodes and Hammond organ, muted trumpet or sax, smoky and warm"
    ),
    "chillout": (
        "downtempo chillout: relaxed broken beat, mellow synth pads, "
        "soft bass, airy and melodic"
    ),
    "techno": (
        "techno: driving four-on-the-floor kick, hypnotic sequenced "
        "synths, industrial textures, relentless and dark"
    ),
    "cyberpunk": (
        "cyberpunk electronic: gritty synth arpeggios, distorted bass, "
        "industrial percussion, neon-noir and menacing"
    ),
}


def genre_style_prompt(genre_key: str) -> str | None:
    """Style descriptor for ``genre_key``, or ``None`` when unlisted.

    Lower-cases and trims so callers can pass a raw ``genre_folder``.
    """
    return GENRE_STYLE_PROMPTS.get((genre_key or "").strip().lower())


def _bpm_diff_bucket(diff: float) -> str:
    diff = abs(diff)
    if diff <= 5:
        return "0-5"
    if diff <= 15:
        return "6-15"
    if diff <= 30:
        return "16-30"
    return ">30"


def _transition_warning(a: dict, b: dict) -> str:
    """Return a warning string if the transition between a and b is problematic, else ''."""
    warnings = []
    bpm_a = a.get("bpm") or 0
    bpm_b = b.get("bpm") or 0
    if bpm_a > 0 and bpm_b > 0:
        ratio = max(bpm_a, bpm_b) / min(bpm_a, bpm_b)
        if ratio > 1.5:
            warnings.append(
                f"  ⚠ BPM clash: {bpm_a:.0f} → {bpm_b:.0f} ({ratio:.2f}× ratio — extreme stretch)"
            )
    steps = _camelot_step_distance(a.get("camelot_key", ""), b.get("camelot_key", ""))
    if steps > 2:
        warnings.append(
            f"  ⚠ Harmonic clash: {a.get('camelot_key','?')} → {b.get('camelot_key','?')} ({steps} Camelot steps)"
        )
    return "\n".join(warnings)


def _camelot_compat(key_a: str, key_b: str) -> str:
    """Return a human-readable compatibility label."""
    if not key_a or not key_b:
        return "unknown (missing key data)"
    if key_b in _camelot_neighbors(key_a):
        if key_a == key_b:
            return "perfect (same key)"
        return "compatible (adjacent on Camelot wheel)"
    # Check if they're 2 steps away
    second_ring = set()
    for n in _camelot_neighbors(key_a):
        second_ring |= _camelot_neighbors(n)
    if key_b in second_ring:
        return "acceptable (2 steps, semi-tone shift)"
    return "clash (unrelated keys — risky transition)"


# ---------------------------------------------------------------------------
# Selection helpers (mirrors main.py logic, path-independent)
# ---------------------------------------------------------------------------
def _bpm_cluster(tracks: list[dict]) -> list[dict]:
    if not tracks:
        return []
    sorted_tracks = sorted(tracks, key=lambda t: t.get("bpm") or 0)
    clusters: list[list[dict]] = []
    for track in sorted_tracks:
        bpm = track.get("bpm") or 0
        placed = False
        for cluster in clusters:
            median = sum(t.get("bpm") or 0 for t in cluster) / len(cluster)
            if abs(bpm - median) <= 10:
                cluster.append(track)
                placed = True
                break
        if not placed:
            clusters.append([track])
    return max(clusters, key=len)


def _harmonic_sort(tracks: list[dict]) -> list[dict]:
    if not tracks:
        return []
    pool = list(tracks)
    current = random.choice(pool)
    pool.remove(current)
    ordered = [current]
    while pool:
        neighbors = _camelot_neighbors(current.get("camelot_key", ""))
        candidates = [t for t in pool if t.get("camelot_key") in neighbors]
        next_track = random.choice(candidates) if candidates else random.choice(pool)
        pool.remove(next_track)
        ordered.append(next_track)
        current = next_track
    return ordered


# ---------------------------------------------------------------------------
# v2.5.0 — environment-aware soft bias for propose_playlist
# ---------------------------------------------------------------------------
#
# The user's listening environment is captured by the Genre Guard as a free
# text string (see ``agent/run.py::_GENRE_GUARD_SYSTEM``). We map a small set
# of keywords to three energy archetypes and reorder each BPM cluster
# accordingly. The catalog's ``tracks.json`` does NOT carry an explicit
# energy field, so we use BPM as a proxy — higher BPM ≈ more energy. This is
# imperfect (a 130 BPM ambient pad is calmer than a 120 BPM peak-time stab)
# but it is the only signal universally available across genres in this
# project, and the bias is explicitly documented as soft (it never hides a
# track the way the user-rating dislike path can — see helper below).
# ---------------------------------------------------------------------------

_ENV_KEYWORDS_HIGH_ENERGY = {
    "loud",
    "crowded",
    "club",
    "bar",
    "party",
    "warehouse",
    "festival",
}
_ENV_KEYWORDS_LOW_ENERGY = {
    "intimate",
    "listening",
    "home",
    "phones",
    "headphones",
    "quiet",
}
_ENV_KEYWORDS_MID = {
    "outdoor",
    "cafe",
    "morning",
    "background",
    "casual",
}


def _classify_environment(environment: str | None) -> str | None:
    """Classify a free-text environment description into an energy archetype.

    Returns one of: ``"high"``, ``"low"``, ``"mid"`` or ``None`` (no bias).

    Matching is case-insensitive, word-aware (we tokenize on non-alphanumeric
    boundaries), and order-deterministic: the FIRST keyword set with a hit
    wins. Empty strings, ``None``, and the sentinel ``"unspecified"`` short-
    circuit to ``None``.
    """
    if not environment:
        return None
    lowered = environment.strip().lower()
    if not lowered or lowered == "unspecified":
        return None
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if not tokens:
        return None
    if tokens & _ENV_KEYWORDS_HIGH_ENERGY:
        return "high"
    if tokens & _ENV_KEYWORDS_LOW_ENERGY:
        return "low"
    if tokens & _ENV_KEYWORDS_MID:
        return "mid"
    return None


def _apply_environment_bias(
    clusters: list[list[dict]],
    environment: str | None,
) -> list[list[dict]]:
    """Reorder each cluster as a soft signal for the user's environment.

    Pure function (no I/O, no ctx access). When ``environment`` is empty,
    ``"unspecified"``, or maps to none of the keyword sets, returns the
    input clusters unchanged (identity-preserving short-circuit).

    Energy archetypes:

    - ``"high"`` (e.g. "loud crowded bar") → within each cluster, sort
      tracks descending by BPM. Tracks above 120 BPM bubble to the front of
      their cluster.
    - ``"low"``  (e.g. "intimate listening room") → ascending by BPM.
      Tracks below 100 BPM bubble to the front.
    - ``"mid"``  (e.g. "outdoor cafe morning") → mild centered bias around
      the cluster's median BPM (tracks closer to median come first).

    BPM is a proxy for energy because ``tracks.json`` doesn't carry an
    explicit energy field — see the module-level docstring above for the
    trade-off rationale. Tracks missing BPM are treated as 0 and slot to
    the back regardless of archetype.

    Args:
        clusters: list of clusters; each cluster is a list of track dicts.
            Tracks should expose a numeric ``bpm`` key (missing/null is
            tolerated).
        environment: free-text environment description from the Genre
            Guard. ``None``, ``""``, and ``"unspecified"`` all short-circuit.

    Returns:
        New list of clusters with the bias applied (fresh outer list and
        inner cluster lists). When the bias is a no-op the input ``clusters``
        object is returned by identity so callers can detect the short-
        circuit. Track dicts are referenced, never cloned.
    """
    archetype = _classify_environment(environment)
    if archetype is None:
        return clusters

    def _bpm(t: dict) -> float:
        try:
            return float(t.get("bpm") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    result: list[list[dict]] = []
    for cluster in clusters:
        if not cluster:
            result.append([])
            continue
        if archetype == "high":
            # Descending BPM; tracks without BPM (treated as 0) drop to back.
            ordered = sorted(cluster, key=lambda t: -_bpm(t))
        elif archetype == "low":
            # Ascending BPM but a missing/zero BPM should NOT be misread as
            # the calmest track — push such tracks to the back.
            ordered = sorted(
                cluster,
                key=lambda t: (_bpm(t) <= 0, _bpm(t)),
            )
        else:  # "mid"
            bpms = [_bpm(t) for t in cluster if _bpm(t) > 0]
            if bpms:
                bpms.sort()
                median = bpms[len(bpms) // 2]
            else:
                median = 0.0
            # Tracks closer to the cluster median come first; missing BPM
            # tracks (treated as 0) sort to the back via the tuple key.
            ordered = sorted(
                cluster,
                key=lambda t: (_bpm(t) <= 0, abs(_bpm(t) - median)),
            )
        result.append(ordered)
    return result


# ---------------------------------------------------------------------------
# v2.3.1 — user-rating bias for propose_playlist
# ---------------------------------------------------------------------------

def _apply_user_rating_bias(
    clustered_tracks: list[list[dict]],
    favorite_ids: set[str] | None,
    dislike_ids: set[str] | None,
) -> list[list[dict]]:
    """Reorder each cluster so user favorites land at the front and user
    dislikes land at the back, while preserving the relative (harmonic)
    order among same-rated tracks.

    Within every cluster the ordering becomes:
        favorites (in original order) +
        unrated/neutral tracks (in original order) +
        dislikes (in original order)

    When ``favorite_ids`` and ``dislike_ids`` are both empty/None the
    function is a no-op — it returns the input cluster list untouched.

    Pure function: no side effects, no mutation of the input lists.

    Args:
        clustered_tracks: list of clusters; each cluster is a list of
            track dicts (must expose an ``id`` key).
        favorite_ids: ids the current user rated >= 4 (treated as
            favorites). ``None`` is equivalent to an empty set.
        dislike_ids: ids the current user rated <= 2 (treated as
            dislikes). ``None`` is equivalent to an empty set.

    Returns:
        New list of clusters with the bias applied. The outer list and
        inner cluster lists are fresh objects; track dicts are NOT
        cloned (they are referenced by identity).
    """
    if not favorite_ids and not dislike_ids:
        return clustered_tracks
    favorite_ids = favorite_ids or set()
    dislike_ids = dislike_ids or set()
    result: list[list[dict]] = []
    for cluster in clustered_tracks:
        favs = [t for t in cluster if t.get("id") in favorite_ids]
        dislkd = [t for t in cluster if t.get("id") in dislike_ids]
        rest = [
            t
            for t in cluster
            if t.get("id") not in favorite_ids and t.get("id") not in dislike_ids
        ]
        result.append(favs + rest + dislkd)
    return result


# ---------------------------------------------------------------------------
# Playback helper
# ---------------------------------------------------------------------------

def _play_audio(path: str, block: bool = True) -> str:
    """Play an audio file using the best available backend.

    Tries afplay (macOS), ffplay (ffmpeg), aplay (Linux ALSA) in order.
    Returns '' on success, an error string on failure.
    """
    backends = [
        ("afplay", ["afplay", path]),
        ("ffplay",  ["ffplay", "-nodisp", "-autoexit", path]),
        ("aplay",   ["aplay", path]),
    ]
    for name, cmd in backends:
        if shutil.which(name):
            try:
                if block:
                    subprocess.run(cmd, check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                else:
                    subprocess.Popen(cmd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                return ""
            except subprocess.CalledProcessError as e:
                return f"Playback error ({name}): {e}"
    return (
        "No audio player found. "
        "Install ffmpeg (provides ffplay), afplay (macOS), or aplay (Linux ALSA)."
    )


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def _load_catalog_genres() -> list[str]:
    """Return the sorted genre_folder set from tracks.json ([] if absent).

    v3.7.3 — shared by the ``list_genres`` tool and the dynamic Genre
    Guard prompt (``agent.run.genre_guard_system``) so both always speak
    from the same catalog.
    """
    if not _CATALOG_PATH.exists():
        return []
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return sorted({t["genre_folder"] for t in data["tracks"] if t.get("genre_folder")})


def list_genres(context_variables: dict) -> str:
    """List all available genre folders from the track catalog."""
    genres = _load_catalog_genres()
    if not genres:
        return "Error: tracks.json not found. Run 'python main.py --build-catalog' first."
    return "Available genres:\n" + "\n".join(f"  - {g}" for g in genres)


def get_catalog(genre: str, context_variables: dict) -> str:
    """List all tracks available for a genre with their BPM, Camelot key, and ID.

    Use the IDs returned here when calling swap_track or analyze_transition.

    Args:
        genre: Genre folder name (e.g. 'lofi - ambient', 'deep house', 'techno', 'cyberpunk').
               Pass 'current' or leave empty to use the session genre.
    """
    if not _CATALOG_PATH.exists():
        return "Error: tracks.json not found. Run 'python main.py --build-catalog' first."

    # Resolve 'current' or empty genre from session context
    if not genre or genre.lower() in ("current", "session"):
        genre = context_variables.get("genre", genre) or genre

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    genre_lower = genre.lower()
    tracks = [t for t in data["tracks"] if t["genre_folder"].lower() == genre_lower]

    if not tracks:
        available = sorted({t["genre_folder"] for t in data["tracks"]})
        return f"No tracks found for genre '{genre}'.\nAvailable genres: {', '.join(available)}"

    lines = [f"Catalog for '{genre}' — {len(tracks)} tracks:\n"]
    for t in sorted(tracks, key=lambda x: x.get("bpm") or 0):
        bpm = f"{t['bpm']:.0f} BPM" if t.get("bpm") else "? BPM"
        key = t.get("camelot_key") or "?"
        variant = f"  [variant of {t['variant_of']}]" if t.get("variant_of") else ""
        dur_sec = t.get("duration_sec")
        dur = f"  {int(dur_sec // 60)}:{int(dur_sec % 60):02d}" if dur_sec else ""
        lines.append(f"  [{t['id']}]  {t['display_name']:<30}  {bpm:<10}  {key}{dur}{variant}")

    return "\n".join(lines)


def propose_playlist(
    genre: str,
    duration_min: int,
    mood: str,
    context_variables: dict,
) -> str:
    """Generate an initial playlist using BPM clustering + harmonic sorting.

    Args:
        genre: Genre folder name
        duration_min: Target session length in minutes
        mood: Free-text description of the vibe (e.g. 'late-night peak, starts mellow')
    """
    if not _CATALOG_PATH.exists():
        return "Error: tracks.json not found. Run --build-catalog first."

    # v3.7.4 — the CONFIRMED genre in ctx is authoritative over the
    # LLM's tool argument. Small local models sometimes call this tool
    # with their prior ("lofi - ambient") instead of the genre the
    # guard just confirmed ("aural"), and this tool used to obey AND
    # overwrite ctx.genre with the wrong value — making the whole
    # session coherent with the model's whim (observed live
    # 2026-07-12: guard confirmed aural, planner built lofi anyway).
    confirmed_genre = (context_variables.get("genre") or "").strip()
    if confirmed_genre and confirmed_genre.lower() != genre.strip().lower():
        print(
            f"[propose_playlist] genre override: LLM asked for {genre!r} but "
            f"ctx has confirmed {confirmed_genre!r} — trusting the confirmed genre",
            flush=True,
        )
        genre = confirmed_genre

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    genre_lower = genre.lower()
    all_tracks = [t for t in data["tracks"] if t["genre_folder"].lower() == genre_lower]
    if not all_tracks:
        available = sorted({t["genre_folder"] for t in data["tracks"]})
        return f"No tracks for '{genre}'. Available: {', '.join(available)}"

    # v3.9.1 — session-eligibility screen: sub-2-minute pieces read as
    # cut-off tracks on stream (aural batch, observed 2026-08-03).
    eligible = filter_session_eligible(all_tracks)
    n_screened = len(all_tracks) - len(eligible)
    if n_screened:
        print(
            f"[propose_playlist] screened {n_screened} sub-minimum-duration "
            f"track(s) out of {len(all_tracks)} for '{genre}'",
            flush=True,
        )
    if not eligible:
        return (
            f"No session-eligible tracks for '{genre}' — all {len(all_tracks)} "
            "catalog entries are shorter than the minimum session duration."
        )
    all_tracks = eligible

    # v3.10 — one take per piece: never schedule 'x' and 'x-v2'/'x bis'
    # (same music, different ids) into the same session playlist.
    deduped = dedupe_takes(all_tracks)
    if len(deduped) < len(all_tracks):
        print(
            f"[propose_playlist] collapsed {len(all_tracks) - len(deduped)} "
            f"duplicate take(s) — {len(deduped)} distinct pieces for '{genre}'",
            flush=True,
        )
    all_tracks = deduped

    cluster = _bpm_cluster(all_tracks)
    ordered = _harmonic_sort(cluster)

    # v2.3.1 — sesgo por ratings: si el usuario logueado tiene
    # favoritos/dislikes en ctx (poblados por load_user_context en v2.3.0),
    # reordenamos manteniendo la adyacencia harmónica entre tracks de la
    # misma categoría. El helper opera sobre clusters, así que envolvemos
    # el orden harmónico de un solo cluster en una lista de un elemento.
    favorite_ids = context_variables.get("favorite_ids") or set()
    dislike_ids = context_variables.get("dislike_ids") or set()
    biased_clusters = _apply_user_rating_bias([ordered], favorite_ids, dislike_ids)
    ordered = biased_clusters[0] if biased_clusters else ordered

    if favorite_ids or dislike_ids:
        progress = context_variables.get("_progress")
        if progress is not None:
            n_fav = sum(1 for t in ordered if t.get("id") in favorite_ids)
            n_dis = sum(1 for t in ordered if t.get("id") in dislike_ids)
            try:
                progress({
                    "stage": "bias",
                    "message": (
                        f"Boosted {n_fav} favorites, demoted {n_dis} dislikes "
                        "within clusters."
                    ),
                })
            except Exception:
                pass  # never let UI plumbing break selection

    # v2.5.0 — environment-aware soft bias. Runs AFTER the user-rating bias
    # so the user's favorites stay at the front of the cluster regardless of
    # their BPM; the environment bias only re-orders the remainder. We split
    # the cluster on the favorite/dislike boundaries to preserve that
    # invariant.
    environment = context_variables.get("environment") or ""
    env_archetype = _classify_environment(environment)
    if env_archetype is not None:
        # Slice ordered into [favs | rest | dislikes] using the same
        # criteria as _apply_user_rating_bias. We bias only `rest` so user
        # favorites remain first and explicit dislikes remain last.
        favs = [t for t in ordered if t.get("id") in favorite_ids]
        dislkd = [t for t in ordered if t.get("id") in dislike_ids]
        rest = [
            t for t in ordered
            if t.get("id") not in favorite_ids and t.get("id") not in dislike_ids
        ]
        env_biased_rest = _apply_environment_bias([rest], environment)[0]
        ordered = favs + env_biased_rest + dislkd

        progress = context_variables.get("_progress")
        if progress is not None:
            try:
                progress({
                    "stage": "env_bias",
                    "message": (
                        f"Applied environment bias ({env_archetype}) "
                        f"based on: \"{environment}\"."
                    ),
                })
            except Exception:
                pass  # never let UI plumbing break selection

    # Fill to duration — deduplicate display_name first, then cycle
    target_sec = duration_min * 60
    seen: set[str] = set()
    first_pass = [t for t in ordered if not (t["display_name"] in seen or seen.add(t["display_name"]))]  # type: ignore[func-returns-value]

    playlist: list[dict] = []
    total_sec = 0.0
    pool = list(first_pass)
    while total_sec < target_sec:
        if not pool:
            pool = list(first_pass)
        track = pool.pop(0)
        playlist.append(track)
        total_sec += track.get("duration_sec") or 300  # fall back to 5 min if not cataloged

    context_variables["playlist"] = playlist
    context_variables["genre"] = genre
    context_variables["mood"] = mood
    # v2.5.0 — keep ``environment`` in ctx as a stable string. The Genre
    # Guard normalizes empty/missing input to "unspecified", but if a caller
    # hands us an unset ctx (CLI direct invocation, tests) we mirror that
    # convention here so downstream code never reads None.
    context_variables.setdefault(
        "environment",
        context_variables.get("environment") or "unspecified",
    )

    return _format_playlist(playlist, header=f"Proposed playlist ({len(playlist)} tracks, ~{duration_min} min) — mood: {mood}")


def show_playlist(context_variables: dict) -> str:
    """Display the current working playlist with transition analysis between every pair."""
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory yet. Use propose_playlist first."
    return _format_playlist(playlist, show_transitions=True)


def analyze_transition(track_a_id: str, track_b_id: str, context_variables: dict) -> str:
    """Analyze the harmonic and rhythmic compatibility between two tracks.

    Args:
        track_a_id: ID of the outgoing track (e.g. 'cyberpunk--akira-boulevard')
        track_b_id: ID of the incoming track
    """
    if not _CATALOG_PATH.exists():
        return "Error: catalog not found."

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    index = {t["id"]: t for t in data["tracks"]}
    a = index.get(track_a_id)
    b = index.get(track_b_id)

    if not a:
        return f"Track '{track_a_id}' not found in catalog."
    if not b:
        return f"Track '{track_b_id}' not found in catalog."

    bpm_a = a.get("bpm") or 0
    bpm_b = b.get("bpm") or 0
    bpm_diff = abs(bpm_a - bpm_b)
    ratio = max(bpm_a, bpm_b) / min(bpm_a, bpm_b) if min(bpm_a, bpm_b) > 0 else 1.0
    key_compat = _camelot_compat(a.get("camelot_key"), b.get("camelot_key"))

    if bpm_diff == 0:
        bpm_status = "identical BPM — no pitch shift needed"
    elif bpm_diff <= 5:
        bpm_status = f"{bpm_diff:.1f} BPM diff — within threshold, keep outgoing tempo"
    elif bpm_diff <= 15:
        bpm_status = f"{bpm_diff:.1f} BPM diff — meet in the middle (16s ramp)"
    else:
        bpm_status = f"{bpm_diff:.1f} BPM diff — large jump, consider a bridge track"

    result = (
        f"Transition: {a['display_name']} → {b['display_name']}\n"
        f"  BPM:      {bpm_a:.0f} → {bpm_b:.0f}  ({bpm_status})\n"
        f"  Key:      {a.get('camelot_key','?')} → {b.get('camelot_key','?')}  ({key_compat})\n"
    )
    if ratio > 1.5:
        result += f"\n  ⚠ Stretch:  pyrubberband ratio {ratio:.2f}× — recommend bridge track"
    return result


def swap_track(
    position: int,
    track_id: str,
    context_variables: dict,
    prefer_favorites: bool = True,
) -> str:
    """Replace the track at a given position (1-indexed) with another track from the catalog.

    Args:
        position: 1-indexed position in the current playlist
        track_id: ID of the replacement track (from get_catalog)
        prefer_favorites: When True (default) and the user has rated tracks,
            this is an advisory flag — the swap itself executes the
            requested track_id, but a note is appended to the response
            when the chosen track is in the user's favorite set or
            dislike set so the LLM can self-correct on the next turn.
            The actual ranking of candidates lives in suggest_bridge_track,
            which honours `prefer_favorites` for its candidate selection.
            See PR body for the full rationale.
    """
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory. Use propose_playlist first."

    if position < 1 or position > len(playlist):
        return f"Position {position} is out of range (playlist has {len(playlist)} tracks)."

    if not _CATALOG_PATH.exists():
        return "Error: catalog not found."

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    index = {t["id"]: t for t in data["tracks"]}
    new_track = index.get(track_id)
    if not new_track:
        return f"Track '{track_id}' not found. Use get_catalog to see valid IDs."

    # v3.9.1 — session-eligibility screen (agent-driven swaps only; a
    # human swapping via the web UI is an explicit override and is not
    # routed through this tool).
    reason = ineligibility_reason(new_track)
    if reason:
        return (
            f"Track '{new_track.get('display_name', track_id)}' was NOT "
            f"swapped in: {reason}. Pick a longer track from get_catalog."
        )

    old = playlist[position - 1]
    playlist[position - 1] = new_track
    context_variables["playlist"] = playlist

    # Pre-flight transition warnings for both affected seams
    pre_warnings: list[str] = []
    if position > 1:
        w = _transition_warning(playlist[position - 2], new_track)
        if w:
            pre_warnings.append(f"Position {position - 1}→{position}:\n{w}")
    if position < len(playlist):
        w = _transition_warning(new_track, playlist[position])
        if w:
            pre_warnings.append(f"Position {position}→{position + 1}:\n{w}")

    # Advisory note: surface user-rating signal for the chosen track.
    # When prefer_favorites=True and we have rating data, hint the LLM
    # whether the chosen replacement aligns with the user's preferences.
    rating_note = ""
    if prefer_favorites:
        favorite_ids = context_variables.get("favorite_ids") or set()
        dislike_ids = context_variables.get("dislike_ids") or set()
        if track_id in favorite_ids:
            rating_note = "Note: replacement is one of the user's favorites (★4+).\n\n"
        elif track_id in dislike_ids:
            rating_note = (
                "Note: replacement is in the user's dislikes (★1-2). "
                "Consider a different track if alternatives are available.\n\n"
            )

    warning_block = ("\n".join(pre_warnings) + "\n\n") if pre_warnings else ""
    return (
        rating_note
        + warning_block
        + f"Swapped position {position}:\n"
        f"  OUT: {old['display_name']} [{old.get('camelot_key','?')}  {old.get('bpm','?')} BPM]\n"
        f"  IN:  {new_track['display_name']} [{new_track.get('camelot_key','?')}  {new_track.get('bpm','?')} BPM]\n\n"
        + _format_playlist(playlist, show_transitions=True)
    )


def move_track(from_pos: int, to_pos: int, context_variables: dict) -> str:
    """Move a track from one position to another (both 1-indexed).

    Args:
        from_pos: Current position of the track to move
        to_pos: Destination position
    """
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory. Use propose_playlist first."

    n = len(playlist)
    if not (1 <= from_pos <= n and 1 <= to_pos <= n):
        return f"Positions must be between 1 and {n}."

    track = playlist.pop(from_pos - 1)
    playlist.insert(to_pos - 1, track)
    context_variables["playlist"] = playlist

    # Pre-flight transition warnings for affected seams around the new position
    dest = to_pos - 1  # 0-indexed after insert
    pre_warnings: list[str] = []
    if dest > 0:
        w = _transition_warning(playlist[dest - 1], track)
        if w:
            pre_warnings.append(f"Position {dest}→{dest + 1}:\n{w}")
    if dest < len(playlist) - 1:
        w = _transition_warning(track, playlist[dest + 1])
        if w:
            pre_warnings.append(f"Position {dest + 1}→{dest + 2}:\n{w}")

    warning_block = ("\n".join(pre_warnings) + "\n\n") if pre_warnings else ""
    return (
        warning_block
        + f"Moved '{track['display_name']}': position {from_pos} → {to_pos}\n\n"
        + _format_playlist(playlist, show_transitions=True)
    )


def suggest_bridge_track(
    from_pos: int,
    to_pos: int,
    context_variables: dict,
    prefer_favorites: bool = True,
) -> str:
    """Find candidate bridge tracks between two BPM-mismatched playlist positions.

    Args:
        from_pos: 1-indexed position of the outgoing track
        to_pos: 1-indexed position of the incoming track
        prefer_favorites: When True (default) and the user has rated tracks,
            re-rank candidates so user favorites (★4+) appear first and
            user dislikes (★1-2) drop to the bottom. The base BPM/key
            score is preserved as the tie-breaker within each group, so
            a slightly worse-scoring favorite still beats a marginally
            better-scoring unrated track. When False, falls back to the
            pure BPM/key ranking — used for regression-control tests.
    """
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory. Use propose_playlist first."

    n = len(playlist)
    if not (1 <= from_pos <= n and 1 <= to_pos <= n):
        return f"Positions must be between 1 and {n}."

    track_a = playlist[from_pos - 1]
    track_b = playlist[to_pos - 1]

    bpm_a = track_a.get("bpm")
    bpm_b = track_b.get("bpm")
    if not bpm_a:
        return f"Track at position {from_pos} is missing BPM data."
    if not bpm_b:
        return f"Track at position {to_pos} is missing BPM data."

    target_bpm = math.sqrt(bpm_a * bpm_b)

    try:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Could not load catalog: {e}"

    genre = context_variables.get("genre", "").lower()
    playlist_ids = {t.get("id") for t in playlist}

    candidates = []
    for c in catalog.get("tracks", []):
        if c.get("genre_folder", "").lower() != genre:
            continue
        if c.get("id") in playlist_ids:
            continue
        c_bpm = c.get("bpm")
        if not c_bpm:
            continue

        bpm_score = 1.0 - min(abs(c_bpm - target_bpm) / target_bpm, 1.0)
        key_dist = _camelot_step_distance(track_a.get("camelot_key", ""), c.get("camelot_key", ""))
        key_score = max(0.0, 1.0 - key_dist / 6.0)
        score = 0.7 * bpm_score + 0.3 * key_score

        candidates.append((score, c))

    # Rank: pure BPM/key score by default, or favorites-first when the
    # caller opts in (which is now the default) and the user has data.
    favorite_ids = context_variables.get("favorite_ids") or set()
    dislike_ids = context_variables.get("dislike_ids") or set()
    biased = prefer_favorites and (favorite_ids or dislike_ids)
    if biased:
        # Tier 0 = favorite, 1 = neutral, 2 = dislike. Higher score wins
        # within each tier; ties keep insertion order.
        def _tier(c: dict) -> int:
            cid = c.get("id")
            if cid in favorite_ids:
                return 0
            if cid in dislike_ids:
                return 2
            return 1

        candidates.sort(key=lambda x: (_tier(x[1]), -x[0]))
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:3]

    if not top:
        return (
            f"No bridge candidates found for genre '{genre}' "
            f"(target BPM: {target_bpm:.1f})."
        )

    lines = [
        f"Bridge candidates between position {from_pos} "
        f"({track_a['display_name']}, {bpm_a:.1f} BPM) and position {to_pos} "
        f"({track_b['display_name']}, {bpm_b:.1f} BPM) — target BPM: {target_bpm:.1f}:"
    ]
    for score, c in top:
        c_bpm = c["bpm"]
        ratio_a = max(bpm_a, c_bpm) / min(bpm_a, c_bpm)
        ratio_b = max(bpm_b, c_bpm) / min(bpm_b, c_bpm)
        marker = ""
        if biased:
            cid = c.get("id")
            if cid in favorite_ids:
                marker = " ★fav"
            elif cid in dislike_ids:
                marker = " ★dislike"
        lines.append(
            f"  {c['id']} | {c['display_name']} | {c_bpm:.1f} BPM | "
            f"{c.get('camelot_key', '?')} | "
            f"ratio_a: {ratio_a:.2f}× | ratio_b: {ratio_b:.2f}× | score: {score:.3f}{marker}"
        )
    return "\n".join(lines)


def insert_bridge_track(after_position: int, track_id: str, context_variables: dict) -> str:
    """Insert a bridge track into the playlist after the given 1-indexed position.

    Args:
        after_position: 1-indexed position after which to insert the new track
        track_id: catalog ID of the track to insert
    """
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory. Use propose_playlist first."

    n = len(playlist)
    if not (1 <= after_position <= n):
        return f"after_position must be between 1 and {n}."

    try:
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Could not load catalog: {e}"

    new_track = next((c for c in catalog.get("tracks", []) if c.get("id") == track_id), None)
    if new_track is None:
        return f"Track ID '{track_id}' not found in catalog."

    playlist.insert(after_position, new_track)
    context_variables["playlist"] = playlist

    seam_warnings: list[str] = []

    # Left seam: track before the inserted track → inserted track
    w = _transition_warning(playlist[after_position - 1], playlist[after_position])
    if w:
        seam_warnings.append(f"Left seam (position {after_position}→{after_position + 1}):\n{w}")

    # Right seam: inserted track → track after it (if exists)
    if after_position + 1 < len(playlist):
        w = _transition_warning(playlist[after_position], playlist[after_position + 1])
        if w:
            seam_warnings.append(
                f"Right seam (position {after_position + 1}→{after_position + 2}):\n{w}"
            )

    warning_block = ("\n".join(seam_warnings) + "\n\n") if seam_warnings else ""
    return (
        warning_block
        + f"Inserted '{new_track['display_name']}' at position {after_position + 1}.\n\n"
        + show_playlist(context_variables)
    )


# Maps notable stdout lines from `main.py --from-session` to short progress
# events. Order matters — first match wins. Each entry: (regex, stage, template).
# Template placeholders {1}, {2}, ... refer to capture groups.
_BUILD_PROGRESS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^=== Loading Agent Session:"), "loading_session", "Loading session config"),
    (re.compile(r"^\[(\d+)/(\d+)\] (.+?) \("), "mixing", "Mixing track {1}/{2}: {3}"),
    (re.compile(r"^Reached target"), "mix_done", "Target duration reached"),
    (re.compile(r"^Transition map:"), "mix_done", "Mix rendered — laying transition map"),
    (re.compile(r"^Exporting audio to .* \(([A-Z0-9]+),"), "export_audio", "Exporting audio ({1})"),
    (re.compile(r"^Generating artwork\.\.\."), "artwork", "Generating artwork"),
    (re.compile(r"^\s*Generating artwork for '(.+?)'"), "artwork_track", "Artwork: {1}"),
    (re.compile(r"^Generating video loops"), "video_loops", "Generating video loops"),
    (re.compile(r"^Loading artwork images"), "artwork_load", "Loading artwork images"),
    (re.compile(r"^Loading audio for waveform"), "waveform", "Analyzing waveform"),
    (re.compile(r"^Rendering video to .* \((\d+x\d+), (\d+)fps\)"), "render_video", "Rendering video ({1} @ {2}fps)"),
    (re.compile(r"^=== Audio Validation"), "validate", "Validating audio"),
]


def _parse_build_progress_line(line: str) -> dict | None:
    """Map a stdout line from the build subprocess to a progress event.

    Returns {"stage": str, "message": str} or None to skip the line.
    """
    for pattern, stage, template in _BUILD_PROGRESS_PATTERNS:
        m = pattern.search(line)
        if m:
            msg = template
            for i, g in enumerate(m.groups(), 1):
                msg = msg.replace(f"{{{i}}}", g)
            return {"stage": stage, "message": msg}
    return None


# v2.6.0 — extracted from build_session so the async render endpoint in
# `web/backend/render.py` can write the same draft + theme payload without
# invoking the sync subprocess loop below.
# Full per-genre copies, kept in sync with ``GENRE_THEMES`` in main.py — the
# canonical table, which also carries each palette's design rationale. Full
# matters: this dict lands in the draft session.json's "theme" block, which
# main.py's _get_session_theme applies as its TOP layer, above main.py's own
# genre defaults — so a value that drifts here doesn't merely degrade, it
# silently overrides the canonical theme at render time.
GENRE_THEMES: dict[str, dict] = {
    "lofi - ambient": {
        "artwork_style": "anime",
        "title_color": "#E8D5B7",
        "title_stroke_color": "#5C4A32",
        "bg_color": [18, 15, 12],
        "waveform_color": [180, 160, 130],
        "particle_color": [200, 180, 150],
        "bg_darken": 0.85,
        "title_font_size": 36,
    },
    "lofi": {
        "artwork_style": "anime",
        "title_color": "#E8D5B7",
        "title_stroke_color": "#5C4A32",
        "bg_color": [18, 15, 12],
        "waveform_color": [180, 160, 130],
        "particle_color": [200, 180, 150],
        "bg_darken": 0.85,
        "title_font_size": 36,
    },
    "deep house": {
        "artwork_style": "deep-house-neon",
        "title_color": "#6A5AFF",
        "title_stroke_color": "#1A0A3E",
        "bg_color": [12, 8, 28],
        "waveform_color": [106, 90, 255],
        "particle_color": [140, 120, 255],
        "bg_darken": 0.7,
        "title_font_size": 32,
    },
    "techno": {
        "artwork_style": "dark-techno",
        "title_color": "#FF1744",
        "title_stroke_color": "#4A0010",
        "bg_color": [5, 2, 8],
        "waveform_color": [255, 23, 68],
        "particle_color": [255, 50, 80],
        "bg_darken": 0.85,
        "title_font_size": 32,
    },
    "cyberpunk": {
        "artwork_style": "dark-techno",
        "title_color": "#00FF88",
        "title_stroke_color": "#004422",
        "bg_color": [8, 8, 14],
        "waveform_color": [0, 255, 136],
        "particle_color": [0, 200, 100],
        "bg_darken": 0.75,
        "title_font_size": 32,
    },
    "cocktail house": {
        "artwork_style": "deep-house-neon",
        "title_color": "#E8B86C",
        "title_stroke_color": "#3A1F1A",
        "bg_color": [22, 10, 16],
        "waveform_color": [232, 184, 108],
        "particle_color": [255, 210, 140],
        "bg_darken": 0.75,
        "title_font_size": 32,
    },
    "soul jazz": {
        "artwork_style": "organic-zen",
        "title_color": "#D98E3B",
        "title_stroke_color": "#2A140A",
        "bg_color": [20, 12, 8],
        "waveform_color": [217, 142, 59],
        "particle_color": [240, 180, 100],
        "bg_darken": 0.8,
        "title_font_size": 32,
    },
    "chillout": {
        "artwork_style": "organic-zen",
        "title_color": "#DCE7E3",
        "title_stroke_color": "#2F4440",
        "bg_color": [14, 20, 19],
        "waveform_color": [150, 190, 180],
        "particle_color": [180, 215, 205],
        "bg_darken": 0.85,
        "title_font_size": 36,
    },
    # Mirrors main.GENRE_THEMES exactly — _get_session_theme applies THIS
    # copy as the top layer at render time, so a drifted value here does
    # not degrade, it silently overrides the canonical theme.
    "aural": {
        "artwork_style": "abstract",
        "title_color": "#8FD8F0",
        "title_stroke_color": "#07202C",
        "bg_color": [6, 16, 26],
        "waveform_color": [143, 216, 240],
        "particle_color": [190, 232, 248],
        "bg_darken": 0.8,
    },
    "synthware": {
        "artwork_style": "dark-techno",
        "title_color": "#F45BD0",
        "title_stroke_color": "#1A0322",
        "bg_color": [18, 4, 24],
        "waveform_color": [244, 91, 208],
        "particle_color": [120, 240, 232],
        "bg_darken": 0.4,
    },
    "healing": {
        "artwork_style": "healing-aura",
        "title_color": "#9FE0D0",
        "title_stroke_color": "#0C2A2A",
        "bg_color": [8, 18, 22],
        "waveform_color": [159, 224, 208],
        "particle_color": [200, 240, 230],
        "bg_darken": 0.85,
        "video_bg_darken": 0.45,
        "title_font_size": 32,
    },
}


def _write_draft_session(session_name: str, context_variables: dict) -> Path:
    """Write the draft ``session.json`` that ``main.py --from-session`` reads.

    Returns the path to the draft file. Raises ``ValueError`` if the
    context is missing playlist or genre.
    """
    playlist = context_variables.get("playlist")
    genre = context_variables.get("genre")
    if not playlist:
        raise ValueError("No playlist in memory. Use propose_playlist first.")
    if not genre:
        raise ValueError("Genre not set in context. Run propose_playlist first.")

    slug = _slugify(session_name)
    draft_path = _PROJECT_DIR / "output" / f"_draft_{slug}" / "session.json"
    draft_path.parent.mkdir(parents=True, exist_ok=True)

    session_config = {
        "name": slug,
        "genre": genre,
        "theme": GENRE_THEMES.get(genre.lower(), {}),
        "playlist": [
            {
                "display_name": t["display_name"],
                "file": t["file"],
                "camelot_key": t.get("camelot_key"),
                "genre": t.get("genre"),
            }
            for t in playlist
        ],
    }

    with open(draft_path, "w", encoding="utf-8") as f:
        json.dump(session_config, f, indent=2)
    return draft_path


def build_session(session_name: str, context_variables: dict) -> str:
    """Save the current playlist as a draft and trigger the full mix + video pipeline.

    Args:
        session_name: Name for the output folder (e.g. 'midnight-techno')
    """
    session_name = _slugify(session_name)
    playlist = context_variables.get("playlist")
    genre = context_variables.get("genre")

    if not playlist:
        return "No playlist in memory. Use propose_playlist first."
    if not genre:
        return "Genre not set in context. Run propose_playlist first."

    try:
        draft_path = _write_draft_session(session_name, context_variables)
    except ValueError as exc:
        return str(exc)

    # Kick off main.py --from-session
    cmd = [
        sys.executable,
        str(_MAIN_PY),
        "--from-session", str(draft_path),
        "--name", session_name,
        "--genre", genre,
    ]

    print(f"\n[Agent] Launching pipeline: {' '.join(cmd)}\n")

    # When a progress callback is present (web path), tail the subprocess
    # stdout so the UI can show per-stage updates instead of a long silence.
    # CLI callers leave _progress unset, so the original `subprocess.run`
    # behaviour is preserved.
    progress = context_variables.get("_progress")
    if progress is not None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            print(line)
            event = _parse_build_progress_line(line)
            if event:
                try:
                    progress(event)
                except Exception:
                    pass  # never let UI plumbing break the build
        returncode = proc.wait()
    else:
        returncode = subprocess.run(cmd, cwd=str(_PROJECT_DIR)).returncode

    if returncode == 0:
        context_variables["last_build"] = session_name
        return f"Build complete! Output → output/{session_name}/"
    else:
        return f"Pipeline exited with code {returncode}. Check the output above for errors."


# ---------------------------------------------------------------------------
# Internal formatter
# ---------------------------------------------------------------------------

def _format_playlist(
    playlist: list[dict],
    header: str = "",
    show_transitions: bool = False,
) -> str:
    lines = []
    if header:
        lines.append(header)
        lines.append("")

    for i, t in enumerate(playlist, 1):
        bpm = f"{t['bpm']:.0f}" if t.get("bpm") else "?"
        key = t.get("camelot_key") or "?"
        track_id = t.get("id", "")
        id_str = f"  [{track_id}]" if track_id else ""
        lines.append(f"  {i:2d}. {t['display_name']:<30}  {bpm} BPM  [{key}]{id_str}")

        if show_transitions and i < len(playlist):
            nxt = playlist[i]
            compat = _camelot_compat(t.get("camelot_key"), nxt.get("camelot_key"))
            bpm_a = t.get("bpm") or 0
            bpm_b = nxt.get("bpm") or 0
            diff = abs(bpm_a - bpm_b)
            flag = " ⚠" if "clash" in compat or diff > 15 else ""
            lines.append(f"       ↓ {compat}  |  Δ{diff:.0f} BPM{flag}")

    return "\n".join(lines)


def catalog_status(context_variables: dict) -> str:
    """Compare audio files on disk vs tracks.json and report what is missing or orphaned.

    Shows per-genre: how many files on disk, how many in catalog, which are new (not yet cataloged),
    and which catalog entries have missing files (orphaned). Accepts both WAV and MP3 inputs.
    """
    import os as _os

    tracks_dir = _PROJECT_DIR / "tracks"
    if not tracks_dir.exists():
        return "Error: tracks/ folder not found."

    # Load catalog
    cataloged: dict[str, dict] = {}
    if _CATALOG_PATH.exists():
        with open(_CATALOG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("tracks", []):
            cataloged[entry["file"]] = entry

    # Scan folders — accept the same extensions the catalog scanner does.
    _AUDIO_EXTS = (".wav", ".mp3")
    disk_files: dict[str, list[str]] = {}  # genre_folder → [rel_path, ...]
    for folder in sorted(_os.listdir(tracks_dir)):
        folder_path = tracks_dir / folder
        if not folder_path.is_dir():
            continue
        audio = sorted(
            str((folder_path / f).relative_to(_PROJECT_DIR)).replace("\\", "/")
            for f in _os.listdir(folder_path)
            if f.lower().endswith(_AUDIO_EXTS)
        )
        if audio:
            disk_files[folder] = audio

    if not disk_files:
        return "No audio files (WAV/MP3) found in any tracks/ subfolder."

    # Find incomplete entries — any required field missing
    def _missing_fields(e: dict) -> list[str]:
        missing = []
        if not e.get("id"):
            missing.append("id")
        if not e.get("bpm"):
            missing.append("bpm")
        if not e.get("camelot_key"):
            missing.append("camelot_key")
        if not e.get("genre"):
            missing.append("genre")
        if not e.get("genre_folder"):
            missing.append("genre_folder")
        if e.get("duration_sec") is None:
            missing.append("duration_sec")
        return missing

    incomplete_entries = [
        e for e in cataloged.values()
        if _missing_fields(e)
    ]

    lines = []
    total_new = 0
    total_orphaned = 0

    for genre, paths in disk_files.items():
        cataloged_in_genre = {p for p in cataloged if p.startswith(f"tracks/{genre}/")}
        new = [p for p in paths if p not in cataloged]
        orphaned = [p for p in cataloged_in_genre if p not in paths]
        incomplete_in_genre = [
            e for e in incomplete_entries
            if e.get("genre_folder", "") == genre
        ]

        lines.append(f"\n{genre}:")
        lines.append(f"  On disk: {len(paths)}  |  In catalog: {len(cataloged_in_genre)}")
        if new:
            lines.append("  NEW (not in catalog):")
            for p in new:
                lines.append(f"    + {_os.path.basename(p)}")
        if orphaned:
            lines.append("  ORPHANED (in catalog, file missing):")
            for p in orphaned:
                lines.append(f"    - {_os.path.basename(p)}")
        if incomplete_in_genre:
            lines.append("  INCOMPLETE (missing required fields):")
            for e in incomplete_in_genre:
                missing = _missing_fields(e)
                name = e.get("display_name") or e.get("file", "?")
                lines.append(f"    ! {name} — missing: {', '.join(missing)}")
        if not new and not orphaned and not incomplete_in_genre:
            lines.append("  ✓ In sync")
        total_new += len(new)
        total_orphaned += len(orphaned)

    summary_parts = []
    if total_new:
        summary_parts.append(f"{total_new} new file(s) → run rebuild_catalog")
    if total_orphaned:
        summary_parts.append(f"{total_orphaned} orphaned entry(ies)")
    if incomplete_entries:
        summary_parts.append(f"{len(incomplete_entries)} incomplete entry(ies) → run fix_incomplete")
    summary = "\nSummary: " + (", ".join(summary_parts) if summary_parts else "everything in sync ✓")
    return "CATALOG STATUS" + "".join(lines) + summary


def fix_incomplete(context_variables: dict) -> str:
    """Re-analyse catalog entries that have missing BPM or Camelot key and update tracks.json.

    Only re-processes existing entries with null/missing fields — does not add new files.
    Use rebuild_catalog to add new files first.
    """
    cmd = [sys.executable, str(_MAIN_PY), "--fix-incomplete"]
    print(f"\n[Catalog Manager] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_PROJECT_DIR), capture_output=False)

    if result.returncode == 0:
        return "Incomplete entries re-analysed and updated in tracks.json."
    return f"Fix failed (exit code {result.returncode}). Check output above."


def redetect_bpm(genre: str, context_variables: dict) -> str:
    """Re-detect BPM for catalog tracks using the current detection algorithm.

    Use when BPM values look wrong (e.g. all showing 110 due to double-time detection).
    Rewrites bpm fields in tracks.json.

    Args:
        genre: Genre folder name to re-process, or 'all' to re-process every genre.
    """
    cmd = [sys.executable, str(_MAIN_PY), "--redetect-bpm"]
    if genre.lower() != "all":
        cmd += ["--genre", genre]
    print(f"\n[Catalog Manager] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_PROJECT_DIR), capture_output=False)

    scope = "all genres" if genre.lower() == "all" else f"'{genre}'"
    if result.returncode == 0:
        return f"BPM re-detected for {scope}. Check output above for changes."
    return f"Re-detection failed (exit code {result.returncode}). Check output above."


def generate_beatgrid(genre: str, context_variables: dict) -> str:
    """Generate beatgrid (first beat position + confirmed BPM) for catalog tracks missing it.

    Safe to run repeatedly — skips entries that already have a beatgrid.
    Beatgrid data is used by the LiveDJ engine for beat-accurate crossfades.

    Args:
        genre: Genre folder name to process, or 'all' to process every genre.
    """
    cmd = [sys.executable, str(_MAIN_PY), "--generate-beatgrid"]
    if genre.lower() != "all":
        cmd += ["--genre", genre]
    print(f"\n[Catalog Manager] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_PROJECT_DIR), capture_output=False)

    scope = "all genres" if genre.lower() == "all" else f"'{genre}'"
    if result.returncode == 0:
        return f"Beatgrid generated for {scope}. LiveDJ will now use beat-accurate crossfade points."
    return f"Beatgrid generation failed (exit code {result.returncode}). Check output above."


def rebuild_catalog(context_variables: dict) -> str:
    """Scan all genre folders and add any new WAV files to tracks.json.

    Detects BPM and Camelot key for each new file. Existing entries are not re-processed.
    This may take a few minutes depending on how many new files need analysis.
    """
    cmd = [sys.executable, str(_MAIN_PY), "--build-catalog"]
    print(f"\n[Catalog Manager] Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(_PROJECT_DIR), capture_output=False)

    if result.returncode == 0:
        # Read catalog to report how many total tracks now
        if _CATALOG_PATH.exists():
            with open(_CATALOG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            total = len(data.get("tracks", []))
            return f"Catalog updated successfully. Total tracks in catalog: {total}."
        return "Catalog updated successfully."
    return f"Catalog build failed (exit code {result.returncode}). Check output above."


def validate_audio(session_name: str, context_variables: dict) -> str:
    """Analyze the exported mix WAV for audio quality issues.

    Args:
        session_name: The output folder name used during build (e.g. 'midnight-techno')
    """
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    wav_path = _PROJECT_DIR / "output" / session_name / "mix_output.wav"
    if not wav_path.exists():
        # Also try mix.wav in case path differs
        wav_path = _PROJECT_DIR / "output" / session_name / "mix.wav"
    if not wav_path.exists():
        return (
            f"Error: mix WAV not found under output/{session_name}/. "
            "Did build_session complete successfully?"
        )

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration_sec = len(y) / sr
    issues: list[str] = []

    # 1. Peak clipping
    clip_mask = np.abs(y) >= 0.98
    if clip_mask.any():
        first_clip = float(np.where(clip_mask)[0][0]) / sr
        m, s = divmod(int(first_clip), 60)
        pct = 100.0 * clip_mask.sum() / len(y)
        issues.append(f"[{m:02d}:{s:02d}] Peak clipping — {pct:.2f}% of samples ≥ 0.98 FS")

    # 2. Spectral flatness per 30s window (bleaching/noise detection)
    window_samples = 30 * sr
    n_windows = max(1, len(y) // window_samples)
    for w in range(n_windows):
        chunk = y[w * window_samples: (w + 1) * window_samples]
        if len(chunk) < sr:
            continue
        flatness = librosa.feature.spectral_flatness(y=chunk)
        mean_flat = float(np.mean(flatness))
        if mean_flat > 0.4:
            m, s = divmod(w * 30, 60)
            issues.append(
                f"[{m:02d}:{s:02d}] High spectral flatness ({mean_flat:.2f}) — "
                "possible noise or bleached track in this 30s window"
            )

    # 3. Silence gaps > 2s
    hop = int(sr * 0.1)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    in_gap = False
    gap_start = 0.0
    for fi, val in enumerate(rms):
        t = fi * hop / sr
        if val < 0.005 and not in_gap:
            in_gap = True
            gap_start = t
        elif val >= 0.005 and in_gap:
            in_gap = False
            gap_dur = t - gap_start
            if gap_dur > 2.0:
                m, s = divmod(int(gap_start), 60)
                issues.append(f"[{m:02d}:{s:02d}] Silence gap of {gap_dur:.1f}s — possible dropout")

    # 4. RMS anomalies — sudden large drops (>12dB) between adjacent 30s windows
    rms_windows = []
    for w in range(n_windows):
        chunk = y[w * window_samples: (w + 1) * window_samples]
        rms_windows.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)
    for w in range(1, len(rms_windows)):
        if rms_windows[w - 1] > 1e-6 and rms_windows[w] > 1e-6:
            ratio_db = 20 * np.log10(rms_windows[w] / rms_windows[w - 1])
            if ratio_db < -12:
                m, s = divmod(w * 30, 60)
                issues.append(
                    f"[{m:02d}:{s:02d}] Sudden RMS drop of {abs(ratio_db):.1f}dB — "
                    "possible bleached or silent section"
                )

    dur_str = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
    header = f"AUDIO QUALITY REPORT — {session_name}\nDuration: {dur_str} | Sample rate: {sr} Hz\n"
    if not issues:
        return header + "Status: PASS\nNo issues detected."
    return header + f"Issues ({len(issues)}):\n" + "\n".join(f"  - {i}" for i in issues)


# ---------------------------------------------------------------------------
# Tool registry — used by agent/run.py to build Anthropic tool schemas
# ---------------------------------------------------------------------------

def read_memory(genre: str, context_variables: dict) -> str:
    """Return a formatted summary of past sessions for a given genre to inform planning.

    Args:
        genre: Genre folder name to filter sessions by (e.g. 'techno')
    """
    if not _MEMORY_PATH.exists():
        return "No memory yet for this genre."

    with open(_MEMORY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    sessions = [
        s for s in data.get("sessions", [])
        if s.get("genre", "").lower() == genre.lower()
    ]
    if not sessions:
        return f"No memory yet for genre '{genre}'."

    recent = sessions[-10:]  # last 10 matching sessions

    # Tracks swapped ≥2× → avoid
    swap_counts: dict[str, int] = {}
    for s in recent:
        for t in s.get("tracks_swapped", []):
            swap_counts[t] = swap_counts.get(t, 0) + 1
    avoid = sorted((t for t, c in swap_counts.items() if c >= 2), key=lambda t: -swap_counts[t])

    # High-rated sessions (rating ≥ 4)
    high_rated = [s for s in recent if s.get("rating", 0) >= 4]

    # Recurring critic problem patterns
    problem_counts: dict[str, int] = {}
    for s in recent:
        for p in s.get("critic_problems", []):
            # Use first 40 chars as key to group similar problems
            key = p[:40]
            problem_counts[key] = problem_counts.get(key, 0) + 1
    recurring = sorted(
        ((k, c) for k, c in problem_counts.items() if c >= 2),
        key=lambda x: -x[1]
    )

    lines = [f"MEMORY SUMMARY — {genre} ({len(recent)} past sessions)\n"]

    if avoid:
        lines.append("Tracks swapped out ≥2× (avoid in new set):")
        for t in avoid:
            lines.append(f"  - {t} (swapped {swap_counts[t]}×)")
        lines.append("")

    if high_rated:
        lines.append("High-rated sessions (rating ≥ 4):")
        for s in high_rated[-5:]:
            lines.append(
                f"  - \"{s.get('mood', '?')}\" | "
                f"{len(s.get('final_playlist', []))} tracks | "
                f"Critic: {s.get('critic_verdict', '?')} | "
                f"Validator: {s.get('validator_status', '?')} | "
                f"Rating: {s.get('rating', '?')}/5"
            )
        lines.append("")

    if recurring:
        lines.append("Recurring Critic patterns (appeared in 2+ sessions):")
        for pattern, count in recurring[:5]:
            lines.append(f"  - \"{pattern}...\" ({count}×)")
        lines.append("")

    # Transition ratings — aggregated by key_pair
    tr_ratings: dict[str, list[int]] = {}
    for s in recent:
        for tr in s.get("transition_ratings", []):
            kp = tr.get("key_pair", "")
            r = tr.get("rating", 0)
            if kp and 1 <= r <= 5:
                tr_ratings.setdefault(kp, []).append(r)

    proven_transitions = [(kp, sum(rs)/len(rs)) for kp, rs in tr_ratings.items() if sum(rs)/len(rs) >= 4.0 and len(rs) >= 2]
    avoid_transitions = [(kp, sum(rs)/len(rs)) for kp, rs in tr_ratings.items() if sum(rs)/len(rs) < 3.0 and len(rs) >= 2]

    if proven_transitions:
        lines.append("Proven transition key pairs (mean rating ≥ 4):")
        for kp, mean in sorted(proven_transitions, key=lambda x: -x[1])[:5]:
            lines.append(f"  ✓ {kp} (avg {mean:.1f}/5, {len(tr_ratings[kp])} samples)")
        lines.append("")

    if avoid_transitions:
        lines.append("Weak transition key pairs (mean rating < 3) — avoid if possible:")
        for kp, mean in sorted(avoid_transitions, key=lambda x: x[1])[:5]:
            lines.append(f"  ✗ {kp} (avg {mean:.1f}/5, {len(tr_ratings[kp])} samples)")
        lines.append("")

    # Structured problems — recurring key-pair clashes
    clash_counts: dict[str, int] = {}
    for s in recent:
        for sp in s.get("structured_problems", []):
            kp = sp.get("key_pair", "")
            if kp:
                clash_counts[kp] = clash_counts.get(kp, 0) + 1

    recurring_clashes = [(kp, c) for kp, c in clash_counts.items() if c >= 2]
    if recurring_clashes:
        lines.append("Recurring harmonic clashes (key pairs flagged in 2+ sessions):")
        for kp, count in sorted(recurring_clashes, key=lambda x: -x[1])[:5]:
            lines.append(f"  ⚠ {kp} ({count} sessions)")
        lines.append("")

    return "\n".join(lines)


def write_session_record(
    session_name: str,
    genre: str,
    duration_min: int,
    mood: str,
    rating: int,
    notes: str,
    critic_verdict: str,
    critic_problems_json: str,
    validator_status: str,
    validator_issues_json: str,
    tracks_swapped_json: str,
    final_playlist_json: str,
    transition_ratings_json: str,
    structured_problems_json: str,
    context_variables: dict,
) -> str:
    """Append a completed session record to memory.json (capped at 50 sessions).

    Args:
        session_name: Name of the built session
        genre: Genre of the session
        duration_min: Target duration in minutes
        mood: Mood description used for planning
        rating: User rating 1-5 (0 = skipped)
        notes: Optional user notes
        critic_verdict: APPROVED / NEEDS_FIXES / REJECT
        critic_problems_json: JSON array string of critic problem strings
        validator_status: PASS / WARNING / FAIL
        validator_issues_json: JSON array string of validator issue strings
        tracks_swapped_json: JSON array string of display_names removed during editing
        final_playlist_json: JSON array string of final display_names in order
        transition_ratings_json: JSON array of {from,to,key_pair,bpm_diff_bucket,rating} dicts
        structured_problems_json: JSON array of {pos_from,pos_to,key_pair,bpm_diff,text} dicts
    """
    from datetime import datetime  # noqa: PLC0415

    if _MEMORY_PATH.exists():
        with open(_MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"schema_version": 2, "sessions": []}

    data["schema_version"] = 2

    record = {
        "session_name": session_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "genre": genre,
        "duration_min": duration_min,
        "mood": mood,
        "rating": rating,
        "notes": notes,
        "critic_verdict": critic_verdict,
        "critic_problems": json.loads(critic_problems_json or "[]"),
        "validator_status": validator_status,
        "validator_issues": json.loads(validator_issues_json or "[]"),
        "tracks_swapped": json.loads(tracks_swapped_json or "[]"),
        "final_playlist": json.loads(final_playlist_json or "[]"),
        "transition_ratings": json.loads(transition_ratings_json or "[]"),
        "structured_problems": json.loads(structured_problems_json or "[]"),
    }

    sessions = data.get("sessions", [])
    sessions.append(record)
    sessions = sessions[-50:]  # cap at 50
    data["sessions"] = sessions

    # Atomic write via temp file
    tmp = _MEMORY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _MEMORY_PATH)

    return f"Session '{session_name}' saved to memory ({len(sessions)} total records)."


def get_energy_arc(context_variables: dict) -> str:
    """Return a structured energy arc table for the current playlist with plateau/peak/release analysis.

    No arguments needed — reads the current playlist from session context.
    """
    playlist = context_variables.get("playlist", [])
    if not playlist:
        return "No playlist in context yet. Call propose_playlist first."

    genre = context_variables.get("genre", "")
    lo, hi = _BPM_GENRE_RANGES.get(genre.lower(), (60, 200))

    def _energy(track: dict) -> float:
        bpm = track.get("bpm") or ((lo + hi) / 2)
        key = track.get("camelot_key", "")
        bpm_range = max(hi - lo, 1)
        e = (float(bpm) - lo) / bpm_range * 10
        if key and len(key) >= 2:
            try:
                num = int(key[:-1])
                if 7 <= num <= 12:
                    e = min(10.0, e + 1)
            except ValueError:
                pass
        return round(max(0.0, min(10.0, e)), 1)

    energies = [_energy(t) for t in playlist]
    n = len(playlist)

    lines = ["POS | TRACK                          | BPM   | KEY  | ENERGY"]
    lines.append("─" * 62)
    for i, (t, e) in enumerate(zip(playlist, energies)):
        name = (t.get("display_name") or "?")[:30]
        bpm = t.get("bpm", "?")
        key = t.get("camelot_key", "?")
        bar = "█" * int(e)
        lines.append(f"{i+1:>3} | {name:<30} | {str(bpm):>5} | {key:<4} | {bar} {e}")

    lines.append("")

    issues = []

    # Plateau: 3+ consecutive tracks within ±1 energy — report once per plateau
    run_start = 0
    for i in range(1, n + 1):
        broke = i == n or abs(energies[i] - energies[i - 1]) > 1.0
        if broke:
            run_len = i - run_start
            if run_len >= 3:
                issues.append(
                    f"Plateau at positions {run_start+1}–{i} "
                    f"(energy {energies[run_start]}–{energies[i-1]})"
                )
            if i < n:
                run_start = i

    # Missing peak: no track energy >= 7
    if all(e < 7.0 for e in energies):
        issues.append("No peak: no track reaches energy ≥ 7 — set may feel flat")

    # Missing release: final 20% should trend downward
    tail_start = max(0, int(n * 0.8))
    tail = energies[tail_start:]
    if len(tail) >= 2 and tail[-1] >= tail[0]:
        issues.append(
            f"No wind-down: final {len(tail)} tracks don't drop in energy "
            f"({tail[0]} → {tail[-1]})"
        )

    if issues:
        lines.append("Arc issues detected:")
        for issue in issues:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("Arc: peak and release present, no long plateaus — looks good.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v1.4 — Live Local Playback
# ---------------------------------------------------------------------------

def play_mix(session_name: str, context_variables: dict) -> str:
    """Stream mix_output.wav for a built session (non-blocking background playback).

    Args:
        session_name: Session name, or "" to use the most recently built session.
    """
    if not session_name:
        session_name = context_variables.get("last_build", "")
    if not session_name:
        return "No session name provided and no recent build found in context."
    wav = _PROJECT_DIR / "output" / session_name / "mix_output.wav"
    if not wav.exists():
        return f"mix_output.wav not found for session '{session_name}'."
    err = _play_audio(str(wav), block=False)
    if err:
        return err
    return f"▶ Playing '{session_name}' in the background."


def preview_transition(pos_a: int, pos_b: int, session_name: str,
                       context_variables: dict) -> str:
    """Extract and play the ±15 s crossfade zone between two adjacent tracks.

    Requires a rendered mix (build_session must have been called first).

    Args:
        pos_a: 1-indexed position of the outgoing track.
        pos_b: 1-indexed position of the incoming track.
        session_name: Session name, or "" to use the most recently built session.
    """
    if not session_name:
        session_name = context_variables.get("last_build", "")
    if not session_name:
        return "No session name provided and no recent build found in context."

    out_dir = _PROJECT_DIR / "output" / session_name
    wav_path = out_dir / "mix_output.wav"
    tj_path  = out_dir / "transitions.json"

    if not wav_path.exists():
        return f"mix_output.wav not found for session '{session_name}'."
    if not tj_path.exists():
        return f"transitions.json not found for session '{session_name}'."

    with open(tj_path, encoding="utf-8") as f:
        transitions = json.load(f)

    # transitions[i] is the entry for the track at 1-indexed playlist position i+1
    idx = pos_b - 1
    if idx < 0 or idx >= len(transitions):
        return (
            f"pos_b={pos_b} is out of range "
            f"(session has {len(transitions)} tracks)."
        )

    entry  = transitions[idx]
    t_sec  = entry["start_sec"]
    name   = entry["name"]

    WINDOW   = 15  # seconds either side of the crossfade point
    start_ms = max(0, int((t_sec - WINDOW) * 1000))
    end_ms   = int((t_sec + WINDOW) * 1000)

    audio  = AudioSegment.from_file(str(wav_path))
    end_ms = min(end_ms, len(audio))
    clip   = audio[start_ms:end_ms]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        clip.export(tmp.name, format="wav")
        tmp.close()
        err = _play_audio(tmp.name, block=True)
    finally:
        os.unlink(tmp.name)

    if err:
        return err
    clip_dur = (end_ms - start_ms) / 1000
    return (
        f"Previewed transition → '{name}' "
        f"(crossfade at {t_sec:.1f}s, {clip_dur:.0f}s clip played)."
    )


def play_track(track_id: str, start_sec: int, duration_sec: int,
               context_variables: dict) -> str:
    """Audition an individual track from the catalog.

    Args:
        track_id: Catalog track ID (use get_catalog to find IDs).
        start_sec: Start offset in seconds (0 = beginning of track).
        duration_sec: How many seconds to play (0 = play full track).
    """
    if not _CATALOG_PATH.exists():
        return "Error: tracks.json not found."
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    index = {t["id"]: t for t in data["tracks"]}
    track = index.get(track_id)
    if not track:
        return f"Track ID '{track_id}' not found in catalog."

    rel      = track["file"]
    abs_path = (_PROJECT_DIR / rel) if not os.path.isabs(rel) else Path(rel)
    if not abs_path.exists():
        return f"Audio file not found on disk: {abs_path}"

    if start_sec > 0 or duration_sec > 0:
        audio    = AudioSegment.from_file(str(abs_path))
        start_ms = int(start_sec * 1000)
        end_ms   = (start_ms + int(duration_sec * 1000)) if duration_sec > 0 else len(audio)
        end_ms   = min(end_ms, len(audio))
        clip     = audio[start_ms:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            clip.export(tmp.name, format="wav")
            tmp.close()
            err = _play_audio(tmp.name, block=True)
        finally:
            os.unlink(tmp.name)
    else:
        err = _play_audio(str(abs_path), block=True)

    if err:
        return err
    label = f"{start_sec}s–{start_sec + duration_sec}s" if duration_sec > 0 else "full track"
    return f"▶ Played '{track['display_name']}' ({label})."


# ---------------------------------------------------------------------------
# v1.5 — Live DJ session tools
# ---------------------------------------------------------------------------

def start_live_session(session_name: str, context_variables: dict) -> str:
    """Start a live DJ session with the current playlist (or load a saved session).

    Launches the LiveDJ engine and enters the event loop — blocks until the
    session ends or the user quits.

    Args:
        session_name: Session name to load playlist from disk, or "" to use
                      the playlist currently in context.
    """
    # Deferred import to break circular dependency (run.py → tools → live_dj → run)
    from agent.live_dj import run_live_session  # noqa: PLC0415

    playlist = context_variables.get("playlist", [])

    if not playlist and session_name:
        session_file = _PROJECT_DIR / "output" / session_name / "session.json"
        if not session_file.exists():
            return f"No playlist in context and session '{session_name}' not found on disk."
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
        # Enrich playlist entries with catalog metadata (bpm, camelot_key, id)
        catalog: dict[str, dict] = {}
        if _CATALOG_PATH.exists():
            with open(_CATALOG_PATH, encoding="utf-8") as f:
                cat_data = json.load(f)
            catalog = {t["file"]: t for t in cat_data.get("tracks", [])}
        playlist = []
        for entry in data.get("playlist", []):
            merged = {**catalog.get(entry.get("file", ""), {}), **entry}
            playlist.append(merged)

    if not playlist:
        return "No playlist available. Build a set first or provide a session_name."

    run_live_session(playlist, context_variables)
    return "Live session ended."


def import_rekordbox(xml_path: str, context_variables: dict) -> str:
    """Import hot cues and beatgrid from a Rekordbox XML export into tracks.json.

    Matches tracks by filename. Writes hot_cues and beatgrid fields into
    matching catalog entries and saves tracks.json.

    Args:
        xml_path: Absolute path to the rekordbox.xml export file.
    """
    try:
        import pyrekordbox.xml as rb_xml  # noqa: PLC0415
    except ImportError:
        return "pyrekordbox is not installed. Run: uv add pyrekordbox"

    xml_file = Path(xml_path)
    if not xml_file.exists():
        return f"File not found: {xml_path}"
    if not _CATALOG_PATH.exists():
        return "tracks.json not found — run --build-catalog first."

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        catalog_data = json.load(f)

    tracks = catalog_data.get("tracks", [])
    # Index catalog by filename (last component) for fuzzy matching
    catalog_by_name: dict[str, dict] = {
        Path(t["file"]).name.lower(): t for t in tracks
    }

    try:
        rb = rb_xml.RekordboxXml(xml_path)
        rb_tracks = list(rb.get_all_tracks())
    except Exception as e:
        return f"Failed to parse Rekordbox XML: {e}"

    updated = 0
    unmatched = []
    for rb_track in rb_tracks:
        filename = Path(rb_track.Location).name.lower()
        cat_entry = catalog_by_name.get(filename)
        if not cat_entry:
            unmatched.append(filename)
            continue

        # Hot cues
        hot_cues = []
        for cue in getattr(rb_track, "marks", []):
            cue_type = "in" if getattr(cue, "Type", 0) == 0 else "out"
            hot_cues.append({
                "label": getattr(cue, "Name", ""),
                "position_sec": round(getattr(cue, "Start", 0) / 1000, 3),
                "type": cue_type,
            })

        # Beatgrid
        beatgrid = None
        bpm_entries = getattr(rb_track, "tempo_entries", [])
        if bpm_entries:
            first = bpm_entries[0]
            beatgrid = {
                "bpm": round(float(getattr(first, "Bpm", cat_entry.get("bpm", 0))), 3),
                "first_beat_sec": round(getattr(first, "Inizio", 0) / 1000, 3),
            }

        if hot_cues:
            cat_entry["hot_cues"] = hot_cues
        if beatgrid:
            cat_entry["beatgrid"] = beatgrid
        updated += 1

    with open(_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2, ensure_ascii=False)

    summary = f"Rekordbox import: {updated} tracks updated."
    if unmatched:
        summary += f" {len(unmatched)} unmatched: {', '.join(unmatched[:5])}"
        if len(unmatched) > 5:
            summary += f" (+{len(unmatched) - 5} more)"
    return summary


# ---------------------------------------------------------------------------
# User context tools (v2.3.0) — surface per-user playlists/ratings to the
# planner. All four read context_variables["user_id"] and short-circuit with
# a stable "User context not available." string when missing (e.g. CLI
# `agent/run.py` runs that have no logged-in user).
#
# `web.backend.db` and `web.backend.pipeline` are imported lazily inside
# each function body to avoid a circular import: pipeline.py imports
# agent/tools.py at module load, so we cannot reference web.backend at
# the top of this file.
# ---------------------------------------------------------------------------

def get_user_playlists(context_variables: dict) -> str:
    """List the user's saved playlists as a Markdown table.

    Returns "User context not available." when no `user_id` is in context
    (e.g. when running via the CLI agent, not the web backend).
    """
    user_id = context_variables.get("user_id")
    if user_id is None:
        return "User context not available."
    from web.backend import db  # noqa: PLC0415 — lazy to avoid circular import
    rows = db.list_playlists_by_user(user_id)
    if not rows:
        return "User has no saved playlists."
    lines = ["| id | name | tracks |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['name']} | {r['track_count']} |")
    return "\n".join(lines)


def get_playlist_tracks(playlist_id: int, context_variables: dict) -> str:
    """Return the ordered tracks of one of the user's playlists.

    Verifies ownership: returns "Not authorized." if the playlist belongs
    to a different user. Returns "Playlist not found." if the id doesn't
    exist.
    """
    user_id = context_variables.get("user_id")
    if user_id is None:
        return "User context not available."
    from web.backend import db, pipeline  # noqa: PLC0415 — lazy
    p = db.get_playlist(playlist_id)
    if not p:
        return "Playlist not found."
    if p["user_id"] != user_id:
        return "Not authorized."
    try:
        catalog_tracks, _ = pipeline.load_catalog(None)
    except pipeline.CatalogUnavailable:
        catalog_tracks = []
    by_id = {t.get("id"): t for t in catalog_tracks if t.get("id")}
    lines = ["| pos | id | display_name | bpm | key |", "|---|---|---|---|---|"]
    for i, tid in enumerate(p["track_ids"]):
        t = by_id.get(tid, {"id": tid, "display_name": tid, "bpm": "—", "camelot_key": "—"})
        lines.append(
            f"| {i} | {t.get('id')} | {t.get('display_name')} | "
            f"{t.get('bpm')} | {t.get('camelot_key')} |"
        )
    return "\n".join(lines)


def get_user_ratings(context_variables: dict, min_rating: int = 1) -> str:
    """Return the user's ratings (filtered by `min_rating`) as JSON.

    Format: `{"<track_id>": <rating>, ...}`. `min_rating` defaults to 1
    (return everything). Returns "No ratings." when the filtered set is
    empty.
    """
    user_id = context_variables.get("user_id")
    if user_id is None:
        return "User context not available."
    from web.backend import db  # noqa: PLC0415 — lazy
    ratings = db.get_user_ratings(user_id)
    filtered = {tid: r for tid, r in ratings.items() if r >= min_rating}
    if not filtered:
        return "No ratings."
    return json.dumps(filtered, sort_keys=True)


def get_favorite_tracks(context_variables: dict, genre: str | None = None) -> str:
    """List tracks the user has rated >= 4 as a Markdown table.

    When `genre` is provided, intersect with the catalog of that genre so
    the planner only sees IDs it can actually pick. Falls back to a clear
    "No favorites." / "No favorites within genre 'X'." message when the
    intersection is empty.
    """
    user_id = context_variables.get("user_id")
    if user_id is None:
        return "User context not available."
    from web.backend import db, pipeline  # noqa: PLC0415 — lazy
    ratings = db.get_user_ratings(user_id)
    fav_ids = {tid for tid, r in ratings.items() if r >= 4}
    if not fav_ids:
        return "No favorites."
    try:
        catalog_tracks, _ = pipeline.load_catalog(genre)
    except pipeline.CatalogUnavailable:
        catalog_tracks = []
    fav_tracks = [t for t in catalog_tracks if t.get("id") in fav_ids]
    if not fav_tracks:
        return f"No favorites within genre '{genre}'." if genre else "No favorites in catalog."
    lines = ["| id | display_name | bpm | key | rating |", "|---|---|---|---|---|"]
    for t in fav_tracks:
        lines.append(
            f"| {t['id']} | {t['display_name']} | {t.get('bpm')} | "
            f"{t.get('camelot_key')} | {ratings.get(t['id'])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v2.5.2 — Live DJ improvisation tools
#
# These three tools turn the LiveDJ from a queue-executor into a real DJ:
#   - get_perception_window: read the last 30 s of mic perception samples
#     so the agent can correlate "the room got louder" with track choice.
#   - pick_next_track: search the FULL catalog (not just the initial
#     queue) by BPM range / key / mood. The agent uses this to swap in
#     tracks that the planner didn't pre-select.
#   - emit_chat: publish a `dj_chat` event over the WS so the LiveStage
#     chat panel surfaces a polite reply ("noted, but staying course").
# ---------------------------------------------------------------------------


def _mood_match(track: dict, mood: str) -> bool:
    """Return True when ``mood`` (case-insensitive) appears in any of the
    track's free-text fields. Used by ``pick_next_track`` for soft mood
    biasing — the planner already pre-filters by genre, so this is just
    a fuzzy contains-style search.
    """
    if not mood:
        return True
    needle = mood.strip().lower()
    if not needle:
        return True
    haystack_parts: list[str] = []
    for k in ("display_name", "genre", "genre_folder", "mood", "tags"):
        v = track.get(k)
        if isinstance(v, str):
            haystack_parts.append(v.lower())
        elif isinstance(v, list):
            haystack_parts.extend(str(x).lower() for x in v)
    suno = track.get("suno")
    if isinstance(suno, dict):
        for k in ("tags", "prompt"):
            v = suno.get(k)
            if isinstance(v, str):
                haystack_parts.append(v.lower())
            elif isinstance(v, list):
                haystack_parts.extend(str(x).lower() for x in v)
    return any(needle in part for part in haystack_parts)


def _format_duration(duration_sec: float | int | None) -> str:
    if duration_sec is None:
        return "?"
    try:
        d = float(duration_sec)
    except (TypeError, ValueError):
        return "?"
    if d <= 0:
        return "?"
    minutes = int(d // 60)
    seconds = int(d - minutes * 60)
    return f"{minutes}:{seconds:02d}"


def get_perception_window(context_variables: dict) -> str:
    """Summarise the last ~30 s of mic perception samples.

    Reads ``context_variables['perception_buffer']`` — populated by the
    web pipeline's ``phase_live`` from samples published by the browser
    (raw audio never leaves the browser; only RMS / onset density / VAD
    likelihood). Returns mean & max RMS for the last 15 samples (≈30 s
    at the default 2 s publish interval), the delta vs the session-wide
    average, and a voice likelihood mean when a VAD payload is present.

    Use to confirm an environment_changed event before reacting, or to
    poll the room when no synthetic event has fired in a while.
    """
    buf = context_variables.get("perception_buffer") or []
    if not buf:
        return "No perception data yet."
    recent = buf[-15:]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    rms_recent = [float(s.get("rms_db", 0.0)) for s in recent]
    rms_session = [float(s.get("rms_db", 0.0)) for s in buf]
    rms_mean = _mean(rms_recent)
    rms_max = max(rms_recent) if rms_recent else 0.0
    rms_session_mean = _mean(rms_session)

    voice_recent = [
        float(s["voice_likelihood"]) for s in recent
        if s.get("voice_likelihood") is not None
    ]
    voice_str = (
        f"{_mean(voice_recent):.2f}" if voice_recent else "N/A"
    )

    return (
        "Perception window (last 30s):\n"
        f"  rms_db: mean={rms_mean:.1f}, max={rms_max:.1f}, "
        f"delta vs session={rms_mean - rms_session_mean:+.1f}\n"
        f"  voice_likelihood: mean={voice_str}\n"
        f"  samples: {len(recent)}"
    )


def pick_next_track(
    bpm_min: float,
    bpm_max: float,
    context_variables: dict,
    key: str | None = None,
    mood: str | None = None,
    include_other_genres: bool = False,
) -> str:
    """Search the catalog for tracks matching BPM range / key / mood.

    Returns up to 5 candidates as a markdown table. The agent then picks
    one and feeds it to ``queue_swap`` or extends the queue with it.

    v3.9.2 — candidates are restricted to the SESSION'S GENRE. One
    out-of-genre pick flips the endless engine's genre permanently
    (it inherits genre from the current track): observed live
    2026-07-28 (aural→synthware) and 2026-08-04 (aural→lofi, the model
    picked a 75 BPM lofi track into a 52 BPM aural set "to get the
    energy up"). Energy changes happen WITHIN the session genre.

    Args:
        bpm_min: Minimum BPM (inclusive).
        bpm_max: Maximum BPM (inclusive).
        key: Optional Camelot key to prefer (e.g. "9A"). Out-of-key
            matches are still returned but ranked below.
        mood: Optional free-text mood — fuzzy matched against the
            track's display_name / genre / tags / suno prompt.
        include_other_genres: Leave False. Set True ONLY when the
            operator or the audience has EXPLICITLY asked, in their own
            words, for music from another genre. Never set it on your
            own initiative — not for energy, variety, or vibe reasons.
    """
    # Lazy import of the backend pipeline — agent/tools.py is imported by
    # the web backend itself, so eager import would cycle. Pattern set up
    # in v2.3.0 for tools that read backend state.
    try:
        from web.backend import pipeline  # noqa: PLC0415
        catalog, _ = pipeline.load_catalog(None)
    except Exception:  # noqa: BLE001 — fall back to direct catalog read
        try:
            with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
                catalog = json.load(fh).get("tracks", [])
        except Exception:  # noqa: BLE001
            return "Catalog unavailable."

    try:
        bpm_lo = float(bpm_min)
        bpm_hi = float(bpm_max)
    except (TypeError, ValueError):
        return "bpm_min and bpm_max must be numeric."
    if bpm_lo > bpm_hi:
        bpm_lo, bpm_hi = bpm_hi, bpm_lo

    # v3.9.1 — never surface sub-minimum-duration tracks as candidates:
    # whatever the LLM picks from this table can end up on stream.
    catalog = filter_session_eligible(catalog)

    # v3.9.2 — session-genre fence (see docstring). No ctx genre (CLI /
    # tests / non-session use) → unrestricted, as before.
    session_genre = (context_variables.get("genre") or "").strip().lower()
    genre_fenced = bool(session_genre) and not include_other_genres
    if genre_fenced:
        catalog = [
            t for t in catalog
            if (t.get("genre_folder") or t.get("genre") or "").strip().lower()
            == session_genre
        ]

    matches: list[dict] = []
    for t in catalog:
        bpm = t.get("bpm")
        if not isinstance(bpm, (int, float)):
            continue
        if not (bpm_lo <= float(bpm) <= bpm_hi):
            continue
        if key:
            # When a key is requested but the track has none, exclude — the
            # agent can re-call without `key` if it wants any match.
            tk = t.get("camelot_key")
            if not tk:
                continue
        if not _mood_match(t, mood or ""):
            continue
        matches.append(t)

    if not matches:
        crit = (
            f"bpm=[{bpm_lo:g}, {bpm_hi:g}]"
            + (f", key={key}" if key else "")
            + (f", mood={mood}" if mood else "")
        )
        if genre_fenced:
            return (
                f"No '{session_genre}' tracks in catalog matching {crit}. "
                "The search is restricted to the session's genre — widen "
                "the BPM range or drop the key/mood filter and try again."
            )
        return f"No tracks in catalog matching {crit}."

    mid_bpm = (bpm_lo + bpm_hi) / 2
    matches.sort(
        key=lambda t: (
            0 if (key and t.get("camelot_key") == key) else 1,
            abs(float(t.get("bpm") or 0) - mid_bpm),
        )
    )
    # v3.10 — one take per piece in the table: showing 'x' and 'x-v2'
    # as two candidates invites the LLM to queue the same music twice.
    # Ranked order is preserved, so the best-ranked take survives.
    matches = dedupe_takes(matches)
    top = matches[:5]

    lines = [
        "| id | display_name | bpm | key | duration |",
        "|---|---|---|---|---|",
    ]
    for t in top:
        lines.append(
            f"| {t.get('id','?')} | {t.get('display_name','?')} | "
            f"{t.get('bpm','?')} | {t.get('camelot_key','?')} | "
            f"{_format_duration(t.get('duration_sec'))} |"
        )
    return "\n".join(lines)


def extend_set(
    track_id: str,
    context_variables: dict,
    allow_other_genre: bool = False,
) -> str:
    """Append a catalog track to the live playlist (v2.6.0 endless mode).

    Use this AFTER ``pick_next_track`` has surfaced a candidate, when the
    `playlist_running_low` event fires and you want the set to keep
    going. The engine then plays the appended track as the new tail.
    If you don't act within ~5 s of `playlist_running_low`, the engine
    auto-picks an in-genre continuation deterministically.

    v3.9.2 — the track must belong to the session's genre, and must not
    already be playing or queued (the engine rejects duplicates).

    Args:
        track_id: The OPAQUE catalog id (a long dashed string ending in
            a UUID, e.g.
            ``lofi-ambient--lofi_2-soft_focus_at_76-38b20abc-...``).
            Must be copied VERBATIM from a ``pick_next_track`` result or
            from an engine event. Do NOT pass the song title /
            display_name / slugified guess — those are not ids and will
            fail. If you're unsure of the exact id, call
            ``pick_next_track`` again and read the ``id`` column.
        allow_other_genre: Leave False. Set True ONLY when the operator
            or the audience has EXPLICITLY asked, in their own words,
            for music from another genre. Never set it on your own
            initiative — not for energy, variety, or vibe reasons.

    Returns:
        Confirmation string with the track's new playlist position, or
        a guidance string if the id wasn't found (re-run pick_next_track,
        copy the id from its ``id`` column character for character).
    """
    engine = context_variables.get("_engine")
    if engine is None:
        return "Engine not running."
    # Lazy import mirrors pick_next_track to avoid the agent ↔ web cycle.
    try:
        from web.backend import pipeline  # noqa: PLC0415
        catalog, _ = pipeline.load_catalog(None)
    except Exception:  # noqa: BLE001 — fall back to direct catalog read
        try:
            with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
                catalog = json.load(fh).get("tracks", [])
        except Exception:  # noqa: BLE001
            return "Catalog unavailable."
    track = next((t for t in catalog if t.get("id") == track_id), None)
    if track is None:
        # v2.7.2 — instead of just saying "not in catalog", coach the LLM
        # back onto the correct path. The most common failure mode is
        # the model fabricating an id from a display_name; the retry
        # hint nudges it to re-fetch a real id rather than reshape its
        # guess into something even further from the truth.
        return (
            f"Track ID '{track_id}' is NOT in the catalog. "
            "This usually means the id was invented from a song title — "
            "track ids are opaque UUID-suffixed strings and must be "
            "copied verbatim from a pick_next_track result. "
            "Call pick_next_track again with your criteria and use the "
            "exact id from the 'id' column of its output."
        )
    # v3.9.1 — session-eligibility screen. pick_next_track no longer
    # surfaces these, but the id could come from an engine event or an
    # older turn's table; reject with the reason so the LLM re-picks.
    reason = ineligibility_reason(track)
    if reason:
        return (
            f"Track '{track.get('display_name', track_id)}' was NOT appended: "
            f"{reason}. Call pick_next_track for a longer candidate."
        )
    # v3.9.2 — session-genre fence, mirrors pick_next_track: one
    # out-of-genre append flips the endless engine's genre permanently
    # (observed live 2026-08-04, aural→lofi via 'Glockenspiel Dream').
    session_genre = (context_variables.get("genre") or "").strip().lower()
    if session_genre and not allow_other_genre:
        track_genre = (
            track.get("genre_folder") or track.get("genre") or ""
        ).strip().lower()
        if track_genre != session_genre:
            return (
                f"Track '{track.get('display_name', track_id)}' was NOT "
                f"appended: it is '{track_genre or 'unknown genre'}' but this "
                f"session is '{session_genre}'. Call pick_next_track for an "
                "in-genre candidate. (Only if the audience explicitly asked "
                "for another genre, retry with allow_other_genre=true.)"
            )
    return engine.append_track(track)


def emit_chat(text: str, context_variables: dict) -> str:
    """Publish a ``dj_chat`` event over the live WebSocket.

    The LiveStage UI surfaces this in its DJ chat panel — used to reply
    to audience requests without acting on the queue ("noted, but staying
    course"), or for ambient banter.

    Args:
        text: The reply to publish. Empty strings are dropped.
    """
    text = (text or "").strip()
    if not text:
        return "(emit_chat: empty text — nothing to publish)"
    emitter = context_variables.get("_event_emitter")
    if emitter is None:
        # Running outside the WS path (CLI mode, tests). Stash on context
        # for inspection rather than failing the tool call.
        context_variables.setdefault("_dj_chat_log", []).append(text)
        return f"(emit_chat: no event_emitter in context — running in non-WS mode) {text}"

    payload = {"type": "dj_chat", "text": text}

    # The emitter is async-callable. Try to schedule on the running loop
    # so the agent's worker thread can call us safely; fall back to a sync
    # invocation when ``emitter`` is a plain callable (CLI / test wrapper).
    try:
        import asyncio  # noqa: PLC0415

        result = emitter(payload)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(result)
            except RuntimeError:
                # No running loop — best effort; create the task on the
                # default policy's loop. In normal operation this branch
                # is unreachable because the WS handler always runs us
                # inside its own loop.
                asyncio.run(result)
    except Exception as exc:  # noqa: BLE001 — never crash the agent on a chat hiccup
        return f"(emit_chat: failed to publish — {type(exc).__name__}: {exc})"

    return f"emitted: {text}"


TOOLS = [
    list_genres,
    get_catalog,
    propose_playlist,
    show_playlist,
    analyze_transition,
    get_energy_arc,
    swap_track,
    move_track,
    suggest_bridge_track,
    insert_bridge_track,
    build_session,
    catalog_status,
    rebuild_catalog,
    fix_incomplete,
    redetect_bpm,
    validate_audio,
    read_memory,
    write_session_record,
    play_mix,
    preview_transition,
    play_track,
    start_live_session,
    import_rekordbox,
    generate_beatgrid,
]
