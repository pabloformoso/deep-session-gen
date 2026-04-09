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
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_AGENT_DIR = Path(__file__).parent
_PROJECT_DIR = _AGENT_DIR.parent
_CATALOG_PATH = _PROJECT_DIR / "tracks" / "tracks.json"
_MAIN_PY = _PROJECT_DIR / "main.py"
_MEMORY_PATH = _AGENT_DIR / "memory.json"

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
    "techno": (120, 160),
    "cyberpunk": (120, 160),
    "deep house": (115, 135),
}


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
# Tool functions
# ---------------------------------------------------------------------------

def list_genres(context_variables: dict) -> str:
    """List all available genre folders from the track catalog."""
    if not _CATALOG_PATH.exists():
        return "Error: tracks.json not found. Run 'python main.py --build-catalog' first."
    with open(_CATALOG_PATH) as f:
        data = json.load(f)
    genres = sorted({t["genre_folder"] for t in data["tracks"]})
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

    with open(_CATALOG_PATH) as f:
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

    with open(_CATALOG_PATH) as f:
        data = json.load(f)

    genre_lower = genre.lower()
    all_tracks = [t for t in data["tracks"] if t["genre_folder"].lower() == genre_lower]
    if not all_tracks:
        available = sorted({t["genre_folder"] for t in data["tracks"]})
        return f"No tracks for '{genre}'. Available: {', '.join(available)}"

    cluster = _bpm_cluster(all_tracks)
    ordered = _harmonic_sort(cluster)

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

    with open(_CATALOG_PATH) as f:
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
    key_compat = _camelot_compat(a.get("camelot_key"), b.get("camelot_key"))

    if bpm_diff == 0:
        bpm_status = "identical BPM — no pitch shift needed"
    elif bpm_diff <= 5:
        bpm_status = f"{bpm_diff:.1f} BPM diff — within threshold, keep outgoing tempo"
    elif bpm_diff <= 15:
        bpm_status = f"{bpm_diff:.1f} BPM diff — meet in the middle (16s ramp)"
    else:
        bpm_status = f"{bpm_diff:.1f} BPM diff — large jump, consider a bridge track"

    return (
        f"Transition: {a['display_name']} → {b['display_name']}\n"
        f"  BPM:      {bpm_a:.0f} → {bpm_b:.0f}  ({bpm_status})\n"
        f"  Key:      {a.get('camelot_key','?')} → {b.get('camelot_key','?')}  ({key_compat})\n"
    )


def swap_track(position: int, track_id: str, context_variables: dict) -> str:
    """Replace the track at a given position (1-indexed) with another track from the catalog.

    Args:
        position: 1-indexed position in the current playlist
        track_id: ID of the replacement track (from get_catalog)
    """
    playlist = context_variables.get("playlist")
    if not playlist:
        return "No playlist in memory. Use propose_playlist first."

    if position < 1 or position > len(playlist):
        return f"Position {position} is out of range (playlist has {len(playlist)} tracks)."

    if not _CATALOG_PATH.exists():
        return "Error: catalog not found."

    with open(_CATALOG_PATH) as f:
        data = json.load(f)

    index = {t["id"]: t for t in data["tracks"]}
    new_track = index.get(track_id)
    if not new_track:
        return f"Track '{track_id}' not found. Use get_catalog to see valid IDs."

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

    warning_block = ("\n".join(pre_warnings) + "\n\n") if pre_warnings else ""
    return (
        warning_block
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


def suggest_bridge_track(from_pos: int, to_pos: int, context_variables: dict) -> str:
    """Find candidate bridge tracks between two BPM-mismatched playlist positions.

    Args:
        from_pos: 1-indexed position of the outgoing track
        to_pos: 1-indexed position of the incoming track
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
        with open(_CATALOG_PATH) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Could not load catalog: {e}"

    genre = context_variables.get("genre", "").lower()
    playlist_ids = {t.get("id") for t in playlist}

    candidates = []
    for c in catalog:
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
        lines.append(
            f"  {c['id']} | {c['display_name']} | {c_bpm:.1f} BPM | "
            f"{c.get('camelot_key', '?')} | "
            f"ratio_a: {ratio_a:.2f}× | ratio_b: {ratio_b:.2f}× | score: {score:.3f}"
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
        with open(_CATALOG_PATH) as f:
            catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Could not load catalog: {e}"

    new_track = next((c for c in catalog if c.get("id") == track_id), None)
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


def build_session(session_name: str, context_variables: dict) -> str:
    """Save the current playlist as a draft and trigger the full mix + video pipeline.

    Args:
        session_name: Name for the output folder (e.g. 'midnight-techno')
    """
    playlist = context_variables.get("playlist")
    genre = context_variables.get("genre")

    if not playlist:
        return "No playlist in memory. Use propose_playlist first."
    if not genre:
        return "Genre not set in context. Run propose_playlist first."

    # Save draft session.json that main.py --from-session will pick up
    draft_path = _PROJECT_DIR / "output" / f"_draft_{session_name}" / "session.json"
    draft_path.parent.mkdir(parents=True, exist_ok=True)

    GENRE_THEMES = {
        "lofi - ambient": {"artwork_style": "anime", "title_color": "#E8D5B7"},
        "deep house": {"artwork_style": "deep-house-neon", "title_color": "#6A5AFF"},
        "techno": {"artwork_style": "dark-techno", "title_color": "#FF1744"},
        "cyberpunk": {"artwork_style": "dark-techno", "title_color": "#00FF88"},
    }

    session_config = {
        "name": session_name,
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

    with open(draft_path, "w") as f:
        json.dump(session_config, f, indent=2)

    # Kick off main.py --from-session
    cmd = [
        sys.executable,
        str(_MAIN_PY),
        "--from-session", str(draft_path),
        "--name", session_name,
        "--genre", genre,
    ]

    print(f"\n[Agent] Launching pipeline: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(_PROJECT_DIR))

    if result.returncode == 0:
        context_variables["last_build"] = session_name
        return f"Build complete! Output → output/{session_name}/"
    else:
        return f"Pipeline exited with code {result.returncode}. Check the output above for errors."


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
        lines.append(f"  {i:2d}. {t['display_name']:<30}  {bpm} BPM  [{key}]")

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
    """Compare WAV files on disk vs tracks.json and report what is missing or orphaned.

    Shows per-genre: how many files on disk, how many in catalog, which are new (not yet cataloged),
    and which catalog entries have missing files (orphaned).
    """
    import os as _os

    tracks_dir = _PROJECT_DIR / "tracks"
    if not tracks_dir.exists():
        return "Error: tracks/ folder not found."

    # Load catalog
    cataloged: dict[str, dict] = {}
    if _CATALOG_PATH.exists():
        with open(_CATALOG_PATH) as f:
            data = json.load(f)
        for entry in data.get("tracks", []):
            cataloged[entry["file"]] = entry

    # Scan folders
    disk_files: dict[str, list[str]] = {}  # genre_folder → [rel_path, ...]
    for folder in sorted(_os.listdir(tracks_dir)):
        folder_path = tracks_dir / folder
        if not folder_path.is_dir():
            continue
        wavs = sorted(
            str((folder_path / f).relative_to(_PROJECT_DIR)).replace("\\", "/")
            for f in _os.listdir(folder_path)
            if f.lower().endswith(".wav")
        )
        if wavs:
            disk_files[folder] = wavs

    if not disk_files:
        return "No WAV files found in any tracks/ subfolder."

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
            with open(_CATALOG_PATH) as f:
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

    with open(_MEMORY_PATH) as f:
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
        with open(_MEMORY_PATH) as f:
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
    with open(tmp, "w") as f:
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
]
