# ApolloAgents — Roadmap

This is the living roadmap for ApolloAgents — from quick polish to long-term vision. Items are grouped by milestone, not by date. v1.0 shipped April 2026; v1.0.1 shipped April 2026. Everything below is what comes next.

Contributions welcome — new tools follow the `fn(params, context_variables: dict) -> str` convention and new agents follow the bounded-role pattern (system prompt + curated tool list + structured output). See [CLAUDE.md](CLAUDE.md) for developer setup.

---

## v1.0 — Released ✓

The baseline — everything that ships today.

- **8-phase agent pipeline**: Janus (genre guard) → Hermes (catalog) → Muse (planner) → Checkpoint → Momus (critic) → Checkpoint → Editor REPL → Themis (validator)
- **Harmonic mixing**: Camelot wheel neighbour graph, BPM clustering, harmonic random-walk sort
- **BPM matching**: pyrubberband phase-vocoder with 16s tempo ramp; −3 dB pre-mix gain + post-mix normalisation to fix crossfade clipping
- **Audio validation**: peak clipping, spectral flatness (bleach detection), silence gaps, RMS anomaly — all via librosa
- **Session memory**: `agent/memory.json` — avoid list, high-rated patterns, recurring critic problems
- **Video output**: 1080p spectral waveform, beat-reactive particles, DALL-E 3 artwork, Press Start 2P retro titles; 20s YouTube Short
- **Catalog manager**: `catalog_status`, `rebuild_catalog`, `fix_incomplete` — keeps `tracks.json` in sync
- **CI**: 45 unit tests (Camelot, parsers, memory), GitHub Actions on Python 3.12 & 3.13

---

## v1.0.1 — Released ✓

Small changes, meaningful impact.

- **Accurate track durations** — `duration_sec` stored in `tracks.json` at catalog build time (WAV header read, no decode). `propose_playlist()` uses real durations instead of a hardcoded 5 min/track estimate; `build_catalog()` backfills existing entries on next run.
- **Pre-flight transition warnings** — `swap_track()` and `move_track()` emit `⚠` warnings inline when a change creates a harmonic clash (>2 Camelot steps) or extreme BPM stretch (>1.5× ratio), before confirming.
- **Catalog onboarding UX** — On startup, if catalog is empty or has unsynced WAV files, the agent proactively asks "Sync catalog before building a set?". After catalog sync, hands off automatically to the DJ set builder.
- **Full theme passthrough** — `_get_session_theme()` merges DEFAULT_THEME → GENRE_THEMES[genre] → session overrides, so `session.json` partial overrides work without losing genre colors. `genre` stored in `session_config` to enable this. Removed dead-code constant fallbacks in video gen functions.
- **Genre Guard UX** — Extracts genre from the user's initial message (e.g. "2h lofi set" → `lofi - ambient`) before falling back to `list_genres`.

---

## v1.2 — Smarter Learning

Making the memory system earn its keep.

- **Per-transition ratings** — After the mix plays, prompt for a rating on each transition (1–5). Index `memory.json` by `(camelot_key_pair, bpm_diff_bucket)` so Muse and Momus learn which *specific* transitions you tolerate, not just which tracks you swapped.
- **Energy arc quantization** — Represent the playlist internally as `[(energy_level 0–10, camelot_key, bpm), ...]`. Muse reasons over this structure explicitly to detect plateaus and missing release sections, rather than describing them in prose.
- **Richer critic memory** — Store which positions and key pairs caused recurring problems (not just raw problem text) so Momus can flag patterns like "your 8A→3B transitions are always a problem".

---

## v1.3 — Audio Safety & Depth

Catching problems before the render, not after.

- **BPM stretch safety bounds** — Before rendering, flag track pairs requiring >1.5× pyrubberband stretch ratio. Momus warns during critique; Editor can suggest a bridge track from the catalog.
- **Bridge track insertion** — New `suggest_bridge_track(from_bpm, to_bpm, genre)` and `insert_bridge_track(position, genre)` tools search the catalog for a track with intermediate BPM to smooth large jumps.
- **EQ matching at crossfade** — Apply gentle high-pass / low-pass shelving at crossfade points to reduce frequency masking between adjacent tracks in different key ranges.

---

## v1.4 — Live Local Playback ⭐

> *"Ask the agent to play music"*

The agent can already build a mix — now it can play it back, and let you hear transitions before committing to a build.

- **`play_mix(session_name)`** — After build, streams `mix_output.wav` locally via `afplay` (macOS), `aplay` (Linux), or `sounddevice` (cross-platform). No video render required.
- **`preview_transition(pos_a, pos_b)`** — During the Editor REPL, extracts and plays the crossfade zone (±15s around the transition point) so you can hear it before building the full mix.
- **`play_track(track_id, start_sec, duration_sec)`** — Audition individual tracks from the catalog during planning. Useful for tracks you haven't heard in a while.

All three tools added to `agent/tools.py` and exposed to the Editor agent.

---

## v2.0 — Vision

Bigger swings. No timeline.

| Feature | Description |
|---|---|
| **Multi-genre sessions** | Janus accepts a genre sequence (e.g., deep house → techno). Validates harmonic/BPM bridges at genre boundaries. |
| **Web UI** | Browser-based frontend replacing the terminal REPL. Drag-and-drop ordering, waveform preview, inline Critic feedback. |
| **Spotify / Rekordbox import** | Map external library exports to `tracks.json` format — BPM and key already exist in both. |
| **Cloud rendering** | Offload the video render to a worker (Modal, RunPod) for faster turnaround on long sessions. |
| **Plugin architecture** | Define a `BaseAgent` protocol so community agents (e.g., a "Crowd Energy" agent, a "Stem Separator" agent) can plug into the pipeline. |
| **Stem separation** | Use Demucs to separate vocals/drums/bass before mixing; enable per-stem crossfade for cleaner transitions. |

---

## Contributing

New tools:
```python
def my_tool(param: str, context_variables: dict) -> str:
    """One-line description used as the tool schema description.

    Args:
        param: What this parameter does
    """
    ...
```

New agents: add a `_SYSTEM` prompt constant, a tool subset list, and a phase block in `_orchestrate()` in `agent/run.py`. Follow the structured output protocol (CONFIRMED blocks, PROBLEMS/VERDICT, Status: fields) for anything that needs to be parsed downstream.

See [CLAUDE.md](CLAUDE.md) for full developer reference.
