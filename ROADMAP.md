# ApolloAgents — Roadmap

This is the living roadmap for ApolloAgents — from quick polish to long-term vision. Items are grouped by milestone, not by date. v1.0 shipped April 2026; v1.1, v1.1.1, and v1.1.2 shipped April 2026. Everything below is what comes next.

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

## v1.1.2 — Released ✓

- **Checkpoint catalog access** — `get_catalog` now accepts `"current"` to resolve the session genre from context; Checkpoint intro passes the genre explicitly; system prompt enforces the call-get_catalog-first workflow to prevent hallucinated track IDs.

---

## v1.1.1 — Released ✓

Bug fixes found during v1.1 testing.

- **BPM detection fixed** — `detect_bpm()` now passes `start_bpm` biased to the genre midpoint and tries halving/doubling before clamping; fixes all lofi tracks being detected at 110 BPM. New `--redetect-bpm` CLI flag (and `redetect_bpm` agent tool) to re-process existing catalog entries.
- **Duration backfill** — `build_catalog()` now patches `duration_sec` for existing entries missing it, so upgrading from v1.1 doesn't require a manual `--fix-incomplete` run.
- **Catalog → session handoff** — After catalog sync, the agent now routes back to the DJ set builder instead of exiting. The user's original session request is preserved through the freshness-check path.
- **Genre Guard UX** — Extracts genre from the user's initial message before calling `list_genres` (e.g. "2h lofi set" → `lofi - ambient` without an extra round-trip).
- **Checkpoint catalog access** — Checkpoint agent now has `get_catalog` and `analyze_transition` tools, so "find a replacement" requests work without asking the user for a track ID.
- **Playlist display** — Removed 300-char truncation on tool result output; full playlist now prints to terminal.

---

## v1.1 — Released ✓

Small changes, meaningful impact.

- **Accurate track durations** — `duration_sec` stored in `tracks.json` at catalog build time (WAV header read, no decode). `propose_playlist()` uses real durations instead of a hardcoded 5 min/track estimate; `build_catalog()` backfills existing entries on next run.
- **Pre-flight transition warnings** — `swap_track()` and `move_track()` emit `⚠` warnings inline when a change creates a harmonic clash (>2 Camelot steps) or extreme BPM stretch (>1.5× ratio), before confirming.
- **Catalog onboarding UX** — On startup, if catalog is empty or has unsynced WAV files, the agent proactively asks "Sync catalog before building a set?". After catalog sync, hands off automatically to the DJ set builder.
- **Full theme passthrough** — `_get_session_theme()` merges DEFAULT_THEME → GENRE_THEMES[genre] → session overrides, so `session.json` partial overrides work without losing genre colors. `genre` stored in `session_config` to enable this. Removed dead-code constant fallbacks in video gen functions.
- **Genre Guard UX** — Extracts genre from the user's initial message (e.g. "2h lofi set" → `lofi - ambient`) before falling back to `list_genres`.

---

## v1.2 — Smarter Learning ✓

Making the memory system earn its keep.

- **Per-transition ratings** — After the mix plays, prompt for a rating on each transition (1–5). Index `memory.json` by `(camelot_key_pair, bpm_diff_bucket)` so Muse and Momus learn which *specific* transitions you tolerate, not just which tracks you swapped.
- **Energy arc quantization** — New `get_energy_arc` tool represents the playlist as `[(energy 0–10, key, bpm), ...]`. Muse calls it after planning to detect plateaus and missing peak/release; Momus calls it during critique. Energy derived from BPM normalization + Camelot key range.
- **Richer critic memory** — `_parse_critic_response` extracts position, key pair, and BPM diff from each problem line into `structured_problems`. `read_memory` aggregates by key pair across sessions and flags recurring clashes.
- **Gemma 4 (4B) local provider** — Full offline operation via Ollama. Auto-detected from environment alongside Anthropic and OpenAI; `AGENT_PROVIDER=ollama` + `AGENT_MODEL=gemma4:4b` to activate. Uses OpenAI-compatible `/v1/chat/completions` endpoint.

---

## v1.3 — Audio Safety & Depth

Catching problems before the render, not after.

- **BPM stretch safety bounds** ✓ — Before rendering, flag track pairs requiring >1.5× pyrubberband stretch ratio. Momus warns during critique; Editor can suggest a bridge track from the catalog.
- **Bridge track insertion** ✓ — New `suggest_bridge_track(from_pos, to_pos)` and `insert_bridge_track(after_position, track_id)` tools let the Editor find and insert a bridge track for BPM gaps.
- **EQ matching at crossfade** ✓ — Apply gentle high-pass / low-pass shelving at crossfade points to reduce frequency masking between adjacent tracks in different key ranges.

---

## v1.4 — Live Local Playback ✓

> *"Ask the agent to play music"*

The agent can already build a mix — now it can play it back, and let you hear transitions before committing to a build.

- **`play_mix(session_name)`** ✓ — After build, streams `mix_output.wav` locally via `afplay` (macOS), `ffplay` (ffmpeg), or `aplay` (Linux ALSA). Non-blocking background playback.
- **`preview_transition(pos_a, pos_b, session_name)`** ✓ — During the Editor REPL, extracts and plays the ±15s crossfade zone between two adjacent tracks (requires a rendered mix).
- **`play_track(track_id, start_sec, duration_sec)`** ✓ — Audition individual tracks from the catalog during planning. Supports time-sliced playback.

All three tools added to `agent/tools.py` and exposed to the Editor agent.

---

## v1.5 — LiveDJ Agent

> *"Spin it live"*

A new proactive event-driven agent that performs a live set in real time,
without rendering a video or a full mix file first.

- **`LiveEngine`** (`agent/live_engine.py`) — two-deck audio engine: loads tracks
  as numpy arrays, pre-stretches the next track in a background thread, blends
  decks with a linear fade over `CROSSFADE_SEC`. Fires events to a
  `threading.Queue` shared with the agent.
- **Event system** — six event types: `track_started`, `approaching_crossfade`,
  `crossfade_triggered`, `crossfade_finished`, `track_ended`, `session_ended`.
- **`LiveDJ` agent** (`agent/live_dj.py`) — proactive LLM agent that reacts to
  engine events. On `approaching_crossfade`: evaluates transition quality and
  decides to confirm, extend, or crossfade early. On user commands: translates
  natural language to engine actions.
- **Hot cues + beatgrid** — optional `hot_cues` and `beatgrid` fields in
  `tracks.json`. Engine uses OUT hot cue as crossfade point and IN hot cue as
  incoming track start. Falls back gracefully when absent.
- **`import_rekordbox(xml_path)`** — catalog tool to import hot cues and beatgrid
  from a Rekordbox XML export via `pyrekordbox`. Enriches `tracks.json` in place.
- **`start_live_session(session_name)`** — Editor tool that launches the LiveDJ
  loop. Coexists with `build_session`; either can be used after set approval.
- **New deps**: `sounddevice`, `pyrekordbox`.
- **Tests**: `tests/test_live_engine.py` (state machine, hot cues, events),
  `tests/test_live_tools.py` (all six live tools with mocked engine).

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
