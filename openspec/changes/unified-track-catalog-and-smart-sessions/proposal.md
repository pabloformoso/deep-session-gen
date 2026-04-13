## Why

Session creation is a fully manual process: tracks are scattered across genre folders, each session requires hand-curating a `session.json`, and there's no shared catalog of what tracks exist with their BPM and harmonic data. This blocks rapid, on-demand session generation by genre and duration.

## What Changes

- **New `tracks/tracks.json`** — a unified catalog of every track across all genre folders (`cyberpunk/`, `deep house/`, `lofi - ambient/`, `techno/`), containing file path, genre folder, display name, Camelot key, and BPM (auto-detected via librosa). Suno `(1)` clip variants are cataloged as separate entries with a `variant_of` reference to the base track name.
- **New catalog builder** — a CLI command (`python main.py --build-catalog`) that scans all genre folders, detects BPM via librosa for each WAV, and writes/updates `tracks/tracks.json`. Existing entries are preserved to avoid re-scanning unchanged files.
- **New smart session generator** — **BREAKING** change to the main CLI: `python main.py --name <session-name> --genre <genre> --duration <minutes>`. Sessions are identified by name, not number. The generator selects tracks harmonically via the Camelot wheel and BPM proximity (±10 BPM window), fills the requested duration, generates a `session.json` in the output folder, generates DALL-E 3 artwork, and renders the full video.
- **BREAKING**: Session number-based invocation (`python main.py <N>`) is removed. All sessions — new and re-runs — use `--name`.

## Capabilities

### New Capabilities

- `track-catalog`: The unified `tracks/tracks.json` schema, catalog builder CLI, and rules for BPM auto-detection and Suno variant tracking.
- `smart-session-generator`: On-demand session generation by genre and duration using harmonic mixing (Camelot wheel) and BPM clustering. Includes track selection algorithm, session.json output, and integration with the existing artwork and render pipeline.

### Modified Capabilities

*(none)*

## Impact

- `main.py` — **BREAKING**: session number CLI removed; new flags `--name`, `--genre`, `--duration`, `--build-catalog`; new track selection and catalog scanning logic (~300–400 new lines estimated)
- `tracks/tracks.json` — new file, becomes the source of truth for all track metadata
- Output paths change from `output/session N/` to `output/<session-name>/` and `artwork/<session-name>/`
- Dependencies: `librosa` (already present) for BPM detection; no new dependencies required
