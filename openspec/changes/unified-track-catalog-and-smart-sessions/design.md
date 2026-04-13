## Context

Currently `main.py` accepts a session number as its sole argument and looks up `tracks/session N/session.json` to drive the mix. Tracks are spread across four genre folders (`cyberpunk/`, `deep house/`, `lofi - ambient/`, `techno/`) each holding WAV files and one or more legacy `session N.json` files. There is no shared index of what tracks exist, their BPM, or their Camelot key — that data lives only inside per-session JSONs and is duplicated across them.

The goal is to replace this with a name-based CLI and a single catalog that powers on-demand session generation.

## Goals / Non-Goals

**Goals:**
- One unified `tracks/tracks.json` as the source of truth for all track metadata
- CLI changes: `--build-catalog` to populate/refresh the catalog; `--name`, `--genre`, `--duration` to generate a session
- Smart track selection: Camelot-compatible transitions + BPM proximity (±10 BPM)
- Output paths keyed by session name (`output/<name>/`, `artwork/<name>/`)
- Suno `(1)` variants tracked as separate catalog entries with a `variant_of` field

**Non-Goals:**
- Keeping backward compatibility with session-number invocation
- Manual BPM override UI (BPM can be edited directly in `tracks.json`)
- Multi-genre sessions (single genre per session for now)
- Re-ranking or re-generating an existing named session (re-run with same name overwrites)

## Decisions

### 1. `tracks.json` schema

Each entry is a flat object:
```json
{
  "id": "lofi-ambient--warm-lamp-glow",
  "display_name": "Warm Lamp Glow",
  "file": "tracks/lofi - ambient/Warm Lamp Glow.wav",
  "genre_folder": "lofi - ambient",
  "genre": "Ambient Lo-Fi",
  "camelot_key": "2A",
  "bpm": 87.4,
  "variant_of": null
}
```
For `(1)` clips: `variant_of` is set to the base track's `display_name`. IDs are derived as `<genre-folder-slug>--<display-name-slug>[-v2]`.

**Why flat over nested-by-genre:** The selection algorithm needs to query across the full pool quickly; flat arrays are easiest to filter and sort without nested iteration.

### 2. Catalog builder strategy

`--build-catalog` scans each genre folder for WAV files, reads existing `session N.json` files to extract known `display_name`, `camelot_key`, and `genre` for matching filenames, then runs `librosa.beat.beat_track` for BPM on any entry not already present in `tracks.json`. Existing entries (matched by `file` path) are skipped to avoid re-scanning.

**Why seed from existing session JSONs:** They already have human-curated Camelot keys and genre labels — re-deriving these automatically would be lossy. BPM is the only field that needs auto-detection.

### 3. Track selection algorithm

```
1. Filter catalog by genre_folder matching --genre
2. Cluster by BPM: group tracks within ±10 BPM of each other; pick the largest cluster
3. Within the cluster, apply Camelot harmonic mixing:
   - Start from a random track
   - At each step, prefer tracks whose key is +1/-1 on the Camelot wheel (same letter) or same number ±1 (letter change)
   - Fall back to same key if no harmonic neighbor is available
4. Accumulate tracks until total duration >= --duration (durations read from WAV headers, no decoding needed)
5. If pool is exhausted before duration is reached, cycle through remaining tracks
```

**Why BPM clustering first, then Camelot:** BPM mismatch creates jarring energy jumps even with harmonic keys. Clustering first ensures the energy envelope is coherent; Camelot then handles the melodic/harmonic flow within that energy band.

**Why ±10 BPM:** Tight enough to keep the session feeling consistent; loose enough to give the selector sufficient track variety, especially in smaller genre pools (cyberpunk has ~12 tracks).

### 4. CLI interface

```
python main.py --build-catalog
python main.py --name "midnight-lofi" --genre "lofi - ambient" --duration 60
```

`--genre` accepts the folder name as-is (case-insensitive match). `--duration` is in minutes. `--name` is used directly as the output folder name (slugified if it contains spaces).

Session number positional argument is removed entirely.

### 5. Output structure

```
output/<session-name>/
  mix.wav
  session_video.mp4
  short.mp4
  session.json          ← saved for reproducibility

artwork/<session-name>/
  <track-display-name>.png
```

The saved `session.json` records the selected playlist in the existing format so the session can be re-rendered without re-running selection.

## Risks / Trade-offs

- **Small genre pools:** Cyberpunk has ~12 tracks; a 2-hour session will require looping. Looping means the same track appears multiple times — acceptable for background sessions but worth surfacing in the console output.
- **BPM detection accuracy:** librosa `beat_track` can be off by 2× on ambient tracks (detects half- or double-time). Mitigation: clamp detected BPM to a reasonable range per genre (lofi: 60–110, techno: 120–160, etc.) and flag outliers in the catalog so they can be corrected manually.
- **`(1)` variant naming:** Some variants are genuinely different tracks, others are near-identical. The catalog makes no quality distinction — both are eligible for selection. This could result in both the base and variant being picked in the same session. Mitigation: treat `display_name` as a deduplication key in session selection (same display name can only appear once per session).

## Migration Plan

1. Run `python main.py --build-catalog` once to generate `tracks/tracks.json`
2. Review and manually correct any BPM outliers in `tracks.json`
3. Existing legacy session JSONs in genre folders are no longer used by the runtime (they remain on disk as reference)
4. No rollback needed — old session numbers are simply unsupported after this change

## Open Questions

- Should `--duration` be a hard cap (stop just under) or a soft target (include the last track even if it goes over)? → Soft target is more natural for DJ sets; stop mid-track feels wrong.
- Should the generator print the proposed tracklist and ask for confirmation before rendering, or render immediately? → To be decided; a `--dry-run` flag could address this without blocking the default flow.
