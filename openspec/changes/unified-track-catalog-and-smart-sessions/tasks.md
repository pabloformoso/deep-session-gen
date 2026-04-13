## 1. CLI Refactor

- [x] 1.1 Remove the positional session-number argument from `main.py`'s argument parser
- [x] 1.2 Add `--name`, `--genre`, `--duration` arguments to the argument parser
- [x] 1.3 Add `--build-catalog` flag to the argument parser
- [x] 1.4 Update output path helpers to use `output/<session-name>/` and `artwork/<session-name>/` instead of `output/session N/`
- [x] 1.5 Add validation: exit with usage message if `--name`, `--genre`, or `--duration` are missing when not running `--build-catalog`

## 2. Catalog Builder

- [x] 2.1 Write a `scan_genre_folders()` function that globs `tracks/*/` and returns a list of WAV files with their parent folder name
- [x] 2.2 Write a `load_existing_session_jsons()` function that reads all `session N.json` files in a genre folder and builds a `{display_name: {camelot_key, genre}}` lookup
- [x] 2.3 Write a `detect_bpm(filepath)` function using `librosa.beat.beat_track` that clamps the result to genre-appropriate ranges (lofi: 60–110, techno/cyberpunk: 120–160, deep house: 115–135) and prints a warning if clamping occurs
- [x] 2.3b Write a `detect_camelot_key(filepath)` function using `librosa.feature.chroma_cqt` to estimate the dominant pitch class, map it to a musical key (major/minor via Krumhansl–Schmuckler profiles), and convert to Camelot notation
- [x] 2.4 Write a `build_catalog()` function that: loads existing `tracks/tracks.json` (if present), identifies new WAV files not yet cataloged, seeds `camelot_key`/`genre` from session JSONs (takes priority over auto-detect), calls `detect_bpm()` and `detect_camelot_key()` for new entries only, assigns `id` and `variant_of`, and writes the updated catalog
- [x] 2.5 Implement ID generation: `<genre-folder-slug>--<display-name-slug>` with `-v2` suffix for `(1)` variants
- [x] 2.6 Implement `variant_of` detection: if filename ends with ` (1)` and a base file with the same display name exists in the folder, set `variant_of` to the base display name
- [x] 2.7 Wire `--build-catalog` in the main entrypoint to call `build_catalog()` and exit

## 3. Track Selection Algorithm

- [x] 3.1 Write a `load_catalog(genre)` function that reads `tracks/tracks.json` and returns entries filtered by `genre_folder` (case-insensitive)
- [x] 3.2 Write a `bpm_cluster(tracks)` function that groups tracks into ±10 BPM clusters and returns the largest cluster
- [x] 3.3 Write a `camelot_neighbors(key)` function that returns the set of Camelot-adjacent keys (same key, ±1 number same letter, same number opposite letter)
- [x] 3.4 Write a `harmonic_sort(tracks, start=None)` function that builds a playlist using the Camelot walk, with fallback to any remaining track and a stdout warning when falling back
- [x] 3.5 Write a `fill_duration(ordered_tracks, duration_minutes)` function that reads WAV durations from file headers (no decoding), accumulates tracks to meet or exceed the target, cycles with a warning if the pool is exhausted, and enforces `display_name` deduplication on the first pass

## 4. Session Generation & Output

- [x] 4.1 Write a `generate_session(name, genre, duration)` function that orchestrates: `load_catalog` → `bpm_cluster` → `harmonic_sort` → `fill_duration` → returns an ordered track list
- [x] 4.2 Create `output/<session-name>/` and `artwork/<session-name>/` directories if they don't exist
- [x] 4.3 Write `output/<session-name>/session.json` with the final ordered playlist (matching existing session.json format: `name`, `theme`, `playlist` with `display_name`, `file`, `camelot_key`, `genre`)
- [x] 4.4 Wire the generated track list into the existing artwork generation pipeline (DALL-E 3, deduplication by display_name)
- [x] 4.5 Wire the generated track list into the existing audio mix and video render pipeline

## 5. Verification

- [x] 5.1 Run `--build-catalog` on the full `tracks/` folder and verify all 4 genre folders are scanned and `tracks/tracks.json` is written with correct fields
- [x] 5.2 Re-run `--build-catalog` and confirm no BPM re-detection occurs for existing entries
- [x] 5.3 Generate a 30-minute lofi session and verify: output folder created, `session.json` written, all tracks are within ±10 BPM of each other
- [x] 5.4 Generate a 30-minute techno session and verify Camelot keys follow harmonic adjacency rules
- [x] 5.5 Confirm running with a session number argument prints an error and exits non-zero
- [x] 5.6 Confirm re-running with the same `--name` overwrites previous output without prompting
- [x] 5.7 Drop a new WAV into a genre folder, re-run `--build-catalog`, and verify the entry appears in `tracks/tracks.json` with non-null `bpm` and `camelot_key`
- [x] 5.8 Manually edit a `camelot_key` in `tracks/tracks.json`, re-run `--build-catalog`, and confirm the manual value is preserved
