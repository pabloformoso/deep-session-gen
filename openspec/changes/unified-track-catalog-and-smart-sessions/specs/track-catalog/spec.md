## ADDED Requirements

### Requirement: Unified track catalog file
The system SHALL maintain a single `tracks/tracks.json` file as the source of truth for all track metadata across all genre folders. Each entry SHALL include: `id`, `display_name`, `file` (relative path from project root), `genre_folder`, `genre`, `camelot_key`, `bpm`, and `variant_of` (null or base track display_name).

#### Scenario: Catalog contains all WAV files from all genre folders
- **WHEN** `--build-catalog` is run on a project with multiple genre folders under `tracks/`
- **THEN** `tracks/tracks.json` contains one entry per WAV file found across all genre folders

#### Scenario: Catalog entry has required fields
- **WHEN** a track entry is written to the catalog
- **THEN** it contains non-null values for `id`, `display_name`, `file`, `genre_folder`, `camelot_key`, and `bpm`

#### Scenario: Suno variant is linked to base track
- **WHEN** a WAV file ends with ` (1)` and a base file with the same display name exists in the same folder
- **THEN** the variant entry has `variant_of` set to the base track's `display_name`

#### Scenario: Base track has null variant_of
- **WHEN** a WAV file has no `(1)` suffix
- **THEN** its catalog entry has `variant_of` set to null

### Requirement: Catalog builder CLI command
The system SHALL provide a `--build-catalog` flag that scans all genre subfolders of `tracks/`, seeds `camelot_key` and `genre` from any existing `session N.json` files in each folder, auto-detects BPM and Camelot key via librosa for each WAV, and writes the result to `tracks/tracks.json`. Adding a new track is done by dropping its WAV file into the appropriate genre folder and re-running `--build-catalog`.

#### Scenario: Existing entries are not re-scanned
- **WHEN** `--build-catalog` is run and `tracks/tracks.json` already contains an entry for a given file path
- **THEN** that entry's BPM and camelot_key are not re-detected and its existing values are preserved

#### Scenario: New tracks are appended
- **WHEN** a WAV file is present in a genre folder but has no entry in `tracks/tracks.json`
- **THEN** a new entry is created with BPM and Camelot key auto-detected via librosa

#### Scenario: Camelot key seeded from session JSON takes priority
- **WHEN** a genre folder contains a `session N.json` that references a file by the same display name
- **THEN** the catalog entry uses the `camelot_key` from the session JSON rather than the auto-detected value

#### Scenario: BPM outlier clamping
- **WHEN** librosa detects a BPM outside the expected range for the genre folder (lofi: 60–110, techno/cyberpunk: 120–160, deep house: 115–135)
- **THEN** the detected value is clamped to the nearest boundary and a warning is printed to stdout

### Requirement: Camelot key auto-detection
For tracks without a `camelot_key` in any session JSON, the system SHALL auto-detect the musical key using librosa chromagram analysis and map it to the corresponding Camelot notation (e.g., A minor → 8A, C major → 8B). The detected value SHALL be written to `tracks/tracks.json` and MAY be manually corrected by editing the file directly.

#### Scenario: Key detected for new track with no session JSON entry
- **WHEN** a new WAV is added to a genre folder that has no matching entry in any session JSON
- **THEN** `--build-catalog` detects the musical key via chromagram and writes the Camelot notation to `camelot_key`

#### Scenario: Manual correction is preserved on re-scan
- **WHEN** a user edits `camelot_key` directly in `tracks/tracks.json` and then re-runs `--build-catalog`
- **THEN** the manually set value is preserved (existing entries are not re-scanned)

### Requirement: Catalog ID uniqueness
Each catalog entry SHALL have a unique `id` derived from the genre folder slug and display name slug. Suno `(1)` variants SHALL receive a `-v2` suffix on their ID.

#### Scenario: IDs are unique across genres
- **WHEN** two tracks in different genre folders share the same display name
- **THEN** their catalog IDs differ due to the genre folder slug prefix

#### Scenario: Variant ID has -v2 suffix
- **WHEN** a track is a Suno `(1)` variant
- **THEN** its `id` ends with `-v2`
