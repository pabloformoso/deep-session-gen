## ADDED Requirements

### Requirement: Name-based CLI invocation
The system SHALL accept `--name <session-name>`, `--genre <genre>`, and `--duration <minutes>` as CLI arguments to generate a session. Session number positional arguments SHALL NOT be supported. `--name` SHALL be used as the output folder name (spaces replaced with hyphens, lowercased).

#### Scenario: Valid invocation generates a session
- **WHEN** `python main.py --name "midnight-lofi" --genre "lofi - ambient" --duration 60` is run
- **THEN** the system generates a mix, renders a video, and writes output to `output/midnight-lofi/`

#### Scenario: Missing required argument fails with usage message
- **WHEN** any of `--name`, `--genre`, or `--duration` is omitted
- **THEN** the system prints a usage message and exits with a non-zero code

#### Scenario: Session number argument is rejected
- **WHEN** `python main.py 5` is run (positional integer)
- **THEN** the system prints an error message and exits with a non-zero code

### Requirement: BPM-coherent track selection
The system SHALL select tracks from the catalog filtered by `genre_folder` (case-insensitive), then group them into BPM clusters (±10 BPM). The largest cluster SHALL be used as the candidate pool.

#### Scenario: Tracks outside BPM window are excluded
- **WHEN** the candidate pool is assembled
- **THEN** no selected track has a BPM more than 10 BPM away from the cluster median

#### Scenario: Largest BPM cluster is preferred
- **WHEN** multiple BPM clusters exist for a genre
- **THEN** the cluster with the most tracks is used as the pool

#### Scenario: Genre filter is case-insensitive
- **WHEN** `--genre "Lofi - Ambient"` is passed
- **THEN** it matches the `lofi - ambient` folder and selects its tracks

### Requirement: Harmonic track ordering via Camelot wheel
Within the BPM cluster, the system SHALL order tracks using Camelot harmonic mixing rules: each successive track SHALL share the same key, be adjacent on the Camelot wheel (±1 number, same letter), or be a relative key change (same number, opposite letter). When no harmonic neighbor is available, the system SHALL fall back to any remaining track in the pool.

#### Scenario: Adjacent Camelot key is preferred
- **WHEN** the current track is in key 6A and candidates include keys 5A, 7A, and 6B
- **THEN** the next track is selected from 5A, 7A, or 6B before any other key

#### Scenario: Fallback when no harmonic neighbor exists
- **WHEN** no remaining track shares a Camelot-adjacent key with the current track
- **THEN** any remaining track from the pool is selected and a warning is printed to stdout

### Requirement: Soft duration target
The system SHALL accumulate tracks until the total playlist duration meets or exceeds `--duration` minutes. The last track SHALL NOT be cut; the session MAY run over the requested duration by up to one track length. Track durations SHALL be read from WAV file headers without full audio decoding.

#### Scenario: Session runs at least the requested duration
- **WHEN** `--duration 60` is specified
- **THEN** the total mix duration is ≥ 60 minutes

#### Scenario: Session does not cut mid-track
- **WHEN** adding the next track would exceed the duration target
- **THEN** the track is still included in full

#### Scenario: Pool exhaustion triggers cycling with a warning
- **WHEN** the genre pool is exhausted before the duration target is met
- **THEN** the system cycles through remaining tracks again and prints a warning listing the repeated tracks

### Requirement: display_name deduplication within a session
The system SHALL not include the same `display_name` more than once per session during initial pool selection. If cycling is required to meet duration, repeated display names are permitted.

#### Scenario: Same display_name not selected twice in first pass
- **WHEN** both a base track and its `(1)` variant share the same `display_name`
- **THEN** only one of them is included before the pool is considered exhausted

### Requirement: Session output structure
The system SHALL write all outputs for a named session to `output/<session-name>/` and artwork to `artwork/<session-name>/`. A `session.json` SHALL be saved inside the output folder recording the final playlist for reproducibility.

#### Scenario: Output directory is created if absent
- **WHEN** `output/<session-name>/` does not exist
- **THEN** the system creates it before writing any files

#### Scenario: session.json is written to output folder
- **WHEN** a session completes generation
- **THEN** `output/<session-name>/session.json` exists and contains the ordered playlist with file paths and metadata

#### Scenario: Re-running with the same name overwrites previous output
- **WHEN** `--name` matches an existing output folder
- **THEN** files in that folder are overwritten without prompting
