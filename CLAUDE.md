# ApolloAgents — Developer Notes

Automated DJ mix generator + multi-agent AI pipeline. Takes WAV audio files,
BPM-matches them, applies crossfades, and renders 1080p YouTube videos with
waveform visualizations, AI-generated artwork, and retro animated titles.

## Running

```bash
# Build/refresh the track catalog (run once after adding new WAV files)
python main.py --build-catalog

# Re-analyse catalog entries with missing BPM or Camelot key
python main.py --fix-incomplete

# Generate a session directly (no agent)
python main.py --name "midnight-lofi" --genre "lofi - ambient" --duration 60

# Run the conversational agent
uv run python agent/run.py
```

`--genre` must match a subfolder name under `tracks/` (case-insensitive).
`--duration` is in minutes (soft target — last track is never cut).

Requires an `.env` file — see `.env.example`.

## Project structure

```
main.py                        # Single-file pipeline (~2600 lines)
agent/
  run.py                       # Apollo orchestrator + all agent loops
  tools.py                     # Tool functions (catalog, playlist, validator, memory)
  memory.json                  # Session history — gitignored, auto-created
tracks/
  tracks.json                  # Unified catalog: id, display_name, file,
                               #   genre_folder, genre, camelot_key, bpm, variant_of
  lofi - ambient/              # WAV files per genre
  deep house/
  techno/
  cyberpunk/
output/
  <session-name>/              # Final video and audio outputs (gitignored)
    mix_output.wav
    mix_video.mp4
    short.mp4
    session.json
    transitions.json
    youtube.md
artwork/
  <session-name>/              # DALL-E 3 generated backgrounds (gitignored)
fonts/
  PressStart2P-Regular.ttf
```

## Architecture decisions

- **Single file (`main.py`)** — intentional, ~2600 lines is manageable for this scope
- **Lossless pipeline** — WAV throughout, only AAC compression at final video encode
- **Per-session output** — `output/<session-name>/`, `artwork/<session-name>/`
- **Artwork deduplication** — tracks with the same `display_name` share one image
- **Agent memory** — `agent/memory.json` is gitignored; each user builds their own

## Key constants (top of `main.py`)

| Constant | Purpose |
|---|---|
| `CROSSFADE_SEC` | Crossfade overlap length (default 12s) |
| `TEMPO_RAMP_SEC` | Gradual BPM ramp after crossfade (default 16s) |
| `BPM_MATCH_THRESHOLD` | Min BPM diff to trigger tempo matching (default 5) |
| `VIDEO_SIZE` | Output resolution (default 1920×1080) |
| `FONT_PATH` | Press Start 2P pixel font |

## Genre themes

Defined in `GENRE_THEMES` dict in `main.py`. Each genre has: `artwork_style`,
`title_color`, `title_stroke_color`, `bg_color`, `waveform_color`, `particle_color`.

Available `artwork_style` values: `abstract`, `realistic`, `anime`,
`dystopic-calm`, `dark-techno`, `organic-zen`, `deep-house-neon`.

## Adding a new genre

1. Create `tracks/<genre-name>/` and add WAV files
2. Run `python main.py --build-catalog`
3. Optionally add a theme entry to `GENRE_THEMES` in `main.py`

## Agent tool conventions

All tools in `agent/tools.py` follow this signature:
```python
def tool_name(param: type, context_variables: dict) -> str
```
`context_variables` is injected by the orchestrator — never passed by the LLM.
List parameters are passed as JSON strings to stay within the schema builder's type system.

## Dependencies

Managed with `uv`. Install: `uv sync`

Key libs: `librosa`, `pyrubberband`, `moviepy`, `pydub`, `openai`, `anthropic`, `Pillow`
