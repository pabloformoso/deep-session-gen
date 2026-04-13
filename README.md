# ApolloAgents

[![CI](https://github.com/pabloformoso/apollo-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/pabloformoso/apollo-agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Roadmap](https://img.shields.io/badge/roadmap-public-blueviolet)](ROADMAP.md)

![ApolloAgents Logo](apollo_agents_logo.png)

> An AI-powered DJ set builder — from track catalog to rendered YouTube video, guided by a team of specialized agents.

ApolloAgents uses a multi-agent pipeline to plan, critique, and build DJ mixes. You describe the vibe. The agents handle harmonic mixing, BPM matching, energy arc planning, and audio quality validation. You stay in control at every checkpoint.

---

## Example Output

Every session in [this YouTube channel](https://www.youtube.com/watch?v=PdTd54Vl8Go) was built with ApolloAgents — from the earliest proof-of-concept cuts in v0.0 to today's fully orchestrated pipeline in v1.0. Same tracks, same taste, progressively better mixing as the agents learned.

<a href="https://www.youtube.com/watch?v=PdTd54Vl8Go">
  <img src="https://img.youtube.com/vi/PdTd54Vl8Go/maxresdefault.jpg" width="480" alt="Watch on YouTube" />
</a>

---

## Architecture

<img src="architecture_infographic.png" width="540" alt="ApolloAgents: The Multi-Agent DJ Architecture" />



```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0d0d1a',
  'primaryColor': '#1a1a2e',
  'primaryTextColor': '#e0e0ff',
  'primaryBorderColor': '#4a4a8a',
  'lineColor': '#6060aa',
  'secondaryColor': '#12122a',
  'tertiaryColor': '#0d0d1a',
  'edgeLabelBackground': '#1a1a2e',
  'clusterBkg': '#12122a',
  'clusterBorder': '#3a3a6a',
  'titleColor': '#c0c0ff',
  'nodeTextColor': '#e0e0ff',
  'fontFamily': 'monospace'
}}}%%

flowchart TD
    User(["👤 User\nprompt"]):::user

    subgraph APOLLO["☀️  APOLLO — Orchestrator"]
        direction TB

        JANUS["🚪 JANUS\nGenre Guard\n─────────────\nvalidates genre · duration · mood"]:::agent
        HERMES["⚡ HERMES\nCatalog Manager\n─────────────\nsyncs WAVs · detects BPM & key"]:::agent

        MUSE["🎵 MUSE\nPlanner\n─────────────\nenergy arc · harmonic order\nreads memory → avoids weak tracks"]:::agent

        CP1{{"🛑 Checkpoint 1\nreview playlist"}}:::checkpoint

        MOMUS["🎭 MOMUS\nCritic\n─────────────\ncold review · PROBLEMS / VERDICT\nreads memory → flags patterns"]:::agent

        CP2{{"🛑 Checkpoint 2\napply fixes"}}:::checkpoint

        EDITOR["✏️ Editor REPL\nswap · move · refine"]:::agent

        PIPELINE[["⚙️ Mix Pipeline\nBPM match → crossfade → WAV\n1080p video + YouTube Short"]]:::pipeline

        THEMIS["⚖️ THEMIS\nValidator\n─────────────\nclipping · spectral flatness\nsilence gaps · RMS drops"]:::agent

        MEMORY[("🧠 Memory\nrating + notes\n→ agents improve")]:::memory
    end

    User --> JANUS
    User --> HERMES
    JANUS -->|"confirmed genre"| MUSE
    MUSE -->|"playlist"| CP1
    CP1 -->|"proceed"| MOMUS
    MOMUS -->|"verdict"| CP2
    CP2 -->|"ok"| EDITOR
    EDITOR -->|"build"| PIPELINE
    PIPELINE --> THEMIS
    THEMIS -->|"PASS"| MEMORY
    MEMORY -.->|"past sessions"| MUSE
    MEMORY -.->|"problem patterns"| MOMUS

    classDef agent        fill:#1a1a3a,stroke:#5858b0,color:#c8c8ff,rx:6
    classDef checkpoint   fill:#2a1a0a,stroke:#c07820,color:#ffc060,shape:diamond
    classDef pipeline     fill:#0a1f1a,stroke:#20a060,color:#60ffb0
    classDef memory       fill:#1a0a2a,stroke:#8040c0,color:#c080ff
    classDef user         fill:#0a0a1a,stroke:#4040a0,color:#8080d0,shape:circle
```

| Agent | Mythological name | Role |
|-------|------------------|------|
| Genre Guard | **Janus** | Gatekeeper — validates genre, duration, mood before planning starts |
| Catalog Manager | **Hermes** | Keeper of records — syncs WAV files to catalog, detects BPM & key |
| Planner | **Muse** | Inspires the set — energy arc, harmonic ordering, track selection |
| Critic | **Momus** | God of fault-finding — cold independent review, structured verdict |
| Editor | *(REPL)* | Interactive editor — swap, move, insert bridge tracks, trigger build |
| Validator | **Themis** | Goddess of order — audio quality analysis after every build |
| Orchestrator | **Apollo** | Conductor — sequences all agents, manages state, collects memory |

### Pipeline phases

```
1. Janus (Genre Guard)   → confirms genre / duration / mood
2. Muse  (Planner)       → proposes playlist + energy arc
3. Checkpoint 1          → user reviews; manual adjustments allowed
4. Momus (Critic)        → cold review: flags key clashes, BPM stretch, arc gaps
5. Checkpoint 2          → user sees critique; decides what to apply
6. Editor REPL           → swap · move · insert bridge tracks → build
7. Themis (Validator)    → audio quality report after build
```

> Checkpoints are hard gates — agents never auto-apply fixes. You stay in control.

---

## Features

- **Conversational planning** — describe the vibe, iterate with the agents, build when ready
- **Harmonic mixing** — Camelot wheel-based track ordering for smooth key transitions
- **BPM matching** — gradual tempo ramps between tracks via pyrubberband
- **BPM stretch safety** — transitions with pyrubberband ratio >1.5× are flagged; Critic mandates a bridge track fix
- **Bridge track insertion** — `suggest_bridge_track` finds candidates between mismatched positions; `insert_bridge_track` splices one in
- **EQ matching at crossfade** — shelving EQ applied to outgoing/incoming segments based on key distance, reducing frequency masking
- **Energy arc planning** — Planner evaluates set shape (warmup → build → peak → wind-down) and iterates until no gaps or plateaus
- **Audio validation** — peak clipping, spectral flatness (bleach detection), silence gap and RMS anomaly checks
- **Per-transition ratings** — rate each session 1–5 after build; Critic memory flags recurring problem transitions
- **Session memory** — agents learn from past sessions: which tracks get swapped, what energy arcs rate highly
- **Catalog management** — scan new WAVs, detect missing BPM/key fields, keep tracks.json in sync
- **Multi-provider** — Claude (Anthropic), GPT-4o (OpenAI), or any local model via Ollama; auto-detected from `.env`
- **1080p video output** — spectral waveform visualizer, beat-reactive particles, DALL-E 3 artwork, retro pixel titles
- **YouTube Short** — auto-generated 20s teaser alongside the full mix

---

## Setup

**Requirements:** Python 3.12+, `uv`, `ffmpeg`

```bash
git clone https://github.com/YOUR_USERNAME/apollo-agents.git
cd apollo-agents

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

**`.env` keys:**

| Key | Required | Purpose |
|-----|----------|---------|
| `ANTHROPIC_API_KEY` | One of these | Claude (recommended, default: `claude-opus-4-6`) |
| `OPENAI_API_KEY` | One of these | GPT-4o — also used for DALL-E 3 artwork |
| `AGENT_PROVIDER=ollama` | One of these | Use a local Ollama model (default: `gemma4:4b`) |
| `OLLAMA_BASE_URL` | Optional | Override Ollama endpoint (default: `http://localhost:11434/v1`) |
| `AGENT_MODEL` | Optional | Override the model for any provider (e.g. `AGENT_MODEL=gpt-4o-mini`) |

---

## Adding Your Tracks

Put WAV files into genre subfolders under `tracks/`:

```
tracks/
  techno/
    Acid Rain.wav
    Zero Day.wav
  deep house/
    Solar Drift.wav
  lofi - ambient/
    Kernel Space.wav
  cyberpunk/
    Chrome Horizon.wav
```

Then build the catalog (detects BPM + Camelot key for each file):

```bash
python main.py --build-catalog
```

Or let **Hermes** do it conversationally:

```bash
uv run python agent/run.py
# → "I added new tracks"
```

---

## Usage

### Conversational agent (recommended)

```bash
# Default (Claude / GPT-4o, whichever key is in .env)
uv run python agent/run.py

# Local model via Ollama (no API key required)
AGENT_PROVIDER=ollama uv run python agent/run.py

# Override model for any provider
AGENT_MODEL=claude-haiku-4-5-20251001 uv run python agent/run.py
```

Example session:
```
What would you like to do?

You: 60min techno set, dark industrial build to a hard peak

── Janus (Genre Guard) ──
[confirms genre: techno, 60min, mood: dark industrial build]

── Muse (Planner) ──
[surveys catalog, proposes 12-track playlist]
[evaluates energy arc: plateau detected at pos 6-8, swaps pos 7 to fix]
Energy arc: 3-track warmup → hard build → peak at pos 9 → wind-down

── Checkpoint 1 ──
You: move track 4 to position 7
[shows updated playlist]
You: proceed

── Momus (Critic) ──
PROBLEMS:
- [pos 2→3] key clash 5A → 11A — fix: swap pos 3 for zero-day
- [pos 8→9] ⚠ Stretch 1.8× — bridge track required
VERDICT: NEEDS_FIXES

── Checkpoint 2 ──
You: swap pos 3 like the critic said
You: ok

── Editor ──
You: fix the stretch at 8→9
[suggest_bridge_track(8, 9) → 3 candidates at 142 BPM]
[insert_bridge_track(after_position=8, track_id="techno--acid-rain")]
You: build midnight-industrial

── Themis (Validator) ──
AUDIO QUALITY REPORT — midnight-industrial
Status: PASS — no issues detected ✓

Rate 1-5 (Enter to skip): 5
Any notes?: peak section was perfect
```

### Direct CLI (no agent)

```bash
# Generate a session directly
python main.py --name "midnight-techno" --genre "techno" --duration 60

# Re-render video from existing mix audio
python main.py --name "midnight-techno" --genre "techno" --video-only

# Fix missing BPM/key fields in catalog
python main.py --fix-incomplete
```

---

## Supported Genres

| Folder name | Visual theme |
|-------------|-------------|
| `techno` | Dark red, industrial |
| `deep house` | Neon violet, deep |
| `lofi - ambient` | Warm cream, anime-style artwork |
| `cyberpunk` | Neon green, dystopic |

Add new genres by creating a subfolder under `tracks/` and running `--build-catalog`.

---

## Output

Every session writes to `output/<session-name>/`:

```
output/midnight-techno/
  mix_output.wav      # lossless mix
  mix_video.mp4       # 1920×1080, 24fps, spectral waveform
  short.mp4           # 1080×1920, 20s YouTube Short
  session.json        # playlist for reproducibility
  transitions.json    # crossfade timestamps
  youtube.md          # title, description, tracklist, tags
```

---

## Agent Memory

After each build, rate your session 1-5. Ratings accumulate in `agent/memory.json`. On the next session of the same genre:

- **Muse (Planner)** avoids tracks that have been swapped out 2+ times
- **Momus (Critic)** flags transition patterns that have been problems before
- High-rated mood/arc combinations are surfaced as references

---

## Project Structure

```
main.py              # Core pipeline (~2600 lines): catalog, mixing, video
agent/
  run.py             # Apollo orchestrator + all agent loops
  tools.py           # Tool functions for all agents
  memory.json        # Persistent session history (auto-created)
tracks/
  tracks.json        # Unified catalog (auto-generated)
  <genre>/           # WAV files per genre
output/              # Generated mixes and videos (gitignored)
artwork/             # DALL-E 3 backgrounds (cached, gitignored)
fonts/
  PressStart2P-Regular.ttf
```

---

## License

MIT — see [LICENSE](LICENSE)
