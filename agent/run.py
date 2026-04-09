"""
DJ Session Agent — 7-phase orchestrated pipeline.

Agents (in order):
  1. Genre Guard    — validates genre, duration, mood before anything starts
  2. Planner        — proposes playlist (energy arc focused)
  3. Checkpoint 1   — user reviews/adjusts before Critic sees it
  4. Critic         — independent cold review (PROBLEMS / VERDICT)
  5. Checkpoint 2   — user sees critique, decides what to address
  6. Editor REPL    — interactive editing until build
  7. Validator      — auto-triggered after build_session; analyses audio quality

Supports Anthropic, OpenAI, and Ollama — auto-detected from .env:
  ANTHROPIC_API_KEY → Claude (default: claude-opus-4-6)
  OPENAI_API_KEY    → GPT   (default: gpt-4o)
  AGENT_PROVIDER=ollama → local Ollama (default: gemma4:4b)

Override: AGENT_MODEL=gpt-4o-mini python agent/run.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agent.tools import (
    TOOLS,
    _bpm_diff_bucket,
    list_genres,
    get_catalog,
    get_energy_arc,
    redetect_bpm,
    propose_playlist,
    show_playlist,
    analyze_transition,
    swap_track,
    move_track,
    suggest_bridge_track,
    insert_bridge_track,
    build_session,
    catalog_status,
    rebuild_catalog,
    fix_incomplete,
    validate_audio,
    read_memory,
    write_session_record,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/", timeout=1)
        return True
    except Exception:
        return False


_HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

_PROVIDER = os.getenv("AGENT_PROVIDER", "")
if not _PROVIDER:
    if _HAS_ANTHROPIC:
        _PROVIDER = "anthropic"
    elif _HAS_OPENAI:
        _PROVIDER = "openai"
    elif os.getenv("OLLAMA_BASE_URL") or _ollama_running():
        _PROVIDER = "ollama"
    else:
        _PROVIDER = "anthropic"  # will fail with a clear missing-API-key error

_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-4o",
    "ollama": "gemma4:4b",
}.get(_PROVIDER, "claude-opus-4-6")
_MODEL = os.getenv("AGENT_MODEL", _DEFAULT_MODEL)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GENRE_GUARD_SYSTEM = """\
You are a DJ set intake assistant. Your only job is to confirm the user's
genre, duration, and mood before any planning starts.

Workflow:
1. Try to extract genre, duration, and mood directly from the user's message.
   Common shorthands: "lofi" → "lofi - ambient", "deep house", "techno", "cyberpunk".
2. If the genre is ambiguous or not recognized, call list_genres and ask the user
   to pick. Be friendly. Repeat until they pick a valid one.
3. Confirm all three values with the user:
   genre / duration in minutes / mood description
4. Once the user confirms, output EXACTLY this block and nothing else:

CONFIRMED
genre: <exact folder name, lowercase>
duration_min: <integer>
mood: <user's description>

Rules:
- Never call any tool other than list_genres.
- Never proceed or guess with an invalid genre.
- One question at a time. Keep it brief.
"""

_PLANNER_SYSTEM = """\
You are an expert DJ curator specializing in set planning.

Your job: given a genre, duration, and mood, build a compelling playlist
using get_catalog then propose_playlist.

Think about:
- Energy arc: sets need shape — warmup, build, peak, release.
- Variety: avoid clustering the same key or tempo many tracks in a row.
- Duration: fill the target time without unnecessary padding.

Workflow:
1. Call get_catalog to survey the available tracks.
2. Call propose_playlist with genre, duration_min, and mood.
3. Call get_energy_arc to evaluate the set shape.
   - If it flags a plateau, a missing peak (no track ≥ 7 energy), or no wind-down,
     call swap_track or move_track to fix it, then call get_energy_arc again to confirm.
4. Explain your energy arc rationale in 2-3 sentences. Then stop.

Do NOT call show_playlist or build_session.
Do NOT ask questions — make confident choices.
"""

_CHECKPOINT_SYSTEM = """\
You are a collaborative checkpoint assistant in a DJ set pipeline.
Your job: show the current playlist and help the user make quick adjustments
before passing it to the next stage.

Workflow:
1. Call show_playlist immediately.
2. If you received critic feedback, present it clearly before asking for input.
3. Ask the user if they want any changes. ONLY make changes the user
   explicitly requests — never apply Critic recommendations automatically.
   After any change, call show_playlist again to confirm.
4. When the user signals they are done (says "proceed", "ok", "looks good",
   "continue", "done", "next", or just presses Enter), output exactly:
   PROCEED

Rules:
- Available tools: show_playlist, get_catalog, analyze_transition, swap_track, move_track.
- To find a replacement track:
  1. Call get_catalog("current") — this returns all tracks with their real IDs.
  2. Pick a candidate by BPM/key that fits the surrounding positions.
  3. Call swap_track with the real ID from step 1 (e.g. "lofi-ambient--quiet-pages").
  Never invent or guess track IDs — they must come from get_catalog output.
- analyze_transition takes catalog IDs (from get_catalog), NOT position numbers.
- Never call build_session or propose_playlist.
- Never make unsolicited changes. Wait for explicit user instruction.
- Keep responses short. You are a checkpoint, not a full editor.
"""

_CRITIC_SYSTEM = """\
You are a harsh but constructive DJ set critic. Your job is to find problems,
not validate choices.

Workflow:
1. Call show_playlist to see the full set with transition data.
2. Call get_energy_arc to evaluate the energy shape of the set.
   Flag any plateau, missing peak, or missing wind-down as a problem.
3. Call analyze_transition on any pair flagged with ⚠ or that looks suspicious.
4. Deliver your verdict using this exact format — no prose before it:

PROBLEMS:
- [pos X→Y] <specific problem> — fix: <concrete action>

VERDICT: APPROVED / NEEDS_FIXES / REJECT

Rules:
- Be specific. Name positions, keys, BPMs. "sounds rough" is not useful.
- If no problems: write "PROBLEMS: none" and "VERDICT: APPROVED".
- Do NOT swap, move, or propose tracks. Only critique.
- Do NOT add prose after the VERDICT line.
"""

_EDITOR_SYSTEM = """\
You are a professional DJ editor helping refine a set after planning and critique.

Available actions:
- show_playlist: display current state with transition analysis
- analyze_transition: diagnose a specific pair by track ID
- swap_track: replace a track at a position
- move_track: reorder tracks
- suggest_bridge_track: find bridge candidates between two BPM-mismatched positions
- insert_bridge_track: insert a bridge track after a given position
- build_session: save and render the final mix (only on explicit user confirmation)

Always call show_playlist after any swap or move.
Be concise. Think like a DJ.

If a transition has BPM stretch ratio >1.5×, call suggest_bridge_track(from_pos, to_pos) \
to find candidates, then insert_bridge_track(after_position, track_id) to smooth the gap.
"""

_CATALOG_MANAGER_SYSTEM = """\
You are a track catalog manager for a DJ set generator.

Your job: keep tracks.json complete and in sync with WAV files on disk.

There are three problems you fix:
1. NEW FILES — WAV files on disk not yet in the catalog → fix with rebuild_catalog
2. INCOMPLETE ENTRIES — tracks already in catalog but missing BPM or Camelot key → fix with fix_incomplete
3. ORPHANED ENTRIES — catalog entries whose WAV file no longer exists → report only, cannot auto-fix

Workflow:
1. Always call catalog_status first to show the full picture.
2. For new files: confirm with the user, then call rebuild_catalog.
3. For incomplete entries: confirm with the user, then call fix_incomplete
   (re-runs BPM and key detection on those specific tracks only).
4. Call catalog_status again after any fix to confirm the result.

Rules:
- Never rebuild or fix without first showing catalog_status.
- Never skip fix_incomplete if there are entries with missing BPM or key —
  those tracks will be excluded from sets or cause bad transitions.
- Be concise. Report what changed, with numbers.
"""

_VALIDATOR_SYSTEM = """\
You are an audio quality validator. Analyze the exported mix for technical problems.

Workflow:
1. Call validate_audio with the session_name from context.
2. Report findings clearly:

AUDIO QUALITY REPORT — <session_name>
Status: PASS / WARNING / FAIL

Issues:
- [MM:SS] <description>   (or "none" if clean)

Recommendations:
- <actionable fix>   (or "none required")

Rules:
- Be specific about timestamps.
- Do NOT rebuild or re-export. Only report.
"""

# ---------------------------------------------------------------------------
# Proceed signals (checked before hitting the LLM)
# ---------------------------------------------------------------------------

_PROCEED_SIGNALS = {
    "", "proceed", "ok", "looks good", "continue", "done",
    "next", "lgtm", "ship it", "yes", "y", "fine", "good",
}

# ---------------------------------------------------------------------------
# Tool schema builders
# ---------------------------------------------------------------------------

_TYPE_MAP = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _python_type_to_json(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    name = getattr(annotation, "__name__", str(annotation))
    return _TYPE_MAP.get(name, "string")


def _parse_arg_docs(doc: str) -> dict[str, str]:
    arg_docs: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            if line.startswith("    ") and ":" in line:
                name, _, desc = line.strip().partition(":")
                arg_docs[name.strip()] = desc.strip()
            elif line.strip() and not line.startswith(" "):
                in_args = False
    return arg_docs


def _build_properties(fn) -> tuple[dict, list[str]]:
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    arg_docs = _parse_arg_docs(doc)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "context_variables":
            continue
        prop: dict[str, Any] = {"type": _python_type_to_json(param.annotation)}
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return properties, required


def _build_anthropic_schemas(tool_fns: list) -> list[dict]:
    schemas = []
    for fn in tool_fns:
        doc = inspect.getdoc(fn) or ""
        description = doc.split("\n\n")[0].strip()
        properties, required = _build_properties(fn)
        schemas.append({
            "name": fn.__name__,
            "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required},
        })
    return schemas


def _build_openai_schemas(tool_fns: list) -> list[dict]:
    schemas = []
    for fn in tool_fns:
        doc = inspect.getdoc(fn) or ""
        description = doc.split("\n\n")[0].strip()
        properties, required = _build_properties(fn)
        schemas.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return schemas


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _run_tool(name: str, inputs: dict, context_variables: dict, tool_index: dict) -> str:
    fn = tool_index.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(**inputs, context_variables=context_variables)
    except Exception as exc:
        return f"Tool error: {exc}"


# ---------------------------------------------------------------------------
# Generic agent runner
# ---------------------------------------------------------------------------

def run_agent(
    system_prompt: str,
    tool_fns: list,
    messages: list[dict],
    context_variables: dict,
    *,
    max_turns: int = 20,
) -> str:
    """Run one agent's tool loop until it stops calling tools. Returns final text."""
    tool_index = {fn.__name__: fn for fn in tool_fns}
    if _PROVIDER == "anthropic":
        return _run_agent_anthropic(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns)
    if _PROVIDER == "ollama":
        return _run_agent_ollama(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns)
    return _run_agent_openai(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns)


def _run_agent_anthropic(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns):
    import anthropic as _anthropic
    client = _anthropic.Anthropic()
    schemas = _build_anthropic_schemas(tool_fns)
    final_text = ""

    for _ in range(max_turns):
        response = client.messages.create(
            model=_MODEL, max_tokens=4096,
            system=system_prompt, tools=schemas, messages=messages,
        )
        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        final_text = "".join(text_parts)
        messages.append({"role": "assistant", "content": response.content})

        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            print(f"  [tool] {tu.name}({json.dumps(tu.input, ensure_ascii=False)})")
            result = _run_tool(tu.name, tu.input, context_variables, tool_index)
            print(f"  → {result}\n")
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return final_text


def _run_agent_ollama(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns):
    """Run agent against a local Ollama instance using the OpenAI-compatible API."""
    from openai import OpenAI
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = OpenAI(base_url=base_url, api_key="ollama")  # key unused by Ollama
    schemas = _build_openai_schemas(tool_fns)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    final_text = ""

    for _ in range(max_turns):
        kwargs: dict = {"model": _MODEL, "messages": full_messages}
        if schemas:
            kwargs["tools"] = schemas
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        final_text = msg.content or ""
        full_messages.append(msg)
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls or None})

        if not tool_calls:
            break

        for tc in tool_calls:
            inputs = json.loads(tc.function.arguments)
            print(f"  [tool] {tc.function.name}({json.dumps(inputs, ensure_ascii=False)})")
            result = _run_tool(tc.function.name, inputs, context_variables, tool_index)
            print(f"  → {result}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            full_messages.append(tool_msg)
            messages.append(tool_msg)

    return final_text


def _run_agent_openai(system_prompt, tool_fns, tool_index, messages, context_variables, max_turns):
    from openai import OpenAI
    client = OpenAI()
    schemas = _build_openai_schemas(tool_fns)
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    final_text = ""

    for _ in range(max_turns):
        response = client.chat.completions.create(model=_MODEL, tools=schemas, messages=full_messages)
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        final_text = msg.content or ""
        full_messages.append(msg)
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls or None})

        if not tool_calls:
            break

        for tc in tool_calls:
            inputs = json.loads(tc.function.arguments)
            print(f"  [tool] {tc.function.name}({json.dumps(inputs, ensure_ascii=False)})")
            result = _run_tool(tc.function.name, inputs, context_variables, tool_index)
            print(f"  → {result}\n")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
            full_messages.append(tool_msg)
            messages.append(tool_msg)

    return final_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_critic_response(
    text: str, playlist: list[dict] | None = None
) -> tuple[str, list[str], list[dict]]:
    """Extract (verdict, problems_list, structured_problems) from Critic output.

    structured_problems: list of {pos_from, pos_to, key_pair, bpm_diff, text} dicts.
    Uses playlist to look up keys/BPMs by position when available.
    """
    import re

    verdict = "APPROVED"
    problems: list[str] = []
    structured: list[dict] = []
    in_problems = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip().upper()
            in_problems = False
        elif stripped.upper().startswith("PROBLEMS:"):
            remainder = stripped.split(":", 1)[1].strip()
            in_problems = True
            if remainder.lower() == "none":
                in_problems = False
        elif in_problems and stripped.startswith("-"):
            problem_text = stripped[1:].strip()
            problems.append(problem_text)

            # Try to extract pos_from, pos_to from [pos X→Y]
            m = re.search(r'\[pos\s+(\d+)\s*[→\-]\s*(\d+)\]', problem_text, re.IGNORECASE)
            if m and playlist:
                pos_from = int(m.group(1))
                pos_to = int(m.group(2))
                a = playlist[pos_from - 1] if 1 <= pos_from <= len(playlist) else {}
                b = playlist[pos_to - 1] if 1 <= pos_to <= len(playlist) else {}
                key_a = a.get("camelot_key", "?")
                key_b = b.get("camelot_key", "?")
                bpm_a = a.get("bpm") or 0
                bpm_b = b.get("bpm") or 0
                structured.append({
                    "pos_from": pos_from,
                    "pos_to": pos_to,
                    "key_pair": f"{key_a}→{key_b}",
                    "bpm_diff": round(abs(float(bpm_a) - float(bpm_b)), 1),
                    "text": problem_text,
                })

    return verdict, problems, structured


def _parse_validator_response(text: str) -> tuple[str, list[str]]:
    """Extract (status, issues_list) from Validator's AUDIO QUALITY REPORT."""
    status = "PASS"
    issues: list[str] = []
    in_issues = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("status:"):
            status = stripped.split(":", 1)[1].strip().upper()
        elif stripped.lower().startswith("issues"):
            in_issues = True
        elif in_issues and stripped.startswith("-"):
            issues.append(stripped[1:].strip())
        elif in_issues and stripped.lower().startswith("recommendations"):
            in_issues = False
    return status, issues


def _parse_confirmed_block(text: str) -> dict | None:
    """Parse Genre Guard's CONFIRMED block → {genre, duration_min, mood} or None."""
    lines = [l.strip() for l in text.splitlines()]
    try:
        idx = lines.index("CONFIRMED")
    except ValueError:
        return None
    result: dict = {}
    for line in lines[idx + 1:]:
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    if not {"genre", "duration_min", "mood"} <= result.keys():
        return None
    try:
        result["duration_min"] = int(result["duration_min"])
    except (ValueError, KeyError):
        return None
    return result


def _run_checkpoint(context_variables: dict, critic_context: str | None = None) -> None:
    """Interactive checkpoint mini-REPL. Exits on proceed signal or PROCEED sentinel."""
    _CHECKPOINT_TOOLS = [show_playlist, get_catalog, analyze_transition, swap_track, move_track]

    genre = context_variables.get("genre", "")
    genre_note = f" Session genre: {genre}." if genre else ""
    intro = f"A playlist is ready for your review.{genre_note}"
    if critic_context:
        intro = (
            f"The Critic has reviewed the playlist:\n\n{critic_context}\n\n"
            f"You can make adjustments or proceed to the editor.{genre_note}"
        )

    cp_messages: list[dict] = [{"role": "user", "content": intro}]
    response = run_agent(_CHECKPOINT_SYSTEM, _CHECKPOINT_TOOLS, cp_messages, context_variables)
    if response:
        print(f"\n[Checkpoint]\n{response}\n")
    if "PROCEED" in response:
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if user_input.lower() in _PROCEED_SIGNALS:
            break

        cp_messages.append({"role": "user", "content": user_input})
        response = run_agent(_CHECKPOINT_SYSTEM, _CHECKPOINT_TOOLS, cp_messages, context_variables)
        if response:
            print(f"\n[Checkpoint]\n{response}\n")
        if "PROCEED" in response:
            break


# ---------------------------------------------------------------------------
# Orchestrator — 7 phases
# ---------------------------------------------------------------------------

_EDITOR_TOOLS = [show_playlist, analyze_transition, swap_track, move_track, suggest_bridge_track, insert_bridge_track, build_session]
_CATALOG_TOOLS = [catalog_status, rebuild_catalog, fix_incomplete, redetect_bpm]

# Keywords that signal the user wants catalog management
_CATALOG_KEYWORDS = {
    "new song", "new songs", "new track", "new tracks", "added", "add track",
    "catalog", "catalogue", "update catalog", "missing", "sync", "import",
}


def _wants_catalog(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _CATALOG_KEYWORDS)


def _catalog_needs_sync() -> bool:
    """Return True if catalog is missing or has WAV files not yet cataloged."""
    _catalog = Path(__file__).parent.parent / "tracks" / "tracks.json"
    _tracks_dir = Path(__file__).parent.parent / "tracks"
    if not _catalog.exists():
        return True
    try:
        with open(_catalog) as f:
            data = json.load(f)
        cataloged = {e["file"] for e in data.get("tracks", [])}
        for folder in os.listdir(_tracks_dir):
            folder_path = _tracks_dir / folder
            if not folder_path.is_dir():
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".wav"):
                    rel = f"tracks/{folder}/{fname}"
                    if rel not in cataloged:
                        return True
    except Exception:
        pass
    return False


def _run_catalog_manager() -> None:
    """Interactive Catalog Manager agent loop."""
    context_variables: dict = {}
    print("\n── Catalog Manager ──\n")

    messages: list[dict] = [
        {"role": "user", "content": "Check the catalog status and help me sync any new tracks."}
    ]

    while True:
        response = run_agent(
            _CATALOG_MANAGER_SYSTEM, _CATALOG_TOOLS, messages, context_variables
        )
        if response:
            print(f"\nAgent: {response}\n")

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if user_input.lower() in ("quit", "exit", "q", "done"):
            print("Bye.")
            return

        messages.append({"role": "user", "content": user_input})


def _orchestrate() -> None:
    context_variables: dict = {}

    # Safe defaults for scope variables used across phases
    memory_summary = ""
    initial_playlist_names: set[str] = set()
    critic_verdict = "APPROVED"
    critic_problems: list[str] = []
    validator_status = "PASS"
    validator_issues: list[str] = []

    print(f"DJ Set Builder [{_PROVIDER} / {_MODEL}]\n")
    print("What would you like to do?")
    print("  Plan a DJ set, or manage the track catalog (add new songs)?\n")

    try:
        first_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return

    if not first_input:
        return

    if _wants_catalog(first_input):
        _run_catalog_manager()
        # Hand off to DJ set builder after catalog work is done
        print("\n── Catalog up to date. Ready to build a set? ──\n")
        try:
            first_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not first_input or first_input.lower() in ("quit", "exit", "q", "done"):
            return

    # ── CATALOG FRESHNESS CHECK ─────────────────────────────────────────────
    # Proactively ask if catalog is empty or has unsynced WAV files, so
    # first-time users don't hit a dead end without knowing the trigger keyword.
    elif _catalog_needs_sync():
        print("[Catalog] New or missing tracks detected.")
        try:
            sync_reply = input("Sync the catalog before building a set? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if sync_reply in ("yes", "y"):
            _run_catalog_manager()
            print()  # blank line, then fall through to Genre Guard with original first_input

    # ── PHASE 1: GENRE GUARD ────────────────────────────────────────────────
    print("── Genre Guard: let's set up your session ──\n")
    guard_messages: list[dict] = [
        {"role": "user", "content": first_input}  # reuse the startup intent
    ]
    confirmed: dict | None = None

    while confirmed is None:
        guard_response = run_agent(
            _GENRE_GUARD_SYSTEM, [list_genres], guard_messages, context_variables
        )
        if guard_response:
            print(f"\n[Genre Guard] {guard_response}\n")

        confirmed = _parse_confirmed_block(guard_response)
        if confirmed is None:
            try:
                user_reply = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return
            guard_messages.append({"role": "user", "content": user_reply})

    context_variables["genre"] = confirmed["genre"]
    context_variables["duration_min"] = confirmed["duration_min"]
    context_variables["mood"] = confirmed["mood"]
    print(
        f"\n✓ Genre: {confirmed['genre']} | "
        f"{confirmed['duration_min']} min | "
        f"Mood: \"{confirmed['mood']}\"\n"
    )

    # Load memory for this genre (used by Planner and Critic)
    memory_summary = read_memory(genre=confirmed["genre"], context_variables=context_variables)
    if "No memory" not in memory_summary:
        print(f"[Memory loaded]\n{memory_summary}")

    # ── PHASE 2: PLANNER ────────────────────────────────────────────────────
    print("── Planner: building playlist... ──\n")
    planner_prompt = (
        f"Build a {confirmed['duration_min']}-minute {confirmed['genre']} set. "
        f"Mood: {confirmed['mood']}"
    )
    if "No memory" not in memory_summary:
        planner_prompt += f"\n\nPAST SESSION MEMORY:\n{memory_summary}"
    planner_messages: list[dict] = [{"role": "user", "content": planner_prompt}]
    planner_response = run_agent(
        _PLANNER_SYSTEM, [get_catalog, propose_playlist, get_energy_arc, swap_track, move_track], planner_messages, context_variables
    )
    if planner_response:
        print(f"\n[Planner]\n{planner_response}\n")

    if not context_variables.get("playlist"):
        print("Error: Planner did not produce a playlist. Try again.")
        return

    # Snapshot initial playlist for swap tracking
    initial_playlist_names = {t["display_name"] for t in context_variables["playlist"]}

    # ── PHASE 3: CHECKPOINT 1 (after Planner, before Critic) ────────────────
    print("── Checkpoint 1: review before Critic ──\n")
    _run_checkpoint(context_variables, critic_context=None)

    # ── PHASE 4: CRITIC ─────────────────────────────────────────────────────
    print("\n── Critic: independent review... ──\n")
    critic_brief = "A playlist has been proposed. Review it and deliver your verdict."
    if "No memory" not in memory_summary:
        critic_brief += f"\n\nPAST SESSION MEMORY (user preferences):\n{memory_summary}"
    critic_messages: list[dict] = [{"role": "user", "content": critic_brief}]
    critic_response = run_agent(
        _CRITIC_SYSTEM, [show_playlist, analyze_transition, get_energy_arc], critic_messages, context_variables
    )
    if critic_response:
        print(f"\n[Critic]\n{critic_response}\n")
    critic_verdict, critic_problems, critic_structured_problems = _parse_critic_response(
        critic_response, playlist=context_variables.get("playlist", [])
    )

    # ── PHASE 5: CHECKPOINT 2 (after Critic, before Editor) ─────────────────
    print("── Checkpoint 2: review Critic findings ──\n")
    _run_checkpoint(context_variables, critic_context=critic_response)

    # ── PHASE 6: EDITOR REPL ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Editor ready. Request changes or type 'build <name>' to render.\n")

    editor_messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Set: {confirmed['duration_min']}-min {confirmed['genre']} | mood: {confirmed['mood']}\n\n"
                f"Planner notes:\n{planner_response}\n\n"
                f"Critic review:\n{critic_response}\n\n"
                "User may now request changes."
            ),
        },
        {"role": "assistant", "content": "Ready to refine the set."},
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if not user_input:
            continue

        editor_messages.append({"role": "user", "content": user_input})
        editor_response = run_agent(_EDITOR_SYSTEM, _EDITOR_TOOLS, editor_messages, context_variables)
        if editor_response:
            print(f"\nAgent: {editor_response}\n")

        # ── PHASE 7: VALIDATOR (auto-triggered after build_session) ──────────
        if context_variables.get("last_build"):
            session_name = context_variables.pop("last_build")
            print("\n── Validator: analysing audio quality... ──\n")
            validator_messages: list[dict] = [
                {
                    "role": "user",
                    "content": f"Session '{session_name}' was just built. Validate its audio quality.",
                }
            ]
            context_variables["last_build"] = session_name
            validator_response = run_agent(
                _VALIDATOR_SYSTEM, [validate_audio], validator_messages, context_variables
            )
            if validator_response:
                print(f"\n[Validator]\n{validator_response}\n")
            validator_status, validator_issues = _parse_validator_response(validator_response)
            context_variables.pop("last_build", None)

            # Compute swapped tracks
            final_playlist = context_variables.get("playlist", [])
            tracks_swapped = sorted(
                initial_playlist_names - {t["display_name"] for t in final_playlist}
            )

            # ── PHASE 8: RATING ──────────────────────────────────────────────
            print("── Rate this session ──\n")
            rating = 0
            while True:
                try:
                    raw = input("Rate 1-5 (Enter to skip): ").strip()
                    if not raw:
                        break
                    val = int(raw)
                    if 1 <= val <= 5:
                        rating = val
                        break
                    print("  Please enter a number between 1 and 5.")
                except (ValueError, EOFError, KeyboardInterrupt):
                    break

            notes = ""
            if rating > 0:
                try:
                    notes = input("Any notes? (Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    notes = ""

            # Per-transition ratings
            transition_ratings: list[dict] = []
            if len(final_playlist) >= 2:
                print("\n── Rate individual transitions (y to rate, Enter to skip) ──\n")
                try:
                    do_tr = input("Rate transitions? (y/Enter to skip): ").strip().lower()
                    if do_tr == "y":
                        for i in range(len(final_playlist) - 1):
                            a = final_playlist[i]
                            b = final_playlist[i + 1]
                            a_name = a.get("display_name", f"pos {i+1}")
                            b_name = b.get("display_name", f"pos {i+2}")
                            a_key = a.get("camelot_key", "?")
                            b_key = b.get("camelot_key", "?")
                            a_bpm = float(a.get("bpm") or 0)
                            b_bpm = float(b.get("bpm") or 0)
                            bpm_diff = abs(a_bpm - b_bpm)
                            while True:
                                try:
                                    raw = input(
                                        f"  {a_name} → {b_name} "
                                        f"({a_key}→{b_key}) 1-5 (Enter=skip): "
                                    ).strip()
                                    if not raw:
                                        break
                                    val = int(raw)
                                    if 1 <= val <= 5:
                                        transition_ratings.append({
                                            "from": a_name,
                                            "to": b_name,
                                            "key_pair": f"{a_key}→{b_key}",
                                            "bpm_diff_bucket": _bpm_diff_bucket(bpm_diff),
                                            "rating": val,
                                        })
                                        break
                                    print("  1-5 only.")
                                except (ValueError, EOFError, KeyboardInterrupt):
                                    break
                except (EOFError, KeyboardInterrupt):
                    pass

            result = write_session_record(
                session_name=session_name,
                genre=context_variables.get("genre", ""),
                duration_min=context_variables.get("duration_min", 0),
                mood=context_variables.get("mood", ""),
                rating=rating,
                notes=notes,
                critic_verdict=critic_verdict,
                critic_problems_json=json.dumps(critic_problems),
                validator_status=validator_status,
                validator_issues_json=json.dumps(validator_issues),
                tracks_swapped_json=json.dumps(tracks_swapped),
                final_playlist_json=json.dumps(
                    [t["display_name"] for t in final_playlist]
                ),
                transition_ratings_json=json.dumps(transition_ratings),
                structured_problems_json=json.dumps(critic_structured_problems),
                context_variables=context_variables,
            )
            print(f"\n{result}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    if _PROVIDER == "ollama":
        print(f"[Provider: Ollama / {_MODEL}]")
    elif not _HAS_ANTHROPIC and not _HAS_OPENAI:
        print("Error: set ANTHROPIC_API_KEY, OPENAI_API_KEY, or AGENT_PROVIDER=ollama in .env")
        sys.exit(1)
    _orchestrate()


if __name__ == "__main__":
    run()
