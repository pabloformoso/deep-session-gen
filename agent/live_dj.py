"""
LiveDJ — proactive event-driven DJ agent for Apollo v1.5.

Architecture:
  - run_live_session(): main loop that drains engine events + user input,
    batches them into LLM turns, and executes tool calls.
  - Events arrive from LiveEngine via a threading.Queue.
  - User input is read in a daemon thread to avoid blocking the event loop.
  - The LLM is called with a tight max_turns budget per event batch (5 turns),
    preventing runaway token spend while staying responsive.

Circular-import note:
  run_agent() lives in agent/run.py which also imports from agent/tools.py.
  live_dj.py is imported by agent/tools.py (via start_live_session), so we
  defer the import of run_agent() inside the function to break the cycle.
"""
from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Any

from agent.live_engine import (
    APPROACHING_CF,
    CROSSFADE_FINISHED,
    CROSSFADE_TRIGGERED,
    SESSION_ENDED,
    TRACK_ENDED,
    TRACK_STARTED,
    LiveEngine,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_LIVE_DJ_SYSTEM = """\
You are Apollo LiveDJ — a proactive AI DJ performing a live set right now.

Music is playing. Your job is to make real-time decisions:

TOOLS:
- get_live_state(): current track, position, BPM, Camelot key, seconds to crossfade
- crossfade_now(): trigger crossfade immediately
- extend_track(seconds): delay the upcoming auto-crossfade by N seconds
- skip_track(): hard-cut to next track without crossfade
- queue_swap(position, track_id): swap a future track (use get_catalog to find IDs)
- set_crossfade_point(position_sec): manually set where crossfade begins

DECISION RULES:
approaching_crossfade event:
  → Call get_live_state() to assess: BPM diff, Camelot distance, seconds remaining.
  → Good transition (Camelot ≤1 step, BPM diff ≤8): confirm silently (no tool call needed).
  → Mediocre transition (Camelot 2 steps OR BPM diff 8–20): extend_track(20) to buy time.
  → Bad transition (Camelot >2 steps OR BPM diff >20): crossfade_now() to escape quickly,
    or queue_swap() a better track if there's time.

crossfade_finished event:
  → Log it. Update energy arc awareness. No action needed unless user asked for change.

User commands:
  "next" / "skip"         → crossfade_now()
  "stay" / "longer"       → extend_track(30)
  "more energetic"        → queue_swap() with higher BPM track
  "wind down" / "chill"   → queue_swap() with lower BPM / further Camelot key

STYLE:
- Be terse. Think in beats, not paragraphs.
- Short confirmations only ("Crossfading now." / "Extended 30s." / "Swapped track 4.").
- Never explain the rules. Just act.
"""

# ---------------------------------------------------------------------------
# Live tools (use engine from context_variables["_engine"])
# ---------------------------------------------------------------------------

def get_live_state(context_variables: dict) -> str:
    """Return current engine state: position, track, BPM, Camelot key, time to crossfade."""
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    s = engine.get_state()
    cur = s["current_track"] or {}
    nxt = s["next_track"] or {}
    lines = [
        f"State: {s['state']}",
        f"Position: {s['position_sec']}s",
        f"Current: {cur.get('display_name','?')} — {cur.get('bpm','?')} BPM, {cur.get('camelot_key','?')}",
        f"Next:    {nxt.get('display_name','?')} — {nxt.get('bpm','?')} BPM, {nxt.get('camelot_key','?')}",
        f"Crossfade in: {s['seconds_to_crossfade']}s",
        f"Tracks remaining: {s['playlist_remaining']}",
    ]
    if cur.get("hot_cues"):
        lines.append(f"Hot cues (current): {cur['hot_cues']}")
    return "\n".join(lines)


def crossfade_now(context_variables: dict) -> str:
    """Trigger crossfade immediately."""
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    return engine.crossfade_now()


def extend_track(seconds: int, context_variables: dict) -> str:
    """Delay the upcoming auto-crossfade by seconds seconds.

    Args:
        seconds: Number of seconds to delay the crossfade.
    """
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    return engine.extend_track(seconds)


def skip_track(context_variables: dict) -> str:
    """Hard-cut to next track without crossfade."""
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    return engine.skip_track()


def queue_swap(position: int, track_id: str, context_variables: dict) -> str:
    """Replace a future playlist slot with a catalog track.

    Args:
        position: 1-indexed future playlist position to replace.
        track_id: Catalog track ID to insert (use get_catalog to find IDs).
    """
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    return engine.queue_swap(position, track_id)


def set_crossfade_point(position_sec: float, context_variables: dict) -> str:
    """Set where in the current track the crossfade begins.

    Args:
        position_sec: Target crossfade start, in seconds from track start.
    """
    engine: LiveEngine = context_variables.get("_engine")
    if not engine:
        return "Engine not running."
    return engine.set_crossfade_point(position_sec)


_LIVE_TOOLS = [
    get_live_state,
    crossfade_now,
    extend_track,
    skip_track,
    queue_swap,
    set_crossfade_point,
]

# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

def run_live_session(playlist: list[dict], context_variables: dict) -> None:
    """Start the LiveDJ session: spin up the engine, run the agent event loop.

    Blocks until the session ends (all tracks played, user quits, or engine stops).
    Stores engine in context_variables["_engine"] so live tools can reach it.
    """
    # Deferred import to break circular dependency (tools → live_dj → run)
    from agent.run import run_agent  # noqa: PLC0415

    event_queue: Queue = Queue()
    user_input_queue: Queue = Queue()

    engine = LiveEngine(playlist, event_queue)
    context_variables["_engine"] = engine

    # Daemon thread reads blocking stdin without stalling the event loop
    input_thread = threading.Thread(
        target=_stdin_reader, args=(user_input_queue,), daemon=True, name="live-stdin"
    )
    input_thread.start()

    print("\n── Apollo LiveDJ ──")
    print("Commands: next | stay [N] | skip | quit | or anything natural language\n")

    engine.play()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Live session started.\n"
                + _playlist_summary(playlist)
            ),
        },
        {"role": "assistant", "content": "On deck. Let's go."},
    ]

    while True:
        time.sleep(0.1)

        events = _drain(event_queue)
        user_inputs = _drain(user_input_queue, limit=1)

        # Hard quit from user
        if any(u.strip().lower() in ("quit", "exit", "q") for u in user_inputs):
            print("\n[LiveDJ] Stopping session.")
            break

        # Session over
        if any(e["type"] == SESSION_ENDED for e in events):
            print("\n[LiveDJ] Set complete. Good night.")
            break

        if not events and not user_inputs:
            continue

        content = _format_turn(events, user_inputs, engine.get_state())
        messages.append({"role": "user", "content": content})

        response = run_agent(
            _LIVE_DJ_SYSTEM,
            _LIVE_TOOLS,
            messages,
            context_variables,
            max_turns=5,
        )
        if response:
            messages.append({"role": "assistant", "content": response})
            print(f"\n[LiveDJ] {response}\n")

    engine.stop()
    context_variables.pop("_engine", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stdin_reader(q: Queue) -> None:
    """Blocking stdin reader — runs in a daemon thread."""
    while True:
        try:
            line = input("You: ").strip()
            if line:
                q.put(line)
        except (EOFError, KeyboardInterrupt):
            break


def _drain(q: Queue, limit: int = 64) -> list[Any]:
    """Non-blocking drain of a Queue, up to `limit` items."""
    items = []
    for _ in range(limit):
        try:
            items.append(q.get_nowait())
        except Empty:
            break
    return items


def _format_turn(
    events: list[dict], user_inputs: list[str], state: dict
) -> str:
    """Build the user-role message for one agent turn from events + input."""
    parts: list[str] = []

    if events:
        parts.append("=== Engine events ===")
        for ev in events:
            parts.append(_format_event(ev))

    if user_inputs:
        parts.append("=== Listener ===")
        for u in user_inputs:
            parts.append(f"  > {u}")

    parts.append(f"=== Current state ===")
    cur = state.get("current_track") or {}
    parts.append(
        f"  {cur.get('display_name','?')} | "
        f"{cur.get('bpm','?')} BPM | {cur.get('camelot_key','?')} | "
        f"{state.get('position_sec','?')}s | "
        f"CF in {state.get('seconds_to_crossfade','?')}s | "
        f"{state.get('playlist_remaining','?')} tracks left"
    )
    return "\n".join(parts)


def _format_event(ev: dict) -> str:
    t = ev["type"]
    if t == TRACK_STARTED:
        tr = ev.get("track") or {}
        return f"  TRACK_STARTED: '{tr.get('display_name','?')}' ({tr.get('bpm','?')} BPM, {tr.get('camelot_key','?')})"
    if t == APPROACHING_CF:
        tr = ev.get("track") or {}
        nx = ev.get("next_track") or {}
        sec = ev.get("seconds_remaining", "?")
        return (
            f"  APPROACHING_CF in {sec}s: "
            f"'{tr.get('display_name','?')}' → '{nx.get('display_name','?')}' "
            f"({tr.get('bpm','?')}→{nx.get('bpm','?')} BPM, "
            f"{tr.get('camelot_key','?')}→{nx.get('camelot_key','?')})"
        )
    if t == CROSSFADE_TRIGGERED:
        fr = ev.get("from_track") or {}
        to = ev.get("to_track") or {}
        return f"  CROSSFADE_TRIGGERED: '{fr.get('display_name','?')}' → '{to.get('display_name','?')}'"
    if t == CROSSFADE_FINISHED:
        fr = ev.get("from_track") or {}
        to = ev.get("to_track") or {}
        return f"  CROSSFADE_FINISHED: now on '{to.get('display_name','?')}' (was '{fr.get('display_name','?')}')"
    if t == TRACK_ENDED:
        tr = ev.get("track") or {}
        return f"  TRACK_ENDED: '{tr.get('display_name','?')}'"
    if t == SESSION_ENDED:
        return "  SESSION_ENDED"
    return f"  {t}: {ev}"


def _playlist_summary(playlist: list[dict]) -> str:
    lines = [f"Playlist ({len(playlist)} tracks):"]
    for i, t in enumerate(playlist, 1):
        lines.append(
            f"  {i}. {t.get('display_name','?')} — {t.get('bpm','?')} BPM, {t.get('camelot_key','?')}"
        )
    return "\n".join(lines)
