"""
Async streaming pipeline bridge.

Wraps the 7 agent phases from agent/run.py as async functions that emit
WebSocket events instead of printing to stdout. Imports system prompts,
schema builders, and tool functions directly from the existing agent code.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

# Make the project root importable
_PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

# ---------------------------------------------------------------------------
# Import system prompts, parsers, and schema helpers from existing agent code
# ---------------------------------------------------------------------------
from agent.run import (  # noqa: E402
    _GENRE_GUARD_SYSTEM,
    _PLANNER_SYSTEM,
    _CRITIC_SYSTEM,
    _EDITOR_SYSTEM,
    _VALIDATOR_SYSTEM,
    _parse_confirmed_block,
    _parse_critic_response,
    _parse_validator_response,
    _build_anthropic_schemas,
    _build_openai_schemas,
    _run_tool,
)

# ---------------------------------------------------------------------------
# Import tool functions used in each phase
# ---------------------------------------------------------------------------
from agent.tools import (  # noqa: E402
    list_genres,
    get_catalog,
    propose_playlist,
    show_playlist,
    analyze_transition,
    get_energy_arc,
    swap_track,
    move_track,
    suggest_bridge_track,
    insert_bridge_track,
    build_session,
    validate_audio,
    read_memory,
    write_session_record,
)

# ---------------------------------------------------------------------------
# Provider / model — mirror agent/run.py detection
# ---------------------------------------------------------------------------
_PROVIDER_ENV = os.getenv("AGENT_PROVIDER", "")
_HAS_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if _PROVIDER_ENV == "anthropic" or (_HAS_ANTHROPIC and not _PROVIDER_ENV):
    _PROVIDER = "anthropic"
elif _PROVIDER_ENV == "openai" or (_HAS_OPENAI and not _PROVIDER_ENV):
    _PROVIDER = "openai"
elif _PROVIDER_ENV == "ollama":
    _PROVIDER = "ollama"
else:
    _PROVIDER = "anthropic"

_DEFAULT_MODELS = {"anthropic": "claude-opus-4-6", "openai": "gpt-4o", "ollama": "gemma4:4b"}
_MODEL = os.getenv("AGENT_MODEL", _DEFAULT_MODELS.get(_PROVIDER, "claude-opus-4-6"))

# ---------------------------------------------------------------------------
# Phase tool lists (web-safe: no local playback tools)
# ---------------------------------------------------------------------------
_GENRE_TOOLS = [list_genres, get_catalog]
_PLANNER_TOOLS = [get_catalog, propose_playlist, get_energy_arc, show_playlist]
_CRITIC_TOOLS = [show_playlist, analyze_transition, get_energy_arc]
_WEB_EDITOR_TOOLS = [
    show_playlist, analyze_transition, swap_track, move_track,
    suggest_bridge_track, insert_bridge_track, build_session,
]
_VALIDATOR_TOOLS = [validate_audio]


# ---------------------------------------------------------------------------
# Async streaming agent runner
# ---------------------------------------------------------------------------

async def _run_anthropic_streaming(
    system: str,
    tool_fns: list[Callable],
    messages: list[dict],
    ctx: dict,
    emit: Callable,
    max_turns: int,
) -> str:
    import anthropic  # noqa: PLC0415

    client = anthropic.AsyncAnthropic()
    schemas = _build_anthropic_schemas(tool_fns)
    tool_index = {fn.__name__: fn for fn in tool_fns}
    final_text = ""

    for _ in range(max_turns):
        full_text = ""

        async with client.messages.stream(
            model=_MODEL,
            system=system,
            tools=schemas or [],
            messages=messages,
            max_tokens=4096,
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                await emit({"type": "text_delta", "content": text})
            final_msg = await stream.get_final_message()

        # Serialize content blocks for next-turn messages
        assistant_content = []
        for block in final_msg.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})
        final_text = full_text

        if final_msg.stop_reason != "tool_use":
            break

        tool_results = []
        for block in final_msg.content:
            if block.type == "tool_use":
                await emit({"type": "tool_call", "name": block.name, "input": block.input})
                result = await asyncio.to_thread(
                    _run_tool, block.name, block.input, ctx, tool_index
                )
                await emit({"type": "tool_result", "name": block.name, "result": str(result)})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return final_text


async def _run_openai_streaming(
    system: str,
    tool_fns: list[Callable],
    messages: list[dict],
    ctx: dict,
    emit: Callable,
    max_turns: int,
    base_url: str | None = None,
) -> str:
    import json as _json  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    schemas = _build_openai_schemas(tool_fns)
    tool_index = {fn.__name__: fn for fn in tool_fns}
    final_text = ""

    sys_messages = [{"role": "system", "content": system}] + messages

    for _ in range(max_turns):
        full_text = ""
        tool_calls_acc: dict[int, dict] = {}

        stream = await client.chat.completions.create(
            model=_MODEL,
            messages=sys_messages,
            tools=schemas or [],
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                full_text += delta.content
                await emit({"type": "text_delta", "content": delta.content})
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

        if tool_calls_acc:
            tc_list = [
                {"id": v["id"], "type": "function",
                 "function": {"name": v["name"], "arguments": v["arguments"]}}
                for v in tool_calls_acc.values()
            ]
            sys_messages.append({"role": "assistant", "content": full_text or None, "tool_calls": tc_list})

            results = []
            for tc in tc_list:
                name = tc["function"]["name"]
                try:
                    inputs = _json.loads(tc["function"]["arguments"])
                except Exception:
                    inputs = {}
                await emit({"type": "tool_call", "name": name, "input": inputs})
                result = await asyncio.to_thread(_run_tool, name, inputs, ctx, tool_index)
                await emit({"type": "tool_result", "name": name, "result": str(result)})
                results.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            sys_messages.extend(results)
        else:
            sys_messages.append({"role": "assistant", "content": full_text})
            final_text = full_text
            break

    return final_text


async def run_agent_streaming(
    system: str,
    tool_fns: list[Callable],
    messages: list[dict],
    ctx: dict,
    emit: Callable,
    max_turns: int = 20,
) -> str:
    """Dispatch to the streaming runner for the configured provider."""
    if _PROVIDER == "anthropic":
        return await _run_anthropic_streaming(system, tool_fns, messages, ctx, emit, max_turns)
    if _PROVIDER == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return await _run_openai_streaming(system, tool_fns, messages, ctx, emit, max_turns, base_url=base)
    return await _run_openai_streaming(system, tool_fns, messages, ctx, emit, max_turns)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

async def phase_genre_guard(
    message: str,
    history: list[dict],
    ctx: dict,
    emit: Callable,
) -> dict | None:
    """Run Genre Guard. Returns {genre, duration_min, mood} or None."""
    history.append({"role": "user", "content": message})
    response = await run_agent_streaming(_GENRE_GUARD_SYSTEM, _GENRE_TOOLS, history, ctx, emit)
    history.append({"role": "assistant", "content": response})
    return _parse_confirmed_block(response)


async def phase_plan(ctx: dict, emit: Callable, memory_summary: str = "") -> str:
    """Run Planner. Populates ctx['playlist']."""
    genre = ctx.get("genre", "")
    duration = ctx.get("duration_min", 60)
    mood = ctx.get("mood", "")
    prompt = f"Build a {duration}-minute {genre} set. Mood: {mood}."
    if memory_summary:
        prompt += f"\n\nPast session notes:\n{memory_summary}"
    messages = [{"role": "user", "content": prompt}]
    return await run_agent_streaming(_PLANNER_SYSTEM, _PLANNER_TOOLS, messages, ctx, emit)


async def phase_critique(
    ctx: dict,
    emit: Callable,
    memory_summary: str = "",
) -> tuple[str, list[str], list[dict]]:
    """Run Critic. Returns (verdict, problems, structured_problems)."""
    prompt = "A playlist has been proposed. Review it and deliver your verdict."
    if memory_summary:
        prompt += f"\n\nMemory context:\n{memory_summary}"
    messages = [{"role": "user", "content": prompt}]
    response = await run_agent_streaming(_CRITIC_SYSTEM, _CRITIC_TOOLS, messages, ctx, emit)
    return _parse_critic_response(response, ctx.get("playlist"))


async def phase_editor(
    message: str,
    history: list[dict],
    ctx: dict,
    emit: Callable,
) -> str:
    """Run one Editor turn. Mutates ctx['playlist'] via tool calls."""
    history.append({"role": "user", "content": message})
    response = await run_agent_streaming(_EDITOR_SYSTEM, _WEB_EDITOR_TOOLS, history, ctx, emit)
    history.append({"role": "assistant", "content": response})
    return response


async def phase_validate(session_name: str, ctx: dict, emit: Callable) -> tuple[str, list[str]]:
    """Run Validator. Returns (status, issues)."""
    messages = [{"role": "user", "content": f"Session '{session_name}' was just built. Validate its audio quality."}]
    response = await run_agent_streaming(_VALIDATOR_SYSTEM, _VALIDATOR_TOOLS, messages, ctx, emit)
    return _parse_validator_response(response)


async def load_memory(genre: str, ctx: dict) -> str:
    """Load past session memory for the given genre (runs in thread — does I/O)."""
    return await asyncio.to_thread(read_memory, genre, context_variables=ctx)


# ---------------------------------------------------------------------------
# Mock mode — AGENT_PROVIDER=mock swaps every phase with deterministic fakes
# so tests/E2E runs never touch Anthropic, OpenAI, librosa, or the filesystem.
# ---------------------------------------------------------------------------

if _PROVIDER_ENV == "mock":
    from . import mock_pipeline  # noqa: E402

    mock_pipeline.install(sys.modules[__name__])
