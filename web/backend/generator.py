"""Apollo G0/G1 — ACE-Step generator feature module.

Owns everything the session wizard's "generate a track" step needs from
the backend side. Follows the ``render.py`` precedent: an ``APIRouter``
defined here, included from ``app.py`` with a single line, so the
feature can grow (G2 publisher) without further surgery on ``app.py``.

Plan: ``docs/acestep-wizard-plan.md``. API contract:
``docs/ACE-STEP-API-SPEC.md``. HTTP client: :mod:`acestep_client`.

Endpoints
---------

``GET /api/generator/health`` (G0) is the feature flag the wizard
renders against: ``{available, blocked_by_live, stats}``. "Not
available" is a normal answer — the ACE-Step box is off most of the
time by design (the VRAM protocol below), so the UI must show
"generator unavailable" instead of an error state.

``POST /api/generator/tasks`` (G1) releases one generation. This is the
ONLY endpoint that touches the GPU, so it is the only one behind the
VRAM guard (409). It also owns the catalog-safety defaults: WAV,
``thinking``, and an explicit in-window ``bpm`` (spec §5.5 — poisoned
BPMs have already caused live genre drift, so the value is pinned at
release time rather than left to the LM).

``GET /api/generator/tasks/{task_id}`` (G1) polls. Polling is allowed
during a live set and is written to survive a flaky box: an ACE-Step
blip degrades to ``{status: "pending", degraded: true}`` rather than
5xx-ing the wizard's 3-second poll loop.

``GET /api/generator/audio`` (G1) proxies take audio. The browser never
talks to :8001 — auth and LAN isolation live here.

``POST /api/generator/publish`` (G2b) lands one take in the catalog. It
downloads the take's WAV and hands it to G2a's ``main.ingest_track`` —
the SAME code path ``python main.py --ingest`` runs, imported lazily
inside the handler. It carries no ``task_id`` on purpose: ACE's job
records are mortal, its result files are not, so the page supplies the
take's decoded path and metas from its own state.

``POST /api/generator/edit`` (G3) re-releases one take as a repaint,
a cover or a completion. It is GPU work like ``/tasks``, so it carries
the same 409 guard, and its answer is a plain ``task_id`` served by the
SAME polling endpoint — an edit is just another task whose source
happens to be an earlier take. Like publish it carries no ``task_id``:
the page supplies the source take's decoded path.

``POST /api/generator/critique`` (G4) scores one take. An LLM cannot
hear, so the SCORE comes from machinery that can — ``bench_wav``, the
project's own definition-of-done gate — and the LLM only adds the read.
Both halves degrade: a genre with no committed references answers
``passed: null`` with a note, and any LLM trouble answers
``critique: null``. Scoring never gates publishing.

``GET /api/generator/generations``, ``PATCH
/api/generator/generations/{id}/takes/{idx}`` and ``POST
/api/generator/generations/{id}/refresh`` (G6) are the library. The
endpoints above RECORD as they succeed — release inserts a pending row,
the first done-poll fills in its takes, publish marks one of them — so
the history outlives the page that made it. Every one of those writes is
best-effort: the store failing must never break the endpoint it hangs
off, least of all the 3-second poll.

``GET /api/generator/audio``, ``POST /api/generator/publish``,
``POST /api/generator/edit`` and ``POST /api/generator/critique`` share
one path validator (:func:`validate_ace_audio_path`) so "what counts as
a take's audio" has a single definition.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import posixpath
import shutil
import tempfile
from dataclasses import dataclass
import httpx

from typing import Any, Literal
from urllib.parse import quote, unquote

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import acestep_client, auth, covers, db
from .brief_parser import detect_provider
from .ws_manager import ws_manager


#: ``ws_manager`` channel the primary live WS registers under. Defined in
#: ``app.live_session_ws`` (``ws_manager.connect(..., channel="live")``).
LIVE_WS_CHANNEL = "live"

#: Rendered VERBATIM by the wizard when a release is refused (the plan's
#: G1 frontend contract) — so it speaks the wizard's language, which is
#: English throughout (<html lang="en">, "Sequence the night"). It matches
#: the voice of the frontend's own blocked_by_live tooltip.
VRAM_CONFLICT_MESSAGE = (
    "VRAM protocol: a set is on air. ACE-Step holds ~12.5 GB of the shared "
    "16 GB GPU, so generating now would starve the live DJ's model. Try "
    "again when the session ends."
)

#: ``ACESTEP_BASE_URL`` unset, or the box unreachable. Both mean the same
#: thing to the wizard, and neither is an error worth a stack trace.
UNAVAILABLE_MESSAGE = (
    "The ACE-Step generator is not available "
    f"({acestep_client.ENV_BASE_URL} unset or the box is off)."
)

# ── Release defaults (docs/acestep-wizard-plan.md, G1 contract) ───────

DEFAULT_AUDIO_DURATION_SEC = 180.0
MIN_AUDIO_DURATION_SEC = 120.0   # Apollo catalog floor (spec §5)
MAX_AUDIO_DURATION_SEC = 600.0   # ACE-Step ceiling (spec §3.1)
DEFAULT_BATCH_SIZE = 2
MAX_BATCH_SIZE = 8
DEFAULT_VOCAL_LANGUAGE = "en"
MIN_BPM = 30
MAX_BPM = 300

#: The catalog contract is WAV (spec §5) — never let the UI pick mp3.
CATALOG_AUDIO_FORMAT = "wav"

#: Fields the server owns. The ``experimental`` passthrough may not
#: shadow them (that is a 422, not a silent override) — otherwise an
#: ``audio_format: "mp3"`` or a second ``duration`` sneaks past the
#: catalog validations G2 depends on. Aliases the API accepts for the
#: same slot (``caption``/``duration``, spec §3.1) are included.
SERVER_OWNED_FIELDS = frozenset({
    "audio_duration", "audio_format", "batch_size", "bpm", "caption",
    "duration", "key_scale", "lyrics", "prompt", "thinking",
    "vocal_language",
})

#: ``query_result`` status → the wizard's vocabulary (spec §4).
TASK_STATUS_NAMES = {0: "pending", 1: "done", 2: "failed"}

#: Poll failures that must NOT 5xx: a queue blip, a restarting box or a
#: momentarily malformed batch are all "ask again in 3 s". An auth or
#: bad-request failure is a misconfiguration and stays loud (502).
_POLL_DEGRADE_ERRORS = (
    acestep_client.AceStepUnavailable,
    acestep_client.AceStepServerError,
    acestep_client.AceStepQueueFull,
    acestep_client.AceStepProtocolError,
)

#: Upstream headers worth mirroring on the audio proxy. ``content-type``
#: is handled separately (Starlette owns it via ``media_type``).
_PROXY_HEADERS = (
    "content-length", "content-range", "accept-ranges", "etag",
    "last-modified",
)


def live_session_active(session_id: str | None = None) -> bool:
    """Apollo's side of the ACE-Step VRAM protocol: is a live set on air?

    The 16 GB GPU is shared between ACE-Step (~12.5 GB once loaded, and
    it does not give it back) and the live DJ's LM Studio model. Starting
    a generation during a broadcast starves the DJ — the symptom is a
    400 "Failed to load model" while ``/v1/models`` still lists it.

    **This is the guard G1 enforces.** ``GET /api/generator/health``
    merely surfaces it; G1's generation endpoints must call this helper
    and REFUSE to release a task while it returns ``True``. Import it
    from here (``from .generator import live_session_active``) rather
    than re-deriving live state, so there is one definition to fix.

    It reads the REAL registry: ``ws_manager``'s connection table, whose
    ``live``-channel entry is created by the primary live WS handler
    after its playlist checks and removed in that handler's ``finally``.
    Log grepping (``docker logs | grep live-ws``, as the API spec
    suggests for humans) is explicitly NOT the mechanism here — it is
    unreliable, racy, and invisible to tests.

    Read-only viewers (OBS via ``/live/viewer``) deliberately do not
    count: they never touch ``ws_manager`` and cannot drive an engine, so
    a lingering viewer with no primary consumes no VRAM.

    Args:
        session_id: check one session instead of the whole box. Defaults
            to ``None`` = "is ANY session live?", which is the right
            question for a shared GPU — another session's broadcast
            blocks generation just as hard as this one's.
    """
    if session_id is not None:
        return ws_manager.is_connected(session_id, channel=LIVE_WS_CHANNEL)
    return bool(ws_manager.active_sessions(channel=LIVE_WS_CHANNEL))


def _client() -> acestep_client.AceStepClient:
    """Build the ACE-Step client for a request.

    A function rather than a module-level singleton so the env is read
    per call (``--reload`` + late ``.env``) and so tests can inject an
    ``httpx.MockTransport`` by monkeypatching this seam — the suite must
    never open a socket toward a real ACE-Step box.
    """
    return acestep_client.AceStepClient()


# ── Genre windows: where the bpm default comes from ──────────────────


def _genre_bpm_windows() -> dict[str, tuple[int, int]]:
    """The BPM window table used for the ``bpm`` default.

    Bound to ``agent.tools._BPM_GENRE_RANGES``, **not** to the canonical
    ``main.BPM_GENRE_RANGES``, on purpose: ``agent.tools`` is already a
    backend dependency (``web.backend.pipeline`` imports it at module
    scope), whereas importing ``main`` costs ~2.6 s and ~1800 modules
    (librosa + numba + moviepy + pedalboard) — unacceptable inside a
    request handler, and no backend module has ever imported it.

    The known cost is drift: ``agent/tools.py`` keeps a partial copy
    (CLAUDE.md, "Adding a new genre" step 5), so genres like
    ``cocktail house`` and ``soul jazz`` are missing from it. That is
    why this table is used ONLY to pick a default and never as the
    genre allow-list (see :func:`_resolve_genre`): a genre with no
    window still generates, it just goes out without a server-pinned
    bpm and says so in the log.

    Lazy import so this module stays importable on its own (the tests
    import it directly) — the ``render.py`` precedent.
    """
    from agent.tools import _BPM_GENRE_RANGES  # noqa: PLC0415

    return _BPM_GENRE_RANGES


def _catalog_genres() -> set[str]:
    """Lower-cased ``genre_folder`` set from tracks.json (empty on error).

    Second half of the "genre_folder must exist" check. Never fatal: a
    missing or unreadable catalog degrades to "no catalog opinion", so
    generation still works from the window table alone (a worktree
    without ``tracks/`` is a normal dev state).
    """
    from . import pipeline  # noqa: PLC0415 — heavy import chain

    try:
        _, genres = pipeline.load_catalog(None)
    except Exception as exc:  # noqa: BLE001 — advisory lookup, never fatal
        print(f"[generator] catalog genres unavailable: {exc}", flush=True)
        return set()
    return {g.strip().lower() for g in genres if g}


def _default_bpm_for(genre_key: str) -> int | None:
    """Centre of the genre's BPM window, or ``None`` when it has none.

    Spec §5.5: pass an explicit bpm at the centre of the destination
    genre's window so ``metas.bpm`` comes back in-window and the take is
    ingestable without re-detection.
    """
    window = _genre_bpm_windows().get(genre_key)
    if not window:
        return None
    lo, hi = window
    return int(round((lo + hi) / 2))


def _resolve_genre(genre_folder: str) -> str:
    """Validate ``genre_folder`` and return its lower-cased key.

    Known = in the BPM window table OR in the catalog (case-insensitive,
    matching ``main``'s ``BPM_GENRE_RANGES.get(genre.lower())``). The
    catalog is only consulted when the window table has no entry, so the
    common path costs no I/O.
    """
    key = genre_folder.strip().lower()
    if key in _genre_bpm_windows():
        return key
    if key in _catalog_genres():
        return key
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unknown genre_folder '{genre_folder}'. It must exist in the "
            "catalog (tracks/<genre>/) — a new genre is a coordinated "
            "checklist, not something the wizard improvises."
        ),
    )


# ── ETA ──────────────────────────────────────────────────────────────


def _eta_seconds(stats: dict | None, queue_position: int | None) -> int | None:
    """``avg_job_seconds × (queue_position + running)``, ``None`` without stats.

    The contract's formula, with one floor: a job that is queued behind
    nothing still takes about one job's worth of time, so the product is
    clamped to at least one ``avg_job_seconds``. ``queue_position`` is
    ``None`` on the poll path (``query_result`` does not re-report a
    position), which reads as "unknown, assume the head of the queue".
    """
    if not isinstance(stats, dict):
        return None
    avg = stats.get("avg_job_seconds")
    if isinstance(avg, bool) or not isinstance(avg, (int, float)) or avg <= 0:
        return None

    running = stats.get("running")
    if isinstance(running, bool) or not isinstance(running, int) or running < 0:
        running = 0
    ahead = (queue_position or 0) + running
    return int(round(avg * max(ahead, 1)))


async def _stats_or_none(client: acestep_client.AceStepClient) -> dict | None:
    """``/v1/stats`` for the ETA. Any failure means "no ETA", never a 5xx."""
    try:
        stats = await client.stats()
    except acestep_client.AceStepError as exc:
        print(f"[generator] /v1/stats unavailable, no ETA: {exc}", flush=True)
        return None
    return stats if isinstance(stats, dict) else None


# ── Request body ─────────────────────────────────────────────────────


class GenerationRequest(BaseModel):
    """The wizard's "Suno mode" surface (API spec §3.1).

    ``extra="forbid"``: the body IS this shape, and a silently ignored
    typo would show up as a mysteriously ordinary take hours later.
    Anything genuinely advanced goes through ``experimental``, which is
    forwarded verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=4000)
    lyrics: str = Field("", max_length=20000)
    audio_duration: float = Field(
        DEFAULT_AUDIO_DURATION_SEC,
        ge=MIN_AUDIO_DURATION_SEC,
        le=MAX_AUDIO_DURATION_SEC,
    )
    vocal_language: str = Field(DEFAULT_VOCAL_LANGUAGE, min_length=1, max_length=16)
    genre_folder: str = Field(..., min_length=1)
    bpm: int | None = Field(None, ge=MIN_BPM, le=MAX_BPM)
    key_scale: str | None = Field(None, max_length=64)
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    experimental: dict[str, Any] | None = None

    @field_validator("prompt", "genre_folder")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _experimental_stays_out_of_server_fields(self) -> GenerationRequest:
        if not self.experimental:
            return self
        clashes = sorted(
            k for k in self.experimental if k.strip().lower() in SERVER_OWNED_FIELDS
        )
        if clashes:
            raise ValueError(
                "experimental may not override server-owned fields: "
                + ", ".join(clashes)
            )
        return self


#: ACE's prompt ceiling. The composed prompt is trimmed to it rather
#: than rejected: a long user prompt must never turn into a 422.
_ACE_PROMPT_MAX = 4000


def _compose_prompt(user_prompt: str, genre_key: str) -> str:
    """Frame the user's prompt with the genre's style descriptor.

    ``genre_folder`` used to reach ACE only as a BPM default and a
    destination folder — the model was never told what the genre sounds
    like, so takes came back off-genre and could not be promoted into
    the catalog. The descriptor goes FIRST (it frames) and the user's
    words follow (they specialise).

    Degrades to the bare prompt for a genre with no descriptor, so an
    unlisted folder keeps generating exactly as it did.
    """
    from agent.tools import genre_style_prompt  # noqa: PLC0415

    user = (user_prompt or "").strip()
    style = genre_style_prompt(genre_key)
    if not style:
        return user
    if not user:
        return style
    return f"{style}. {user}"[:_ACE_PROMPT_MAX]


def _release_payload(
    req: GenerationRequest, bpm: int | None, genre_key: str = ""
) -> dict[str, Any]:
    """Body for ``POST /release_task`` — the server's defaults applied.

    ``audio_format`` and ``thinking`` are pinned here (catalog contract +
    spec's quality advice); ``bpm`` is included only when known, so a
    genre with no window falls back to the LM instead of being refused.
    """
    payload: dict[str, Any] = {
        "prompt": _compose_prompt(req.prompt, genre_key),
        "lyrics": req.lyrics,
        "audio_duration": req.audio_duration,
        "vocal_language": req.vocal_language.strip(),
        "batch_size": req.batch_size,
        "audio_format": CATALOG_AUDIO_FORMAT,
        "thinking": True,
    }
    if bpm is not None:
        payload["bpm"] = bpm
    if req.key_scale and req.key_scale.strip():
        payload["key_scale"] = req.key_scale.strip()
    if req.experimental:
        payload.update(req.experimental)
    return payload


# ── Publish body (G2b) ───────────────────────────────────────────────


class PublishMetas(BaseModel):
    """ACE's ``metas`` for the take being published.

    ``extra="ignore"`` — deliberately looser than the body around it.
    This block's shape belongs to ACE (it also carries ``genres`` and
    ``timesignature``), and the ingest only ever reads these three; an
    ACE upgrade that adds a field must not start 422-ing publishes.
    """

    model_config = ConfigDict(extra="ignore")

    bpm: float = Field(..., ge=MIN_BPM, le=MAX_BPM)
    keyscale: str = Field(..., min_length=1, max_length=64)
    #: Advisory. The authoritative duration is probed off the downloaded
    #: file by the ingest; this only buys an early refusal (below).
    duration: float | None = Field(None, gt=0)


class PublishRequest(BaseModel):
    """Publish one take into the catalog (plan: G2b).

    No ``task_id``: the backend never re-queries an old ACE task. Their
    job records are in-memory, expire after 24 h and die with the
    process (which the VRAM protocol stops between batches), while the
    result FILES are never reaped. So the page persists each take's
    decoded path + metas the moment a poll returns them, and hands that
    back here — the plan's persistence rule.

    ``extra="forbid"`` for the same reason ``GenerationRequest`` uses
    it: a silently ignored field would surface hours later as a track
    with the wrong name in the catalog.
    """

    model_config = ConfigDict(extra="forbid")

    #: The take's URL-DECODED ACE path, as persisted by the page.
    file: str = Field(..., min_length=1, max_length=4096)
    metas: PublishMetas
    #: Carried for the log/provenance only — tracks.json has no prompt
    #: slot, and inventing one here would fork the catalog schema.
    prompt: str | None = Field(None, max_length=4000)
    #: TEXT, not a filename — the same semantics as the ingest sidecar's
    #: ``lyrics`` key. Lands as a ``.lrc`` beside the WAV.
    lyrics: str | None = Field(None, max_length=20000)
    display_name: str = Field(..., min_length=1, max_length=120)
    genre_folder: str = Field(..., min_length=1, max_length=120)
    #: Base take's catalog id or display_name (the ingest resolves an id
    #: to its display_name, which is what links the two as one piece).
    variant_of: str | None = Field(None, max_length=120)

    @field_validator("display_name", "genre_folder")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


#: Appended to every successful publish. The ingest is deliberately
#: madmom-free, so the entry lands without duration, beatgrid, waveform
#: peaks or an MP3 sibling — everything the live engine's beatmatching
#: wants. Saying so in the payload is the only thing standing between a
#: fresh take and a contrabombo on air.
FIX_INCOMPLETE_NOTE = (
    "Ingested without madmom: duration, beatgrid, waveform peaks and the MP3 "
    "sibling are still missing. Run `python main.py --fix-incomplete` (in the "
    "main checkout, in Docker) before this track goes into a set."
)

#: Streaming chunk for the take download. A 3-minute 48 kHz WAV is
#: ~35 MB — never buffered whole, same rule as the proxy.
PUBLISH_CHUNK_BYTES = 1 << 20


# ── Edit body (G3) ───────────────────────────────────────────────────

#: The three edits the wizard offers, and the ``task_type`` each maps to
#: (spec §3.3). ``text2music`` is deliberately absent — this endpoint
#: always has a source take; a fresh generation is ``POST /tasks``.
#: ``lego``/``extract`` are out of scope: neither produces a track the
#: catalog could take.
EditMode = Literal["repaint", "cover", "complete"]

#: Spec §3.3: "bajo ≈ 0.2 para style transfer". Pinned server-side when
#: the caller says nothing, for the same reason ``bpm`` is: an unset
#: knob on a generation is a knob the model gets to invent.
DEFAULT_COVER_STRENGTH = 0.2

#: ``repainting_end`` sentinel — "to the end of the source" (spec §3.3).
REPAINT_TO_THE_END = -1.0

#: Without this the range is a hint; with it, the mask is exact (§3.3).
REPAINT_CHUNK_MASK_MODE = "explicit"

#: ``experimental`` may not shadow anything the edit decides. Everything
#: ``POST /tasks`` owns, plus the edit's own vocabulary — otherwise an
#: ``experimental.task_type`` would quietly turn a repaint into an
#: extract, and the source path could be redirected off the box.
EDIT_SERVER_OWNED_FIELDS = SERVER_OWNED_FIELDS | frozenset({
    "audio_cover_strength", "chunk_mask_mode", "ctx_audio",
    "repainting_end", "repainting_start", "src_audio", "src_audio_path",
    "task_type",
})

#: ACE's own words when it refuses to read a file off its own disk: its
#: validator compares ``src_audio_path`` against the process's
#: ``gettempdir()``, which a foreign ``TMPDIR`` in the launch env moves
#: elsewhere — so the box can 400 the very paths it just handed out (the
#: plan's G3 fallback caveat). Matched as a marker SET over whitespace-
#: normalised lower-case text, because the exact sentence is theirs to
#: change and a fatal error here would cost the operator the edit.
ABSOLUTE_PATH_REFUSAL_MARKERS = ("absolute", "audio file path", "not allowed")

#: Every take Apollo releases asks for WAV (the catalog contract), so a
#: source take being uploaded back is one too.
EDIT_UPLOAD_CONTENT_TYPE = "audio/wav"

#: Spec §3.3's multipart field name for the source audio.
EDIT_UPLOAD_FIELD = "src_audio"


class EditRequest(BaseModel):
    """Re-release one take as a repaint / cover / completion (plan: G3).

    No ``task_id``, for the reason :class:`PublishRequest` documents:
    ACE's job records are mortal and its result files are not, so the
    page carries the source take's decoded path here.

    ``extra="forbid"`` **and** wrong-mode parameters are a 422 rather
    than a silent drop. A ``repainting_start`` sent with ``mode:
    "cover"`` means the caller believes something about the request that
    is not true; ignoring it would hand back a take that does not match
    what was asked for, and the only evidence would be the audio.
    """

    model_config = ConfigDict(extra="forbid")

    #: The SOURCE take's URL-DECODED ACE path, as persisted by the page.
    #: It travels to ACE as ``src_audio_path``, so it is root-checked.
    file: str = Field(..., min_length=1, max_length=4096)
    mode: EditMode
    #: Style/direction for the edit. Absent = the page reuses the take's
    #: own prompt, which it holds; the backend never re-queries it.
    prompt: str | None = Field(None, max_length=4000)
    #: SECONDS into the source (spec §3.3), not bars and not a fraction.
    repainting_start: float | None = Field(None, ge=0)
    repainting_end: float | None = Field(None, ge=REPAINT_TO_THE_END)
    audio_cover_strength: float | None = Field(None, ge=0.0, le=1.0)
    #: Only drives the bpm default, exactly as on ``POST /tasks``.
    genre_folder: str | None = Field(None, min_length=1, max_length=120)
    experimental: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _mode_owns_its_parameters(self) -> EditRequest:
        start, end = self.repainting_start, self.repainting_end
        strength = self.audio_cover_strength

        if self.mode == "repaint":
            if strength is not None:
                raise ValueError(
                    "audio_cover_strength belongs to mode 'cover', not 'repaint'"
                )
            if start is None or end is None:
                raise ValueError(
                    "repaint needs repainting_start and repainting_end (seconds; "
                    "repainting_end -1 means 'to the end of the take')"
                )
            if end != REPAINT_TO_THE_END and end <= 0:
                raise ValueError(
                    "repainting_end must be a positive offset in seconds, or "
                    "-1 for 'to the end of the take'"
                )
            if end != REPAINT_TO_THE_END and start >= end:
                raise ValueError(
                    f"repainting_start ({start}) must be before repainting_end "
                    f"({end}) — the range is the slice ACE regenerates"
                )
        elif self.mode == "cover":
            if start is not None or end is not None:
                raise ValueError(
                    "repainting_start/repainting_end belong to mode 'repaint', "
                    "not 'cover' — a cover re-sings the whole take"
                )
        else:  # complete
            if start is not None or end is not None or strength is not None:
                raise ValueError(
                    "mode 'complete' takes no range and no strength — it "
                    "continues the take from where it ends"
                )

        if self.experimental:
            clashes = sorted(
                k for k in self.experimental
                if k.strip().lower() in EDIT_SERVER_OWNED_FIELDS
            )
            if clashes:
                raise ValueError(
                    "experimental may not override server-owned fields: "
                    + ", ".join(clashes)
                )
        return self


def _edit_payload(
    req: EditRequest, src_audio_path: str, bpm: int | None, genre_key: str = ""
) -> dict[str, Any]:
    """Body for ``POST /release_task`` in edit form (spec §3.3).

    ``audio_format``/``thinking`` are pinned exactly as on a fresh
    generation — ``thinking`` is auto-ignored by ACE in repaint and
    cover, and sending it anyway keeps one release shape instead of two.
    """
    payload: dict[str, Any] = {
        "task_type": req.mode,
        "src_audio_path": src_audio_path,
        "audio_format": CATALOG_AUDIO_FORMAT,
        "thinking": True,
    }
    composed = _compose_prompt(req.prompt or "", genre_key)
    if composed:
        payload["prompt"] = composed
    if bpm is not None:
        payload["bpm"] = bpm

    if req.mode == "repaint":
        payload["repainting_start"] = float(req.repainting_start or 0.0)
        payload["repainting_end"] = float(req.repainting_end)  # type: ignore[arg-type]
        payload["chunk_mask_mode"] = REPAINT_CHUNK_MASK_MODE
    elif req.mode == "cover":
        payload["audio_cover_strength"] = (
            DEFAULT_COVER_STRENGTH
            if req.audio_cover_strength is None
            else float(req.audio_cover_strength)
        )

    if req.experimental:
        payload.update(req.experimental)
    return payload


def _is_absolute_path_refusal(exc: acestep_client.AceStepError) -> bool:
    """Is this 400 the TMPDIR caveat, i.e. the one worth degrading on?"""
    haystack = " ".join(
        str(part) for part in (exc.message, exc.payload) if part
    ).lower()
    haystack = " ".join(haystack.split())
    return all(marker in haystack for marker in ABSOLUTE_PATH_REFUSAL_MARKERS)


# ── Audio proxy helpers ──────────────────────────────────────────────


def _authorize_audio(request: Request, token: str | None) -> dict:
    """Bearer header OR ``?token=`` — an ``<audio>`` tag cannot set headers.

    Same query-string escape hatch as ``stream_track`` /
    ``download_asset``; the header path keeps ``fetch()`` callers working
    with the wizard's normal auth.
    """
    header = request.headers.get("authorization") or ""
    raw = header[7:].strip() if header[:7].lower() == "bearer " else (token or "").strip()
    if not raw:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth.user_from_query_token(raw)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


# ── ACE-Step audio paths: ONE validator, two call sites ──────────────
#
# The audio proxy (G1) and the publisher (G2b) both receive "a take's
# audio location" from the browser, and both have to decide whether it
# is safe to act on. They used to be able to drift apart; they now share
# :func:`validate_ace_audio_path` and the shape table below, so flipping
# the decision (a new ACE deployment, a different tmp root) is a
# constant change rather than a refactor.
#
# Confirmed with the ACE session (2026-08-29, read from their server):
# a result's ``file`` is ``/v1/audio?path=<quote(p, safe="")>`` where
# ``p`` is an ABSOLUTE POSIX path on the ACE box, under its
# ``tmp_root``/``api_audio`` directory, named ``<uuid>_<take>.wav``.
# The encoding is total — slashes travel as ``%2F`` — so never look for
# literal ``/`` inside the query parameter.

#: Where ACE-Step's result files live on its own disk. Override per
#: deployment (comma-separated for more than one). Read at CALL time,
#: never cached at import: the backend runs under ``--reload`` with a
#: late-loaded ``.env`` (the ``brief_parser`` lesson, web/CLAUDE.md).
ENV_AUDIO_ROOT = "ACESTEP_AUDIO_ROOT"

#: The current box's root, as confirmed by the ACE session.
DEFAULT_AUDIO_ROOTS = ("/home/pablo/code/ACE-Step-1.5/.cache/acestep/tmp/api_audio",)

#: ACE's audio endpoint. A take's ``file`` is this plus ``?path=``.
ACE_AUDIO_ENDPOINT = "/v1/audio"


def ace_audio_roots() -> tuple[str, ...]:
    """Accepted filesystem prefixes for a take's audio, newest env wins."""
    raw = (os.getenv(ENV_AUDIO_ROOT) or "").strip()
    if not raw:
        roots = DEFAULT_AUDIO_ROOTS
    else:
        roots = tuple(part.strip() for part in raw.split(",") if part.strip())
    # posixpath: the ACE box is Linux even when Apollo runs on Windows,
    # so ntpath's backslash rules must not get a vote here.
    return tuple(posixpath.normpath(r) for r in roots if r) or DEFAULT_AUDIO_ROOTS


@dataclass(frozen=True)
class AceAudioPath:
    """A validated take-audio location.

    ``api_path`` is what to hand :meth:`AceStepClient.stream_audio`;
    ``file_path`` is the DECODED path on the ACE box, known only for the
    shapes that name a filesystem location.
    """

    shape: str                 # "api" | "absolute" | "relative"
    api_path: str
    file_path: str | None = None


class AceAudioPathError(ValueError):
    """A take-audio path Apollo refuses to act on.

    Raised shape-neutral so each call site can pick its own status: the
    proxy answers 400 (a bad query parameter), publish answers 422 (a
    bad body field).
    """


def _is_under_a_root(path: str) -> bool:
    norm = posixpath.normpath(path)
    return any(
        norm == root or norm.startswith(root + "/") for root in ace_audio_roots()
    )


def _outside_the_root() -> AceAudioPathError:
    """The one sentence for "that absolute path is not a take of ours".

    Shared by the two shapes that name an absolute location (``api``'s
    decoded inner path and a bare ``absolute``) so the wording — and the
    ``ACESTEP_AUDIO_ROOT`` hint the operator actually needs — cannot
    drift between them.
    """
    return AceAudioPathError(
        f"path must be an ACE-Step result file under "
        f"{' or '.join(ace_audio_roots())} (set {ENV_AUDIO_ROOT} if the "
        f"generator writes elsewhere)"
    )


def _inner_path_of(api_path: str) -> str | None:
    """The ``?path=`` parameter of an ACE API path, decoded exactly once.

    Hand-rolled rather than ``parse_qs``: that helper also turns ``+``
    into a space, and ACE encodes with ``quote(p, safe="")``, which
    leaves a literal ``+`` in a filename as ``+``. One wrong character
    and the download 404s.
    """
    _, _, query = api_path.partition("?")
    for field in query.split("&"):
        name, sep, value = field.partition("=")
        if sep and name == "path":
            return unquote(value)
    return None


def validate_ace_audio_path(value: str, *, resolve_file: bool = False) -> AceAudioPath:
    """Validate one take-audio location. **The single source of truth.**

    Three shapes are accepted, and every ABSOLUTE path any of them names
    is held to the SAME root check on EVERY route, proxy included:

    ``api`` — ``/v1/audio?path=<encoded>``, a take's ``file`` field
        verbatim. It names an ACE *endpoint* and ACE's own validator is
        still the far-side authority on what it will serve — but the
        decoded inner path is screened against :func:`ace_audio_roots`
        here too. Defence in depth: Apollo does not FORWARD a location
        it would refuse to PUBLISH, and one rule for one value is one
        fewer thing for the two call sites to disagree about. (The proxy
        used to leave this opaque; that asymmetry was the bug.)
    ``absolute`` — a bare ``/home/.../api_audio/<uuid>_0.wav``, the
        decoded path the wizard persists per the plan's persistence rule.
        This names a FILESYSTEM LOCATION, so it must sit under
        :func:`ace_audio_roots`. It still streams: the proxy re-encodes
        it into the endpoint form exactly as publish resolves it.
    ``relative`` — ``api_audio/x.wav``. The root check CANNOT apply to
        this shape — ACE resolves it against a root only ACE knows, so
        there is no absolute path here to prefix-match, and inventing
        one would mean guessing at the far side's layout. It is
        forwarded (ACE resolves and screens it) but is never enough for
        a publish, which needs a concrete file to download.

    ``resolve_file=True`` is the publisher's mode: it needs a concrete
    file to download and write into the catalog, so it unwraps the
    ``api`` shape and refuses ``relative``. What separates the two modes
    is now only what each DOES with the value — not how hard it looked
    at it.

    Raises :class:`AceAudioPathError` naming what was wrong.
    """
    raw = (value or "").strip()
    if not raw:
        raise AceAudioPathError("path is required")

    lowered = raw.lower()
    if "://" in lowered or raw.startswith("//") or lowered.startswith(("http:", "https:")):
        # A host would turn the proxy into an open redirector / SSRF hop.
        raise AceAudioPathError("path must be relative to the ACE-Step base URL")
    if "\\" in raw or (len(raw) > 1 and raw[1] == ":"):
        raise AceAudioPathError("path must not be a local file path")
    if ".." in raw.split("?")[0].split("/"):
        raise AceAudioPathError("path must not traverse directories")

    if raw.startswith("/v1/"):
        inner = _inner_path_of(raw)
        if inner is not None and ".." in inner.split("/"):
            raise AceAudioPathError("path must not traverse directories")
        # An ABSOLUTE inner path names a file on the ACE box, so it gets
        # publish's root check even when we are only proxying. A missing
        # or relative one is ACE's to resolve (see the docstring).
        if inner and inner.startswith("/") and not _is_under_a_root(inner):
            raise _outside_the_root()
        resolved = AceAudioPath(
            shape="api",
            api_path=raw,
            file_path=inner if inner and inner.startswith("/") else None,
        )
    elif raw.startswith("/"):
        # Already decoded by contract — do NOT unquote again, a literal
        # '%' in a filename would be silently mangled.
        if not _is_under_a_root(raw):
            raise _outside_the_root()
        resolved = AceAudioPath(
            shape="absolute",
            api_path=f"{ACE_AUDIO_ENDPOINT}?path={quote(raw, safe='')}",
            file_path=raw,
        )
    else:
        resolved = AceAudioPath(shape="relative", api_path=raw)

    if not resolve_file:
        return resolved

    if resolved.file_path is None:
        raise AceAudioPathError(
            "file must be the take's decoded ACE-Step path (an absolute path "
            f"under {' or '.join(ace_audio_roots())}), as the page persisted it "
            "when the take arrived — ACE's job records expire, its result files "
            "do not"
        )
    # Backstop, not the gate: every shape that sets ``file_path`` was
    # root-checked above. Kept because this is the last line before a
    # download, and a fourth shape added later must not slip past — the
    # same defence-in-depth argument that closed the proxy's asymmetry.
    if not _is_under_a_root(resolved.file_path):
        raise AceAudioPathError(
            f"file '{resolved.file_path}' is not under "
            f"{' or '.join(ace_audio_roots())} (set {ENV_AUDIO_ROOT} if the "
            f"generator writes elsewhere)"
        )
    return AceAudioPath(
        shape=resolved.shape,
        api_path=f"{ACE_AUDIO_ENDPOINT}?path={quote(resolved.file_path, safe='')}",
        file_path=resolved.file_path,
    )


def _validate_proxy_path(path: str) -> str:
    """The proxy's half of :func:`validate_ace_audio_path` — 400 on refusal."""
    try:
        return validate_ace_audio_path(path).api_path
    except AceAudioPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── G6 recording hooks: the generations library ──────────────────────
#
# The wizard's page state was the only record that a generation ever
# happened — close the tab and the history was gone, while ACE's result
# files stayed on its disk forever. These hooks hang off the endpoints
# that ALREADY know (release, poll, publish, edit), so the page stays
# dumb and the library survives it. Nothing here changes a single
# request or response contract; the store is written on the way past.
#
# **A store failure is never the endpoint's problem.** Every write goes
# through :func:`_store`, which logs and returns ``None``. The poll is
# the one that matters most: the wizard polls every 3 s and the module's
# oldest rule is that a poll must not 5xx — a SQLite hiccup is even less
# of a reason to break it than an ACE-Step blip is.


async def _store(what: str, fn, *args, **kwargs):
    """Run one library write off the event loop. **Never raises.**"""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — history is never load-bearing
        print(f"[generator] library store — {what} failed: {exc}", flush=True)
        return None


def _decoded_take_path(file: str | None) -> str | None:
    """The take's decoded ACE path, or ``None`` if it does not name one.

    The SAME validator publish resolves with, so the value stored here is
    byte-identical to the one a later publish arrives carrying — that
    equality is the whole lookup key of :func:`db.mark_take_published`.
    A path the validator refuses (a relative shape, or a root that moved)
    is stored as ``None``: the take is still recorded, it simply cannot
    be matched by path later, which is what refusing it means anyway.
    """
    try:
        return validate_ace_audio_path(file or "").file_path
    except AceAudioPathError:
        return None


async def _record_release(
    user_id: int, task_id: str, payload: dict[str, Any], genre_folder: str | None
) -> None:
    """Release hook: one ``pending`` generation, keyed by ACE's task id.

    ``request_json`` is the ACTUAL outgoing payload — every default the
    server pinned included — plus the ``genre_folder`` that ACE never
    sees. An edit's payload already carries ``task_type`` and
    ``src_audio_path``, so lineage is queryable without a second shape,
    and it stays the JSON payload even when the request degraded to a
    multipart upload: what was ASKED for is the interesting record.
    """
    await _store(
        f"record generation {task_id}",
        db.record_generation,
        task_id,
        user_id,
        {**payload, "genre_folder": genre_folder},
    )


async def _record_takes(user_id: int, task_id: str, takes: list[dict]) -> None:
    """Done-poll hook: persist the takes and mark the generation ``done``.

    Idempotent by construction (see :func:`db.save_generation_takes`), so
    a wizard that polls one more time after the answer arrives writes the
    same rows again and changes nothing the user has done to them.

    ``takes`` is the poll's own response list; the decoded path is added
    to a COPY so the wire shape stays exactly the poll contract.
    """
    records = [
        {**take, "decoded_path": _decoded_take_path(take.get("file"))}
        for take in takes
    ]
    saved = await _store(
        f"save takes of {task_id}", db.save_generation_takes, task_id, user_id, records
    )
    if saved is False:
        # Not an error: a task released before G6, or one this user does
        # not own. Either way the poll itself is unaffected.
        print(
            f"[generator] library store — {task_id} is not this user's "
            "generation; takes not recorded",
            flush=True,
        )


async def _record_status(user_id: int, task_id: str, status: str) -> None:
    """Failed / stale hook: move one generation to a terminal status."""
    await _store(
        f"mark generation {task_id} {status}",
        db.set_generation_status,
        task_id,
        user_id,
        status,
    )


async def _record_publish(user_id: int, decoded_path: str, track_id: str) -> None:
    """Publish hook: mark the take at ``decoded_path`` ``published``.

    Zero contract change — publish still carries no ``task_id``, so the
    take is found by the one value both sides hold. An unmatched path is
    LOGGED, never an error: publishing a take generated before G6 (or
    from another machine's session) is a perfectly good publish.
    """
    matched = await _store(
        f"mark {decoded_path} published",
        db.mark_take_published,
        user_id,
        decoded_path,
        track_id,
    )
    if not matched:
        print(
            f"[generator] library store — no recorded take at '{decoded_path}' "
            f"for this user; published '{track_id}' without a library link",
            flush=True,
        )


router = APIRouter()


#: Bound on the LLM-server probe. It is a local box on the tailnet; anything
#: slower than this is "not answering" for a panel that renders on mount.
ENGINES_TIMEOUT_SEC = 3.0


def _llm_models_url() -> str | None:
    """Where to ask the local LLM server which models it HOLDS.

    Derived from `OLLAMA_BASE_URL` (the one place the app's OpenAI-compatible
    endpoint is configured) rather than a second variable: two addresses for
    one server is how they end up disagreeing. LM Studio serves the state under
    `/api/v0/models`; the OpenAI-compatible `/v1/models` lists what EXISTS,
    which is not the question — a listed model is not a loaded one, and that
    distinction is written into the root CLAUDE.md in blood.

    Read at CALL time (the `brief_parser` lesson).
    """
    base = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not base:
        return None
    return base.rstrip("/").removesuffix("/v1") + "/api/v0/models"


async def _llm_engine_state() -> dict[str, Any]:
    """Which local models are loaded. Never raises: unreachable is an answer."""
    url = _llm_models_url()
    if not url:
        return {"configured": False, "reachable": False, "loaded": [], "known": 0}
    try:
        async with httpx.AsyncClient(timeout=ENGINES_TIMEOUT_SEC) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 — every failure means "no answer"
        print(f"[generator] LLM state probe failed: {exc}", flush=True)
        return {"configured": True, "reachable": False, "loaded": [], "known": 0}

    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {"configured": True, "reachable": True, "loaded": [], "known": 0}
    loaded = [
        m.get("id")
        for m in entries
        if isinstance(m, dict) and m.get("state") == "loaded" and m.get("id")
    ]
    return {
        "configured": True,
        "reachable": True,
        "loaded": loaded,
        "known": len(entries),
    }


@router.get("/api/generator/engines")
async def generator_engines(
    current_user: dict = Depends(auth.get_current_user),
):
    """What the two GPU tenants are HOLDING right now.

    The panel this feeds exists because the shared 16 GB is managed by a
    protocol nobody can see: ACE wants ~12.5 GB and the live DJ's model ~6, so
    they cannot both be resident, and the failure when they collide is silent —
    LM Studio answers 400 for every model while still LISTING them, and the
    first brief after an unload simply takes a long time for no visible reason.
    Both cost hours on 2026-09-02/04.

    Reports what each side SAYS about itself, never a guess:

    * `ace.loaded` — `models_initialized` from ACE's own /health. Started with
      `--no-init` it is reachable and holding nothing, which is the intended
      resting state and is invisible from `available` alone.
    * `llm.loaded` — the ids LM Studio reports as `state=loaded`.

    Deliberately NOT reported: free VRAM. Neither box exposes it and
    `nvidia-smi` is not in this container, so any number here would be an
    inference dressed as a measurement. It needs a host-side helper.

    Never 5xx: a status panel that can take the page down is worse than no
    panel. Every unreachable path answers with flags, not an error.
    """
    client = _client()
    ace_state = await client.engine_state()

    ace: dict[str, Any] = {
        "configured": client.enabled(),
        "reachable": ace_state is not None,
        "loaded": False,
        "llm_loaded": False,
        "model": None,
        "lm_model": None,
    }
    if ace_state:
        body = ace_state.get("data") if isinstance(ace_state.get("data"), dict) else ace_state
        ace["loaded"] = bool(body.get("models_initialized"))
        ace["llm_loaded"] = bool(body.get("llm_initialized"))
        ace["model"] = body.get("loaded_model")
        ace["lm_model"] = body.get("loaded_lm_model")

    return {
        "ace": ace,
        "llm": await _llm_engine_state(),
        "blocked_by_live": live_session_active(),
    }


@router.get("/api/generator/health")
async def generator_health(
    current_user: dict = Depends(auth.get_current_user),
):
    """Feature flag + queue snapshot for the wizard's generation step.

    ``available``: ``ACESTEP_BASE_URL`` is set AND ``/health`` answered.
    ``blocked_by_live``: a live set is on air (see
    :func:`live_session_active`) — generation must not start.
    ``stats``: ``/v1/stats`` (queue depth, ``avg_job_seconds`` for the
    UI's ETA) when up, else ``None``.

    Never 5xx on a dead generator: unreachable is a normal state, so it
    reports ``available: false``. A ``/health`` that answers while
    ``/v1/stats`` fails still reports available with ``stats: null``
    rather than lying about the box being down.
    """
    client = _client()
    available = await client.health()

    stats = None
    if available:
        try:
            stats = await client.stats()
        except acestep_client.AceStepError as exc:
            # Up but not answering /v1/stats — degrade to no ETA data.
            print(f"[generator] /v1/stats failed: {exc}", flush=True)
            stats = None

    return {
        "available": available,
        "blocked_by_live": live_session_active(),
        "stats": stats,
    }


@router.post("/api/generator/tasks")
async def create_generation_task(
    req: GenerationRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """Release one generation to ACE-Step. **The only GPU-touching call.**

    Refusal ladder (checked in this order, cheapest and most fundamental
    first — note FastAPI runs body validation before the handler, so a
    malformed body is a 422 even on a disabled generator):

    * **503** — generator disabled (``ACESTEP_BASE_URL`` unset) or the
      box unreachable. Both are normal states, not outages.
    * **409** — a live set is on air (:func:`live_session_active`, the
      G0 guard). Detail is :data:`VRAM_CONFLICT_MESSAGE`, which the
      wizard renders verbatim.
    * **422** — unknown ``genre_folder``, or a field outside its range.
    * **429** — ACE-Step's queue is full (``ACESTEP_QUEUE_MAXSIZE``).
      Backpressure the wizard can honour, not a failure.
    * **502** — ACE-Step answered but rejected/broke (bad key, protocol).

    Returns ``{task_id, queue_position, eta_seconds}``.
    """
    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    if live_session_active():
        raise HTTPException(status_code=409, detail=VRAM_CONFLICT_MESSAGE)

    genre_key = await asyncio.to_thread(_resolve_genre, req.genre_folder)
    bpm = req.bpm if req.bpm is not None else _default_bpm_for(genre_key)
    if bpm is None:
        # Visible, not silent: this genre has no window in the table this
        # module binds to, so the LM picks the tempo and G2's in-window
        # validation may reject the take.
        print(
            f"[generator] no BPM window for genre '{genre_key}' — "
            "releasing without a server-pinned bpm",
            flush=True,
        )

    payload = _release_payload(req, bpm, genre_key)
    try:
        released = await client.release_task(payload)
    except (acestep_client.AceStepDisabled, acestep_client.AceStepUnavailable) as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except acestep_client.AceStepQueueFull as exc:
        raise HTTPException(
            status_code=429,
            detail="ACE-Step's queue is full — retry in a moment.",
        ) from exc
    except acestep_client.AceStepError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # G6: the library's first row. After this point the record outlives
    # the page — the poll hook fills in the takes.
    await _record_release(
        current_user["id"], released.task_id, payload, req.genre_folder
    )

    stats = await _stats_or_none(client)
    return {
        "task_id": released.task_id,
        "queue_position": released.queue_position,
        "eta_seconds": _eta_seconds(stats, released.queue_position),
    }


@router.get("/api/generator/tasks/{task_id}")
async def get_generation_task(
    task_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    """Poll one task. Allowed during a live set — polling costs no VRAM.

    Maps ``query_result``'s ``status`` 0/1/2 onto
    ``pending``/``done``/``failed`` and flattens the batch into
    ``takes``. Written to survive a flaky box: an ACE-Step blip (or a
    task id it does not know yet) answers ``{status: "pending",
    degraded: true}`` with HTTP 200, because a wizard polling every 3 s
    must not fall over on one bad round trip. ``result_parse_error`` is
    carried through rather than raised, for the same reason.

    The keys are always present (``degraded`` and ``result_parse_error``
    included) so the poll loop never has to feature-detect.
    """
    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)

    try:
        results = await client.query_result([task_id])
    except acestep_client.AceStepDisabled as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except _POLL_DEGRADE_ERRORS as exc:
        print(f"[generator] poll of {task_id} degraded: {exc}", flush=True)
        return _degraded_poll(task_id)
    except acestep_client.AceStepError as exc:
        # Auth / bad request: a misconfiguration that retrying cannot fix.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    entry = next((r for r in results if r.task_id == task_id), None)
    if entry is None and len(results) == 1 and not results[0].task_id:
        entry = results[0]  # answered positionally, without echoing the id
    if entry is None:
        print(f"[generator] poll of {task_id}: no entry in the batch", flush=True)
        return _degraded_poll(task_id)

    status = TASK_STATUS_NAMES.get(entry.status, "pending")
    eta = None
    if status == "pending":
        eta = _eta_seconds(await _stats_or_none(client), None)

    takes = [
        {
            "index": index,
            "file": take.file,
            "prompt": take.prompt,
            "lyrics": take.lyrics,
            "metas": take.metas,
            "seed_value": take.seed_value,
        }
        for index, take in enumerate(entry.takes)
    ]

    # G6: the first done-poll is what turns a pending row into history.
    # Re-polls hit the same rows and change nothing (the store's upsert
    # leaves published/discarded alone), and an id the store never saw is
    # a logged no-op — polling has never required a library row.
    if status == "done":
        await _record_takes(current_user["id"], task_id, takes)
    elif status == "failed":
        await _record_status(current_user["id"], task_id, "failed")

    return {
        "task_id": task_id,
        "status": status,
        "takes": takes,
        "eta_seconds": eta,
        "degraded": False,
        "result_parse_error": entry.result_parse_error,
    }


def _degraded_poll(task_id: str) -> dict:
    """The "ask again in 3 s" answer — same shape, ``degraded: true``."""
    return {
        "task_id": task_id,
        "status": "pending",
        "takes": [],
        "eta_seconds": None,
        "degraded": True,
        "result_parse_error": None,
    }


@router.get("/api/generator/audio")
async def proxy_take_audio(
    request: Request,
    path: str = Query(..., min_length=1),
    token: str | None = Query(None),
):
    """Stream a take's audio from ACE-Step. Allowed during a live set.

    The browser never talks to :8001: auth (header or ``?token=``, since
    ``<audio>`` cannot set headers) and LAN isolation live here.
    ``path`` is the take's ``file`` field, forwarded UNREWRITTEN to
    ``AceStepClient.audio_url`` — but not unexamined:
    :func:`_validate_proxy_path` refuses anything carrying a host,
    naming a local file, or pointing outside ``ACESTEP_AUDIO_ROOT``,
    the same rule publish applies (400 here, 422 there).

    Range-friendly the cheap way: the client's ``Range`` header is
    forwarded and upstream's status (206) plus its range headers are
    mirrored back, so ``<audio>`` seeking works the way it does on
    ``FileResponse`` elsewhere without buffering a 35 MB WAV in the
    backend.
    """
    _authorize_audio(request, token)
    safe_path = _validate_proxy_path(path)

    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)

    forwarded = {}
    range_header = request.headers.get("range")
    if range_header:
        forwarded["range"] = range_header

    stream = client.stream_audio(safe_path, headers=forwarded)
    try:
        upstream = await stream.__aenter__()
    except (acestep_client.AceStepDisabled, acestep_client.AceStepUnavailable) as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except acestep_client.AceStepError as exc:
        status = 404 if exc.status_code == 404 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except ValueError as exc:  # audio_url() rejected an empty path
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {
        name: value
        for name in _PROXY_HEADERS
        if (value := upstream.headers.get(name))
    }

    async def body():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            # Closes the response AND the httpx client behind it, on a
            # client disconnect as much as on a clean finish.
            await stream.__aexit__(None, None, None)

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type") or "audio/wav",
    )


# ── POST /api/generator/publish (G2b) ────────────────────────────────


async def _download_take(client: acestep_client.AceStepClient, api_path: str, dest: str):
    """Stream one take's audio to ``dest``. Typed refusals, no buffering."""
    stream = client.stream_audio(api_path)
    try:
        upstream = await stream.__aenter__()
    except (acestep_client.AceStepDisabled, acestep_client.AceStepUnavailable) as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except acestep_client.AceStepError as exc:
        status = 404 if exc.status_code == 404 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        with open(dest, "wb") as fh:
            async for chunk in upstream.aiter_bytes(PUBLISH_CHUNK_BYTES):
                fh.write(chunk)
    finally:
        await stream.__aexit__(None, None, None)


@router.post("/api/generator/publish")
async def publish_take(
    req: PublishRequest,
    background: BackgroundTasks,
    current_user: dict = Depends(auth.get_current_user),
):
    """Land one take in the catalog: download → G2a's ingest → entry.

    Refusal ladder:

    * **503** — the generator is disabled or the box is unreachable.
      Unlike every other refusal here this one is structural: the take's
      audio only exists on the ACE box, so without it there is nothing
      to publish.
    * **422** — a bad body, a ``file`` that is not an ACE result path,
      or an INGEST refusal (short track, unknown genre, out-of-window
      bpm, unparseable keyscale, id collision). Ingest refusals are
      passed through **verbatim**: they already name the window, the
      floor or the colliding id, which is exactly what the user has to
      act on, and paraphrasing them would fork the wording from the CLI.
    * **404 / 502** — the result file is gone, or ACE broke serving it.

    **No 409 here, deliberately.** The VRAM guard exists because
    releasing a task loads ~12.5 GB of model onto a GPU shared with the
    live DJ. Publishing touches the disk, not the GPU: it downloads a
    file ACE rendered earlier (its result files survive restarts) and
    writes a WAV plus a tracks.json entry. Refusing it during a set
    would cost the operator a take for no VRAM saved.

    Concurrency, on the other hand, is real but not enforceable here —
    see the ``--build-catalog`` note in ``web/CLAUDE.md``. The cheap
    freshness the ingest already provides is that it re-reads
    tracks.json inside the same call that appends to it and backs the
    file up first, so the loser of a race loses one entry, not the
    catalog.
    """
    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)

    try:
        resolved = validate_ace_audio_path(req.file, resolve_file=True)
    except AceAudioPathError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Cheap pre-flight: refuse a take that could never be selected into a
    # session BEFORE pulling 35 MB across the LAN. The ingest re-checks
    # against the probed duration — this only trusts ACE's own number.
    from agent.eligibility import MIN_TRACK_DURATION_SEC  # noqa: PLC0415

    if req.metas.duration is not None and req.metas.duration < MIN_TRACK_DURATION_SEC:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the take is {req.metas.duration:.1f}s long — shorter than the "
                f"{MIN_TRACK_DURATION_SEC:.0f}s minimum for session eligibility, "
                f"so it could never be selected into a session"
            ),
        )

    if req.prompt:
        # tracks.json has no prompt column; keep the provenance in the log
        # rather than dropping it silently or forking the catalog schema.
        print(
            f"[generator] publishing '{req.display_name}' from prompt: "
            f"{req.prompt.strip()[:160]}",
            flush=True,
        )

    work_dir = tempfile.mkdtemp(prefix="apollo-publish-")
    try:
        wav_path = os.path.join(work_dir, "take.wav")
        await _download_take(client, resolved.api_path, wav_path)

        lyrics_path = None
        if req.lyrics and req.lyrics.strip():
            # The ingest takes lyrics as a FILE (its `--lyrics` flag); the
            # wire carries TEXT, matching the sidecar's `lyrics` key. One
            # temp file bridges the two without a second ingest entry point.
            lyrics_path = os.path.join(work_dir, "lyrics.txt")
            with open(lyrics_path, "w", encoding="utf-8") as fh:
                fh.write(req.lyrics)

        # `main` is imported HERE, inside the handler, and never at module
        # scope: it costs a measured 2.61 s and ~1800 modules (librosa,
        # numba, moviepy, pedalboard). That is unacceptable on the import
        # path of a web backend — it is the G1 rule that keeps
        # `_genre_bpm_windows` bound to `agent.tools` instead. Publishing
        # is a rare, human-triggered action that already spends seconds
        # downloading a WAV and running ffmpeg, so paying it once here is
        # the right trade, and it is the ONLY way to run the identical
        # ingest the CLI runs (`--ingest`) rather than reimplementing the
        # catalog conventions and drifting from them.
        main = await asyncio.to_thread(importlib.import_module, "main")

        try:
            entry = await asyncio.to_thread(
                main.ingest_track,
                wav_path,
                req.genre_folder,
                display_name=req.display_name,
                bpm=req.metas.bpm,
                keyscale=req.metas.keyscale,
                variant_of=req.variant_of,
                lyrics=lyrics_path,
            )
        except main.IngestRefused as exc:
            raise HTTPException(status_code=422, detail=exc.message) from exc
        except SystemExit as exc:  # pragma: no cover — defensive
            raise HTTPException(
                status_code=500, detail=f"ingest exited unexpectedly: {exc}"
            ) from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # G6: the take stops being "one of a batch" and becomes a catalog
    # track. Matched by the decoded path, user-scoped, and never allowed
    # to turn a successful publish into a failure.
    await _record_publish(
        current_user["id"], resolved.file_path or req.file, entry["id"]
    )

    # G7: give the new track a cover. 510 of the catalog's 513 tracks
    # already show one (imported from Suno with a remote cover_url); the
    # blanks were exactly the tracks made at home, which is every track
    # this endpoint creates. Runs in the BACKGROUND: an image call takes
    # ~20 s and publish has already done the slow work (download +
    # beat-analysing ingest), so blocking the response on artwork would
    # double a wait the user is already staring at. The cover appears on
    # the next catalog load. Failures are logged inside generate_cover
    # and never surface — same rule as _record_publish above.
    background.add_task(
        covers.generate_cover,
        entry["id"],
        entry["display_name"],
        req.genre_folder,
    )

    return {
        "track_id": entry["id"],
        "file": entry["file"],
        "display_name": entry["display_name"],
        "camelot_key": entry["camelot_key"],
        "bpm": entry["bpm"],
        "variant_of": entry["variant_of"],
        "note": FIX_INCOMPLETE_NOTE,
    }


# ── POST /api/generator/edit (G3) ────────────────────────────────────


async def _release_edit(
    client: acestep_client.AceStepClient,
    payload: dict[str, Any],
    resolved: AceAudioPath,
) -> acestep_client.ReleaseTaskResult:
    """Release the edit, degrading to a multipart upload if ACE refuses.

    The happy path hands ACE a path on its OWN disk: the take is already
    there, so nothing crosses the LAN twice. ACE validates that path
    against its process's ``gettempdir()`` though, and a foreign
    ``TMPDIR`` in its launch env points that somewhere else — at which
    point the box 400s the very paths it handed out (the plan's G3
    fallback caveat).

    That specific 400 is not a failure, it is a slower route: download
    the take through the proxy's own client and re-release it as a
    multipart upload (spec §3.3's ``src_audio`` field). Every OTHER 400
    is a real bad request and propagates into the normal taxonomy — a
    mistyped ``task_type`` must not be answered by uploading 35 MB and
    failing the same way.
    """
    try:
        return await client.release_task(payload)
    except acestep_client.AceStepBadRequest as exc:
        if not _is_absolute_path_refusal(exc):
            raise
        print(
            "[generator] edit degraded to multipart: ACE-Step refused "
            f"src_audio_path '{resolved.file_path}' ({exc.message}) — "
            "uploading the take instead",
            flush=True,
        )

    name = posixpath.basename(resolved.file_path or "") or "source.wav"
    work_dir = tempfile.mkdtemp(prefix="apollo-edit-")
    try:
        local = os.path.join(work_dir, "source.wav")
        await _download_take(client, resolved.api_path, local)
        upload = {k: v for k, v in payload.items() if k != "src_audio_path"}
        with open(local, "rb") as fh:
            return await client.release_task(
                upload,
                files={EDIT_UPLOAD_FIELD: (name, fh, EDIT_UPLOAD_CONTENT_TYPE)},
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post("/api/generator/edit")
async def edit_take(
    req: EditRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """Re-release one take as a repaint / cover / completion (plan: G3).

    **This releases GPU work**, so unlike publish it carries the full
    ``POST /tasks`` refusal ladder, 409 included: an edit loads the same
    ~12.5 GB onto the GPU the live DJ's model needs. The message is
    :data:`VRAM_CONFLICT_MESSAGE`, rendered verbatim by the wizard.

    * **503** — generator disabled or the box unreachable.
    * **409** — a live set is on air.
    * **422** — a ``file`` that is not an ACE result path (root-checked:
      the value goes to ACE as ``src_audio_path``), an unknown
      ``genre_folder``, or a parameter that belongs to another mode.
    * **429** — ACE-Step's queue is full.
    * **502** — ACE answered but rejected/broke.

    Returns ``{task_id, queue_position, eta_seconds}`` — the same shape
    ``POST /tasks`` returns, polled by the same
    ``GET /tasks/{task_id}``. An edit is just another task; what makes
    it an edit is its source, which the PAGE remembers (that is where
    the chained "edited from …" card gets its lineage).
    """
    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    if live_session_active():
        raise HTTPException(status_code=409, detail=VRAM_CONFLICT_MESSAGE)

    try:
        resolved = validate_ace_audio_path(req.file, resolve_file=True)
    except AceAudioPathError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    bpm = None
    # genre_folder is optional on an edit — seed the key so the style
    # composition below degrades to the bare prompt instead of NameError.
    genre_key = ""
    if req.genre_folder:
        genre_key = await asyncio.to_thread(_resolve_genre, req.genre_folder)
        bpm = _default_bpm_for(genre_key)
        if bpm is None:
            print(
                f"[generator] no BPM window for genre '{genre_key}' — "
                "editing without a server-pinned bpm",
                flush=True,
            )

    payload = _edit_payload(
        req, resolved.file_path or req.file, bpm, genre_key
    )
    try:
        released = await _release_edit(client, payload, resolved)
    except (acestep_client.AceStepDisabled, acestep_client.AceStepUnavailable) as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except acestep_client.AceStepQueueFull as exc:
        raise HTTPException(
            status_code=429,
            detail="ACE-Step's queue is full — retry in a moment.",
        ) from exc
    except acestep_client.AceStepError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # G6: an edit is an ordinary generation with a source, so it gets its
    # own row. The lineage rides in ``request_json`` (``task_type`` +
    # ``src_audio_path``), which is what makes it queryable server-side —
    # the page's chained card remains the UI's own memory.
    await _record_release(
        current_user["id"], released.task_id, payload, req.genre_folder
    )

    stats = await _stats_or_none(client)
    return {
        "task_id": released.task_id,
        "queue_position": released.queue_position,
        "eta_seconds": _eta_seconds(stats, released.queue_position),
    }


# ── POST /api/generator/critique (G4) ────────────────────────────────
#
# Two layers, one endpoint, and they fail independently:
#
#   1. the SCORE — ``agent.generative.bench.bench_wav``, the same
#      function the generative engine's merge gate runs. An LLM cannot
#      hear; this can. Without it there is no number worth printing.
#   2. the READ — one LLM completion over those numbers plus what the
#      user asked for. Nice to have, never load-bearing: any trouble at
#      all (no provider, unreachable box, timeout, empty reply) answers
#      ``critique: null``.
#
# Neither layer gates publishing. The bench's own philosophy is
# automated evidence + human decision, and a take the bench dislikes is
# still the operator's to keep.


#: Hard bound on the LLM read. The gateway is the same tunnelled LiteLLM
#: / LM Studio node the live DJ speaks to, and the wizard is waiting on
#: this request, so the read gets a fraction of ``brief_parser``'s 45 s:
#: the score is already worth showing without it, and a paragraph that
#: costs the operator half a minute is not worth waiting for. Enforced
#: with ``asyncio.wait_for`` on top of the SDK's own timeout — the SDK's
#: is the polite bound, this one is the real one. A worker thread that
#: outlives the deadline is simply abandoned.
CRITIQUE_TIMEOUT_SEC = 15.0

#: Completion budget. The answer is ~80 words, but assume a reasoner: it
#: spends its budget thinking before the first content token, which is
#: exactly how a 512 ceiling once truncated ``brief_parser`` into
#: silence. Headroom costs a non-reasoning model nothing.
CRITIQUE_MAX_TOKENS = 2048

#: Paragraph ceiling. A model that ignores "one paragraph" must not be
#: able to push a wall of text into the wizard's take row.
CRITIQUE_MAX_CHARS = 1200

#: Provider defaults, mirroring the house wiring. The model itself is
#: ``GENERATIVE_MODEL`` > ``AGENT_MODEL`` > these (the #123 precedent,
#: shared with ``agent/generative/strudel_mind.py``): the critic is a
#: generative-lane job and must be free to run on a different model from
#: the live DJ, which is tuned for tool calls rather than prose.
CRITIQUE_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "ollama": "gemma4:4b",
    "litellm": "qwen3.6-27b",
}

#: Override for the reference band file. Unset (the normal case) means
#: the bench's own committed ``quality_references.json`` — the same
#: numbers ``scripts/quality_bench.py`` gates the generative engine on,
#: which is the whole point of scoring against them. Read at CALL time,
#: never captured as a default argument: ``--reload`` plus a late
#: ``.env`` is the house rule, and it is also what lets a test point the
#: endpoint at bands it built around a synthetic take.
ENV_BENCH_REFERENCES = "APOLLO_BENCH_REFERENCES"

CRITIQUE_SYSTEM = (
    "You are the critic of an automated DJ catalog. You are given "
    "measurements of ONE generated take and the request it came from. "
    "You cannot hear the audio: the numbers are your ears, and they were "
    "produced by the same quality bench the project gates its own "
    "generative engine on.\n"
    "Answer with ONE short paragraph of plain prose (80 words at most, no "
    "markdown, no lists, no headings, no preamble): does this take match "
    "what was asked for, and what is the one thing you would change in the "
    "next attempt. Be concrete about the numbers you were given. If they "
    "sit inside their bands, say so plainly instead of inventing a fault. "
    "Never claim to have listened."
)


class CritiqueMetas(BaseModel):
    """ACE's ``metas`` for the take being scored.

    ``extra="ignore"``, like :class:`PublishMetas` — the block's shape
    belongs to ACE and an upgrade that adds a field must not start
    422-ing a read-only request.

    Every field is OPTIONAL here, unlike publish. Publishing refuses to
    guess a bpm or a key because the catalog would carry the guess
    forever; scoring writes nothing, and a take whose metas failed to
    parse is exactly the take an operator most wants a second opinion
    on. What is absent is simply absent from the LLM's brief.
    """

    model_config = ConfigDict(extra="ignore")

    bpm: float | None = Field(None, ge=MIN_BPM, le=MAX_BPM)
    keyscale: str | None = Field(None, max_length=64)
    duration: float | None = Field(None, gt=0)


class CritiqueRequest(BaseModel):
    """Score one take against its genre's references (plan: G4).

    ``extra="forbid"`` and no ``task_id``, for the reasons
    :class:`PublishRequest` documents: the page persists the take's
    decoded path and hands it back, because ACE's job records are mortal
    and its result files are not.
    """

    model_config = ConfigDict(extra="forbid")

    #: The take's URL-DECODED ACE path, as persisted by the page. It
    #: names a file this handler downloads, so it is root-checked.
    file: str = Field(..., min_length=1, max_length=4096)
    metas: CritiqueMetas = Field(default_factory=CritiqueMetas)
    #: What the take was asked to be — the half of "does this match?"
    #: the bench knows nothing about.
    prompt: str | None = Field(None, max_length=4000)
    #: Picks the reference band set. NOT run through
    #: :func:`_resolve_genre`: this endpoint writes nothing, the value
    #: only chooses which numbers to compare against, and the bench's own
    #: refusal ("no references for genre 'techno' (has: ambient, deep,
    #: lofi)") tells the operator more than "unknown genre_folder" would.
    genre_folder: str = Field(..., min_length=1, max_length=120)


def _reference_genre(genre_folder: str, genre_folders: dict[str, str]) -> str:
    """Map a catalog ``genre_folder`` to a bench reference key.

    The bench's references are keyed by GENRE (``lofi``, ``ambient``,
    ``deep``) while the wizard speaks in FOLDERS (``lofi - ambient``,
    ``deep house``), and ``bench.GENRE_FOLDERS`` is the map between them
    — one folder can back several genres, which is why this needs a rule
    rather than a lookup.

    The rule: a genre matching the folder's leading text wins
    (``lofi - ambient`` scores as ``lofi``, the folder's primary genre),
    otherwise the first candidate in sorted order, so the answer is
    deterministic. A folder no genre claims passes through UNCHANGED —
    that hands the raw name to ``bench_wav``, whose refusal then names
    both the genre and the ones that do have references, which is the
    message the wizard shows.
    """
    key = (genre_folder or "").strip().lower()
    candidates = sorted(
        genre for genre, folder in genre_folders.items()
        if folder.strip().lower() == key
    )
    if not candidates:
        return key
    for candidate in candidates:
        if key.startswith(candidate):
            return candidate
    return candidates[0]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _band(lo: Any, hi: Any, *, pad_lo, pad_hi, digits: int) -> dict | None:
    """One metric's band: what fails, plus the raw reference range."""
    if not (_is_number(lo) and _is_number(hi)):
        return None
    return {
        "min": round(pad_lo(lo), digits),
        "max": round(pad_hi(hi), digits),
        "reference_min": round(float(lo), digits),
        "reference_max": round(float(hi), digits),
    }


def _reference_bands(
    genre_ref: dict, *, centroid_ratio: float, tilt_delta: float
) -> dict | None:
    """The bands the wizard draws its chips against.

    ``min``/``max`` are the EFFECTIVE band — the reference range widened
    by the bench's own margins — because that is the band that decides
    ``passed``, and a chip reading "out of band" for a value the bench
    passed would be contradicting the verdict next to it. The raw catalog
    range travels alongside as ``reference_min``/``reference_max``, so a
    value that clears the margin but sits far from real records still
    reads as far.

    ``advisory_lufs`` gets no margin: nothing advisory can fail, so its
    band IS the reference range. Keeping it in the same shape as the
    other two is what lets the frontend fold every chip with one rule.
    """
    if not genre_ref:
        return None
    centroid = genre_ref.get("centroid_hz") or {}
    tilt = genre_ref.get("tilt_db_per_oct") or {}
    lufs = genre_ref.get("advisory_lufs") or {}
    bands = {
        "centroid_hz": _band(
            centroid.get("min"), centroid.get("max"),
            pad_lo=lambda v: v / centroid_ratio,
            pad_hi=lambda v: v * centroid_ratio,
            digits=1,
        ),
        "tilt_db_per_oct": _band(
            tilt.get("min"), tilt.get("max"),
            pad_lo=lambda v: v - tilt_delta,
            pad_hi=lambda v: v + tilt_delta,
            digits=2,
        ),
        "advisory_lufs": _band(
            lufs.get("min"), lufs.get("max"),
            pad_lo=float, pad_hi=float, digits=1,
        ),
    }
    return {k: v for k, v in bands.items() if v is not None} or None


def _no_verdict(genre: str, note: str) -> dict:
    """The answer when the bench cannot put a number on this take.

    A 200, never an error. Every refusal ``bench_wav`` raises lands here
    — no references for the genre, an unreadable download, a broken
    references file — because this endpoint is advisory by construction:
    the wizard asked for a second opinion, and "there isn't one, here is
    why" is a complete answer, while a 5xx would read as a fault the
    operator has to fix before publishing (which it never is). The note
    is the bench's OWN message, so the wording has one source.
    """
    return {
        "passed": None,
        "reference_genre": genre,
        "reference_informed": None,
        "advisory": None,
        "bands": None,
        "failures": [],
        "critique": None,
        "note": note,
    }


def _critique_brief(req: CritiqueRequest, report: dict, bands: dict | None) -> str:
    """The user half of the LLM read: numbers, bands, and what was asked.

    Pure and text-only, so what the model sees is exactly what a test can
    assert on. Never includes a file path — the model has no use for one
    and it is the one field here that names the box's filesystem.
    """
    audio = report.get("audio") or {}
    ri = audio.get("reference_informed") or {}
    adv = audio.get("advisory") or {}
    metas = req.metas

    def _band_text(key: str) -> str:
        band = (bands or {}).get(key) or {}
        if not band:
            return "no band"
        return f"band {band['min']}–{band['max']}"

    reported = ", ".join(
        part for part in (
            f"{metas.bpm:g} BPM" if metas.bpm is not None else "",
            f"key {metas.keyscale.strip()}" if (metas.keyscale or "").strip() else "",
            f"{metas.duration:.0f}s" if metas.duration is not None else "",
        ) if part
    )
    lines = [
        f"Asked for: {(req.prompt or '').strip() or '(no prompt recorded)'}",
        f"Genre folder: {req.genre_folder} "
        f"(scored against '{report.get('genre')}' references)",
        f"Reported metadata: {reported or 'none reported'}",
        "",
        "Measured (reference-informed — these decide the verdict):",
        f"- spectral centroid {ri.get('centroid_hz')} Hz ({_band_text('centroid_hz')})",
        f"- spectral tilt {ri.get('tilt_db_per_oct')} dB/oct "
        f"({_band_text('tilt_db_per_oct')})",
        "",
        "Measured (advisory — reported, never a failure):",
        f"- {adv.get('lufs')} LUFS, LRA {adv.get('lra')}, "
        f"crest {adv.get('crest_db')} dB",
        "",
        "Bench verdict: " + (
            "PASS" if report.get("passed")
            else "FAIL — " + "; ".join(report.get("reference_informed_failures") or [])
        ),
    ]
    return "\n".join(lines)


def _resolve_critique_model(provider: str) -> str:
    """``GENERATIVE_MODEL`` > ``AGENT_MODEL`` > the provider's default."""
    fallback = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "") if provider == "azure"
        else CRITIQUE_DEFAULT_MODELS.get(provider, "")
    )
    return os.getenv("GENERATIVE_MODEL") or os.getenv("AGENT_MODEL") or fallback


def _llm_paragraph(system: str, user: str, provider: str) -> str:
    """ONE completion against whichever provider the env has wired.

    Blocking on purpose — called through ``asyncio.to_thread`` under a
    hard ``wait_for``, the way ``brief_parser.parse`` is. It deliberately
    does NOT reuse ``brief_parser``'s client builder: that one carries a
    45 s timeout and the ``BRIEF_MODEL`` precedence, both right for
    parsing a brief and wrong for a paragraph the wizard is waiting on.
    What IS shared is the provider detection, so there is one answer to
    "which LLM is this box wired to".

    This is the single seam the tests replace; everything above it stays
    honest about degradation.
    """
    model = _resolve_critique_model(provider)
    if not model:
        raise RuntimeError(f"no model configured for provider {provider!r}")

    if provider == "anthropic":
        from anthropic import Anthropic  # noqa: PLC0415 — heavy, optional SDK

        anthropic_client = Anthropic(timeout=CRITIQUE_TIMEOUT_SEC)
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=CRITIQUE_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in message.content if hasattr(b, "text"))

    if provider == "azure":
        from openai import AzureOpenAI  # noqa: PLC0415

        client: Any = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout=CRITIQUE_TIMEOUT_SEC,
        )
    else:
        from openai import OpenAI  # noqa: PLC0415

        if provider == "litellm":
            base_url = os.environ["LITELLM_BASE_URL"]
            api_key = os.getenv("LITELLM_API_KEY", "sk-litellm")
        else:
            # The generic OpenAI-compatible path: LM Studio over the
            # tunnel is what this actually is, not Ollama specifically.
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api_key = "ollama"
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=CRITIQUE_TIMEOUT_SEC)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=CRITIQUE_MAX_TOKENS,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _clean_paragraph(text: str | None) -> str | None:
    """One line of prose, or ``None`` if there is nothing usable left.

    Reasoning models leak ``<think>`` blocks into the content and small
    ones fence prose as if it were code, so both are stripped before the
    whitespace is collapsed. An empty result is treated exactly like a
    failed call: the take renders its numbers without a read.
    """
    if not text:
        return None
    body = text
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1]
    body = " ".join(body.replace("```", " ").split()).strip()
    if not body:
        return None
    return body[:CRITIQUE_MAX_CHARS].strip()


async def _critique_paragraph(
    req: CritiqueRequest, report: dict, bands: dict | None
) -> str | None:
    """The optional LLM read. **Never raises, never retries.**

    Degradation is the contract, not a safety net: no provider wired,
    ``AGENT_PROVIDER=mock``, a box that does not answer, a reply that is
    all thinking and no prose — every one of them is ``None``, and the
    wizard shows the bench numbers alone. A retry would double the wait
    the operator is already sitting through, for the least important
    thing on the panel.
    """
    provider = detect_provider()
    if provider == "mock":
        # The E2E and unit suites set this precisely so nothing reaches a
        # network. Short-circuit BEFORE any SDK import.
        return None

    brief = _critique_brief(req, report, bands)
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_llm_paragraph, CRITIQUE_SYSTEM, brief, provider),
            timeout=CRITIQUE_TIMEOUT_SEC,
        )
    except (asyncio.TimeoutError, TimeoutError):
        print(
            f"[generator] critique LLM timed out after {CRITIQUE_TIMEOUT_SEC:g}s "
            f"(provider {provider}) — scoring without a read",
            flush=True,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — the read is never load-bearing
        print(
            f"[generator] critique LLM unavailable ({provider}): {exc} — "
            "scoring without a read",
            flush=True,
        )
        return None
    return _clean_paragraph(raw)


@router.post("/api/generator/critique")
async def critique_take(
    req: CritiqueRequest,
    current_user: dict = Depends(auth.get_current_user),
):
    """Score one take: the bench puts the number on it, the LLM reads it.

    Flow: validate the path (the shared validator, root-checked — this
    handler downloads the file) → pull the take through ``stream_audio``
    → ``bench_wav`` against the genre's committed references → one
    optional LLM paragraph over those numbers and the original prompt.

    Answers ``{passed, reference_genre, reference_informed, advisory,
    bands, failures, critique, note}``. Every key is always present, the
    poll endpoint's rule: a UI that has to feature-detect its own
    contract eventually gets it wrong.

    * **503** — the generator is disabled or the box unreachable.
      Structural, exactly as on publish: the take's audio exists only on
      the ACE box, so there is nothing to measure without it.
    * **422** — a ``file`` that is not an ACE result path.
    * **404 / 502** — the result file is gone, or ACE broke serving it.
    * **200 with ``passed: null`` and a ``note``** — the bench refused
      (most often: this genre has no committed references yet). Not an
      error; see :func:`_no_verdict`.
    * **200 with ``critique: null``** — the LLM layer was off or did not
      answer in time. Also not an error.

    **No 409, deliberately** — publish's exemption, with one wrinkle
    worth naming. The VRAM guard exists because releasing a task parks
    ~12.5 GB of ACE-Step on the GPU the live DJ's model needs, and it
    does not give it back. This endpoint parks nothing: it reads a file
    ACE rendered earlier and runs the bench on the CPU. The LLM read is
    the wrinkle — it does travel to the same tunnelled gateway the live
    DJ speaks to, so it is not literally free during a set. But it is
    ONE short completion under :data:`CRITIQUE_TIMEOUT_SEC`, competing
    for a few seconds of queue rather than for resident VRAM, and it
    abandons itself rather than waiting. Refusing an operator the score
    on a take they are about to publish, for that, would be the wrong
    trade — the guard protects residency, not politeness.
    """
    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)

    try:
        resolved = validate_ace_audio_path(req.file, resolve_file=True)
    except AceAudioPathError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    work_dir = tempfile.mkdtemp(prefix="apollo-critique-")
    try:
        wav_path = os.path.join(work_dir, "take.wav")
        await _download_take(client, resolved.api_path, wav_path)

        # `agent.generative.bench` is imported HERE, inside the handler,
        # for the reason `main` is in publish: it pulls the librosa
        # family (numpy + soundfile at module scope, librosa itself the
        # moment a foreign sample rate needs resampling — which every
        # 48 kHz ACE take does). That is import weight a web backend must
        # not pay at module scope, and scoring is a rare, human-triggered
        # action that already spent seconds downloading a WAV.
        bench = await asyncio.to_thread(
            importlib.import_module, "agent.generative.bench"
        )
        genre = _reference_genre(req.genre_folder, bench.GENRE_FOLDERS)
        references = os.getenv(ENV_BENCH_REFERENCES) or bench.REFERENCES_PATH
        try:
            report, passed = await asyncio.to_thread(
                bench.bench_wav, wav_path, genre, references_path=references
            )
        except bench.BenchInputError as exc:
            # A genre nobody has extracted references for is the common
            # case here, and it is a normal state — the catalog grows
            # faster than the reference sweep.
            print(f"[generator] no bench verdict for '{genre}': {exc}", flush=True)
            return _no_verdict(genre, str(exc))
    finally:
        # The WAV's whole job was to be measured. The LLM read below
        # works from numbers, so the download does not outlive the bench.
        shutil.rmtree(work_dir, ignore_errors=True)

    audio = report.get("audio") or {}
    bands = _reference_bands(
        report.get("reference") or {},
        centroid_ratio=bench.CENTROID_RATIO_MAX,
        tilt_delta=bench.TILT_DELTA_MAX,
    )
    return {
        "passed": bool(passed),
        "reference_genre": genre,
        "reference_informed": audio.get("reference_informed"),
        "advisory": audio.get("advisory"),
        "bands": bands,
        # The bench's own words, so the chips and the sentence under them
        # cannot drift apart.
        "failures": list(report.get("reference_informed_failures") or []),
        "critique": await _critique_paragraph(req, report, bands),
        "note": None,
    }


# ── The generations library (G6) ─────────────────────────────────────
#
# Three read/repair routes over what the hooks above recorded. They are
# the ONLY generator endpoints that never speak to ACE-Step at all —
# except ``refresh``, which is the resume lane for a generation whose
# page died mid-flight.


class TakeStateUpdate(BaseModel):
    """``PATCH .../takes/{idx}`` body — the two states a human may set.

    ``published`` is deliberately NOT in the Literal: a take becomes
    published by BEING published (the publish hook writes the state and
    the catalog id together), and letting the feed assert it would put a
    track id in the row that no catalog entry backs. A body asking for it
    is a 422 from pydantic, alongside every other unknown state.
    """

    model_config = ConfigDict(extra="forbid")

    state: Literal["discarded", "fresh"]


async def _own_generation(generation_id: str, user: dict) -> dict:
    """One generation of THIS user's, or 404 — the ``_own_playlist`` rule.

    Unknown and someone else's answer identically on purpose: a 403 would
    confirm the id exists, and the feed is per-user by construction.
    """
    generation = await asyncio.to_thread(db.get_generation, generation_id)
    if not generation or generation["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Generation not found")
    return generation


@router.get("/api/generator/generations")
async def list_generations(
    limit: int = Query(db.DEFAULT_GENERATIONS_LIMIT, ge=1, le=db.MAX_GENERATIONS_LIMIT),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(auth.get_current_user),
):
    """The feed: this user's generations, newest first, takes embedded.

    A bare JSON array, the ``GET /api/playlists`` / ``GET /api/sessions``
    shape — the page and offset are the caller's own query, so echoing
    them back would only be a second place for them to disagree.

    Each take is POLL-SHAPED (``index``, ``file``, ``prompt``, ``lyrics``,
    ``metas``, ``seed_value``) so the library renders through the wizard's
    existing take components, plus the three things only the store knows:
    ``decoded_path`` (what publish/edit/score want, already decoded),
    ``state`` and ``published_track_id``.

    ``limit`` is 1–100 (default 20) and ``offset`` ≥ 0; both are enforced
    by FastAPI, so a 200-item page is a 422 rather than a slow query.
    """
    return await asyncio.to_thread(
        db.list_generations_by_user, current_user["id"], limit=limit, offset=offset
    )


@router.patch("/api/generator/generations/{generation_id}/takes/{idx}")
async def set_generation_take_state(
    generation_id: str,
    idx: int,
    body: TakeStateUpdate,
    current_user: dict = Depends(auth.get_current_user),
):
    """Discard a take, or bring a discarded one back. Returns the take.

    * **404** — no such generation, no such take, or not this user's.
    * **422** — a state outside ``discarded``/``fresh``, ``published``
      included (see :class:`TakeStateUpdate`).

    Patching a take that was published is allowed and keeps its
    ``published_track_id``: the catalog entry is a fact about the past,
    not a state the feed owns, so discarding the row hides it without
    pretending the track was never made.
    """
    generation = await _own_generation(generation_id, current_user)
    if not any(take["index"] == idx for take in generation["takes"]):
        raise HTTPException(
            status_code=404,
            detail=f"generation '{generation_id}' has no take {idx}",
        )

    await asyncio.to_thread(db.set_take_state, generation_id, idx, body.state)
    refreshed = await asyncio.to_thread(db.get_generation, generation_id)
    takes = (refreshed or generation)["takes"]
    return next(take for take in takes if take["index"] == idx)


@router.post("/api/generator/generations/{generation_id}/refresh")
async def refresh_generation(
    generation_id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    """Re-poll ACE for a ``pending`` generation — the resume lane.

    A wizard tab that dies between the release and the first done-poll
    leaves a row that nothing else will ever finish. This is the button
    that finishes it, inside ACE's 24 h record window.

    **``stale`` and ``degraded`` are different answers and must never be
    conflated.** ACE ANSWERING that it has no such task is terminal — the
    record window closed, and no amount of retrying brings the job back,
    so the generation becomes ``stale``. ACE not answering at all (a
    transport blip, a 500, a restarting box) says nothing about the job:
    the generation stays ``pending`` and the response carries
    ``degraded: true``, the poll endpoint's own word for "ask again".

    * **404** — unknown generation, or not this user's.
    * **409** — the generation is already ``done``/``failed``/``stale``.
      All three are terminal, so a refresh has nothing to do; the detail
      names the current status.
    * **503** — the generator is disabled or the box is unreachable.
    * **502** — an auth/bad-request failure retrying cannot fix (the poll
      endpoint's one loud case).

    Returns the generation in the feed's shape plus ``degraded``.
    """
    generation = await _own_generation(generation_id, current_user)
    if generation["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=(
                f"generation '{generation_id}' is {generation['status']} — refresh "
                "is the resume lane for a generation still pending; a terminal one "
                "has nothing left to poll for"
            ),
        )

    client = _client()
    if not client.enabled():
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)

    try:
        results = await client.query_result([generation_id])
    except acestep_client.AceStepDisabled as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc
    except _POLL_DEGRADE_ERRORS as exc:
        print(f"[generator] refresh of {generation_id} degraded: {exc}", flush=True)
        return {**generation, "degraded": True}
    except acestep_client.AceStepError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    entry = next((r for r in results if r.task_id == generation_id), None)
    if entry is None and len(results) == 1 and not results[0].task_id:
        entry = results[0]  # answered positionally, without echoing the id

    if entry is None:
        # The box answered and does not know this task. Unlike the poll
        # endpoint — where an id ACE has not registered YET must not tear
        # the wizard's card down — a refresh is asked about a generation
        # released long enough ago to have been abandoned, so the honest
        # answer is that the record is gone for good.
        print(
            f"[generator] refresh of {generation_id}: ACE-Step answered without "
            "the task — its record window closed, marking stale",
            flush=True,
        )
        await _record_status(current_user["id"], generation_id, "stale")
    else:
        status = TASK_STATUS_NAMES.get(entry.status, "pending")
        takes = [
            {
                "index": index,
                "file": take.file,
                "prompt": take.prompt,
                "lyrics": take.lyrics,
                "metas": take.metas,
                "seed_value": take.seed_value,
            }
            for index, take in enumerate(entry.takes)
        ]
        if status == "done":
            await _record_takes(current_user["id"], generation_id, takes)
        elif status == "failed":
            await _record_status(current_user["id"], generation_id, "failed")

    refreshed = await asyncio.to_thread(db.get_generation, generation_id)
    return {**(refreshed or generation), "degraded": False}
