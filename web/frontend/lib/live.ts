"use client";
/**
 * useLiveSession — bridge between the browser audio decks and the
 * v2.5.1 ``/ws/live/{id}`` backend WebSocket.
 *
 * The backend's ``LiveEngineBrowser`` maintains the playlist state machine
 * but never reads or writes audio — it relies on the browser to drive
 * playback. This hook owns:
 *
 *   - Two ``BufferDeck`` instances (deck A / deck B) — v3.4 replaced the
 *     prior HTMLAudioElement substrate with AudioBufferSourceNode-based
 *     decks scheduled on the dedicated audio rendering thread. The
 *     active deck plays the current track; the inactive deck is preloaded
 *     with the next track's PCM AudioBuffer so a crossfade is just a
 *     sample-accurate start(when) plus a gain ramp scheduled at the
 *     SAME ``when`` — eliminating the 10–50 ms "cabalgar" / phase walk
 *     the MediaElementAudioSourceNode + MP3 frame-quantised seek used
 *     to cause.
 *   - A ``BufferCache`` keyed by stream URL so the next track's decode
 *     (~0.5–2 s) is amortised across the APPROACHING_CF window and the
 *     actual transition scheduling is synchronous.
 *   - State derived from server engine events (``track_started`` /
 *     ``approaching_crossfade`` / ``crossfade_*`` / ``track_ended`` /
 *     ``session_ended``) plus the running countdown to the next crossfade
 *     based on the active deck's virtual position.
 *   - A throttled ``playback_pos`` ping back to the server every ~250 ms
 *     so the engine knows where the browser is in the track. The interval
 *     is set up in a ``useEffect`` cleanup pair (canonical v2.4 pattern —
 *     no setState inside the effect, no ref writes during render).
 *
 * The active deck's virtual position is exposed via the ``audioRef``
 * compatibility shim (a synthetic object with a ``currentTime`` field
 * updated on every tick) so the v2.5.3 ``<VisualLayer>`` keeps its
 * existing read pattern without API changes.
 */

import {
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
} from "react";
import { getToken } from "./auth";
// Next does not proxy WebSockets, so the browser dials the backend directly.
// Where that is lives in ONE place — see lib/ws-base.ts.
import { wsBase } from "./ws-base";
import { streamUrl } from "./api";
import {
  BufferCache,
  BufferDeck,
  SCHEDULE_LOOKAHEAD_SEC,
  isSkippableLoadFailure,
} from "./audio_buffer_decks";
import {
  computeCrossfadeWhen,
  idealLandingTime,
  lateCompensatedOffset,
  buildMeasurementProfile,
  applyPitchNudge,
} from "./crossfade_timing";
import type { Greeting } from "./greetings";

/**
 * v3.4 — synthetic "audio element" shim VisualLayer reads to drive its
 * beat clock. The previous substrate exposed the active HTMLAudioElement
 * directly; with BufferDeck we read the deck's virtual position once
 * per playback_pos tick and write it onto this object so VisualLayer's
 * existing ``audio.currentTime`` access pattern keeps working without
 * a prop-API change. ``duration`` mirrors the buffer's duration for the
 * UI progress bar; ``paused`` reflects whether a source is currently
 * scheduled.
 */
export interface VisualAudioShim {
  currentTime: number;
  duration: number;
  paused: boolean;
}

export type LiveEngineState = "idle" | "playing" | "crossfading" | "ended";

/**
 * v3.0 — equal-power crossfade curve generator.
 *
 * WebAudio's ``GainNode.setValueCurveAtTime`` takes a Float32Array
 * sampled across the duration; the API interpolates between adjacent
 * samples. 257 samples (a power-of-two plus one) is plenty for a
 * 12-second crossfade — far below the audible threshold for a
 * stepwise gain change at ~48 ms per sample.
 *
 * Curves match ``agent.phase_lock.phase_locked_crossfade_np`` so the
 * offline render, the terminal-live engine, and the browser-live engine
 * all sound the same for the same input. Two facts the algebra hinges on:
 *
 *   - cos(t·π/2)² + sin(t·π/2)² = 1 for all t — perceived power is
 *     constant across the overlap.
 *   - At t = 0: cos = 1 (full outgoing), sin = 0 (no incoming yet).
 *     At t = 1: cos = 0, sin = 1.
 *
 * The 64-sample raised-cosine edge guard in the numpy / sounddevice
 * paths is intentionally NOT applied here. Browsers route ``<audio>``
 * through MediaElementAudioSourceNode which already smooths sample-rate
 * conversion at the playback boundary, so the click the guard exists
 * to mask doesn't appear on this path.
 */
export function buildEqualPowerCurve(
  direction: "in" | "out",
  samples = 257,
): Float32Array {
  const curve = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    const angle = t * (Math.PI / 2);
    curve[i] = direction === "out" ? Math.cos(angle) : Math.sin(angle);
  }
  return curve;
}

export interface LiveTrackSummary {
  id: string;
  display_name: string;
  bpm?: number | null;
  camelot_key?: string | null;
  duration_sec?: number | null;
  /** Optional precomputed beatgrid — populated when the catalog has it.
   * Consumed by the v2.5.3 ``<VisualLayer>`` for sample-accurate beat sync.
   * Tracks without a beatgrid fall back to AnalyserNode onset detection. */
  beatgrid?: { bpm: number; first_beat_sec: number } | null;
}

export interface LiveCommandLogEntry {
  role: "user" | "assistant";
  text: string;
  ts: number;
}

export interface DjChatEntry {
  text: string;
  ts: number;
}

/**
 * v3.0.1 — phase-lock observability. Emitted by the backend when the
 * upcoming transition's planner couldn't land on a phrase boundary
 * (typically because one or both tracks lack a v2 beatgrid). The UI
 * shows these as non-blocking warnings so the DJ knows the next
 * transition will use the legacy linear-fade path AND knows which
 * track to regenerate beatgrids for.
 *
 * ``reason`` is the machine-readable enum the frontend can branch on
 * for tailored remediation copy. ``message`` is the human-readable
 * fallback the backend ships pre-localised; the UI can show it
 * verbatim instead of mapping ``reason`` if it prefers.
 */
export type CriticWarningReason =
  | "no_beatgrid_either_side"
  | "no_beatgrid_outgoing"
  | "no_beatgrid_incoming"
  | "no_phrase_anchor_in_window";

export interface CriticWarning {
  id: string;
  ts: number;
  kind: "phase_lock_fallback";
  reason: CriticWarningReason;
  message: string;
  outgoingTrack: { id: string | null; displayName: string | null };
  incomingTrack: { id: string | null; displayName: string | null };
}

export interface UseLiveSessionApi {
  state: LiveEngineState;
  connected: boolean;
  currentTrack: LiveTrackSummary | null;
  nextTrack: LiveTrackSummary | null;
  secondsToCrossfade: number;
  playlistRemaining: number;
  playlist: LiveTrackSummary[];
  /** 1-based position of the current track in the playlist (0 if unknown). */
  currentPosition: number;
  /** Elapsed seconds within the active deck's current track (throttled ~250ms). */
  currentTrackTime: number;
  /** Duration of the active deck's current track (best-effort, may be 0). */
  currentTrackDuration: number;
  log: LiveCommandLogEntry[];
  /**
   * Append-only feed of ``dj_chat`` messages emitted by the agent via the
   * v2.5.2 ``emit_chat`` tool. The UI surfaces these in the LiveStage
   * "DJ chat" panel so the audience sees rejection / acknowledgement text.
   */
  djChat: DjChatEntry[];
  /**
   * v3.0.1 — active phase-lock fallback warnings, oldest first. The UI
   * surfaces these as a non-blocking banner ("upcoming transition will
   * use a linear fade — regenerate beatgrid for Track X"). Capped at
   * the most recent 10 entries so a long broken-catalog session
   * doesn't accumulate dozens of stale warnings.
   */
  criticWarnings: CriticWarning[];
  /**
   * Drop a single warning from ``criticWarnings`` (typically wired to
   * the X button on the banner). Safe to call with an id that no
   * longer exists — no-op.
   */
  dismissCriticWarning: (id: string) => void;
  error: string | null;
  /** True when the browser blocked autoplay; the UI must surface a click-to-start. */
  autoplayBlocked: boolean;
  /**
   * Active deck — exposed as a stable ref for `<VisualLayer>`.
   *
   * v3.4 — type widened from HTMLAudioElement to VisualAudioShim |
   * HTMLAudioElement so the BufferDeck substrate (which has no DOM
   * element) can provide the same ``currentTime`` / ``duration`` /
   * ``paused`` shape VisualLayer reads. The shim is updated on every
   * playback_pos tick from the active BufferDeck's virtual position;
   * VisualLayer's beat clock therefore lands within one render
   * quantum (~2.7 ms @ 48 kHz) of the actual audio output, the same
   * precision its sample-accurate gain ramps get.
   */
  audioRef: React.RefObject<VisualAudioShim | HTMLAudioElement | null>;
  /** Send a control command to the backend agent. */
  sendCommand: (cmd: LiveCommand) => void;
  /** Send free-text user message to the agent. */
  sendUserMessage: (text: string) => void;
  /**
   * Low-level publish escape hatch used by v2.5.2 mic perception. Sends the
   * message verbatim — the WS handler dispatches by ``type``. Avoid in
   * application code; prefer ``sendCommand`` / ``sendUserMessage``.
   */
  sendRaw: (message: Record<string, unknown>) => void;
  /** Resume playback after autoplay was blocked — must be called from a user gesture. */
  resumePlayback: () => void;
  /** Quit the live session — closes the WS, stops audio, releases mic etc. */
  quit: () => void;
  // v2.6.0 — endless / improvisation mode (YouTube-streaming use case).
  /** Server-confirmed endless-mode flag. Mirrors session.context_variables.endless_mode. */
  endlessMode: boolean;
  /** True between PLAYLIST_RUNNING_LOW and the next track_started — UI surfaces a banner. */
  playlistRunningLow: boolean;
  /** Toggle endless mode. Sends a WS command; the server echoes the new state. */
  setEndlessMode: (enabled: boolean) => void;
  // v3.7.0 — chat greeting overlay feed.
  /** Recent chat_greeting events (capped at 20). GreetingOverlay consumes
   *  the tail; older entries age out naturally. */
  greetings: Greeting[];
  // v2.7 — YouTube Live Chat ingest status, surfaced as a pill in the /live header.
  /** Compact view-model the UI binds to. ``state === "off"`` means no events arrived yet
   *  (either the operator hasn't linked YT or the backend isn't configured); the UI uses
   *  that to suppress the pill on plain `/live` sessions. */
  youtube: {
    state: "off" | "connected" | "no_broadcast" | "quota_exceeded" | "disconnected" | "error";
    broadcastTitle?: string;
    reason?: string;
  };
  // v2.7.3 — WS reconnect telemetry for the /live banner.
  /** 0 when not retrying. 1..wsRetryMax while a backoff is scheduled or in flight. */
  wsRetryAttempt: number;
  /** Mirror of ``MAX_WS_RETRIES`` so the UI can render "N/MAX" without importing the const. */
  wsRetryMax: number;
  /** True after wsRetryMax non-4001 closes in a row — UI shows a Reconnect button. */
  wsExhausted: boolean;
  /** Manual "try again" trigger — clears exhausted flag and reopens the WS. */
  reconnectNow: () => void;
  /**
   * W3 (beatmatch feedback loop) — manual pitch-bend on the active deck.
   * "earlier" speeds the incoming track up momentarily (it entered late),
   * "later" slows it; the correction accumulates as the reinforcement signal
   * for the current transition's beatmatch_measurement.
   */
  nudgePitch: (direction: "earlier" | "later") => void;
}

export type LiveCommand =
  | { type: "skip" }
  | { type: "stay" }
  | { type: "more_energetic" }
  | { type: "wind_down" };

/**
 * v3.0 — phase-lock anchors carried alongside the crossfade trigger.
 *
 * When the catalog has v2 beatgrids on both sides of a transition, the
 * backend (``LiveEngineBrowser``) computes the chosen outgoing-anchor
 * downbeat, the incoming-anchor downbeat (with the pickup-skip RMS
 * heuristic applied), and the xfade duration in catalog seconds. The
 * frontend uses this to:
 *
 *   1. Seek the incoming ``<audio>`` element to ``incoming_anchor_sec``
 *      so the first sample played IS a downbeat — that's what makes
 *      the overlay-add phase-lock by construction.
 *   2. Replace its legacy linear ``GainNode`` ramp with equal-power
 *      cos/sin curves (the same algebra ``main.build_mix`` uses on the
 *      offline render path), so the perceived loudness through the
 *      overlap stays constant instead of dipping in the middle.
 *
 * The payload is empty (``{}``) when the catalog lacks v2 beatgrids
 * or the heuristics fell back — that's the frontend's signal to take
 * the legacy linear ramp.
 */
export interface PhaseLockPayload {
  outgoing_anchor_sec?: number;
  incoming_anchor_sec?: number;
  xfade_sec?: number;
  phrase_tier?: string;
  incoming_pickup_skipped?: boolean;
  edge_guard_samples?: number;
  sample_rate?: number;
  /**
   * v3.1 — tempo-match playback rate for the incoming deck. Applied as
   * ``HTMLMediaElement.playbackRate`` (with ``preservesPitch=true``) so the
   * incoming track's BPM matches the outgoing's during the crossfade,
   * mirroring the pyrubberband pre-stretch the CLI ``LiveEngineLocal``
   * runs. ``1.0`` (or undefined) means "no rate change" — either the BPM
   * delta is within ``DEFAULT_BPM_MATCH_THRESHOLD`` (~5 BPM, inaudible),
   * or one of the catalog entries lacks a usable BPM.
   *
   * Clamped to ``[1/1.5, 1.5]`` server-side so a corrupted catalog entry
   * can't produce a runaway rate. We keep the same value past the
   * crossfade window — the body of the incoming track stays at the
   * outgoing's BPM for parity with the CLI engine (no ramp-back).
   */
  incoming_rate?: number;
  /**
   * v3.1 — placeholder for a future meet-in-middle strategy on the
   * outgoing deck. Always ``1.0`` today.
   */
  outgoing_rate?: number;
  /**
   * v3.5 — feed-forward beat-lock grid-warp. A per-bar playback-rate
   * curve for the incoming deck that keeps every incoming downbeat on an
   * outgoing downbeat across the whole overlap — the software pitch-fader
   * ride that kills the residual "cabalgar" on tight 4/4 grids that the
   * single static ``incoming_rate`` couldn't (it corrects only the average
   * BPM, not per-bar micro-tempo or madmom estimation error).
   *
   * Each entry's ``at_sec`` is seconds after the shared ``when`` clock;
   * ``ramp`` selects setValueAtTime (stepped per-bar lock) vs
   * linearRampToValueAtTime (the release glide back to native rate).
   * Present ONLY for tight grids; absent for loose-grid genres
   * (jazz/soul/lofi), where the frontend keeps using ``incoming_rate``.
   */
  beat_rate_schedule?: { at_sec: number; rate: number; ramp: boolean }[];
  /**
   * v3.3 — which crossfade move the engine picked. ``smooth_blend`` is
   * the legacy equal-power overlay-add (no EQ touch); ``bass_swap``
   * adds a high-pass filter automation to the incoming deck, snapping
   * the cutoff back down on a phrase-boundary downbeat for a
   * tension/release "drop" feel. Always present when ``xfade_sec`` is
   * set so the deck can branch on it without optional chaining.
   *
   * v3.8 — ``drift`` is the beatless-genre profile (aural / ambient:
   * drones, healing frequencies, no valid beat structure). The backend
   * resolves a per-genre transition profile and emits ``drift`` when
   * EITHER endpoint track maps to a ``dj_mix=False`` profile. Drift is
   * one long equal-power crossfade at native rate and NOTHING else —
   * the frontend MUST ignore every phase-lock anchor / rate field when
   * this is set (see ``crossfadeToNext``). Unknown / absent values are
   * treated exactly as before (legacy path), so an older backend that
   * never sends ``drift`` is unaffected.
   */
  transition_style?: "smooth_blend" | "bass_swap" | "drift";
  /**
   * v3.3 — automation envelope for the ``bass_swap`` style. Present iff
   * ``transition_style === "bass_swap"``. The deck reads:
   *
   *   - ``hpf_cutoff_during_hz``: cutoff applied at xfade start.
   *   - ``hpf_cutoff_after_hz``: cutoff applied at the drop downbeat.
   *   - ``drop_at_incoming_sec``: catalog-time position in the incoming
   *     track where the cutoff snaps from "during" to "after". Convert
   *     to AudioContext time using the deck's currentTime offset.
   */
  bass_swap?: {
    hpf_cutoff_during_hz: number;
    hpf_cutoff_after_hz: number;
    drop_at_incoming_sec: number;
  };
}

interface ServerEngineCommand {
  type: "engine_command";
  command:
    | "load"
    | "skip"
    | "crossfade"
    | "queue_swap"
    | "stop"
    /** v2.5.0.1 — release the active deck so the next ``load`` starts cleanly. */
    | "stop_deck";
  track?: LiveTrackSummary;
  to_track?: LiveTrackSummary;
  from_track?: LiveTrackSummary;
  position?: number;
  crossfade_sec?: number;
  /** v3.0 — phase-lock anchors for the upcoming crossfade. See type above. */
  phase_lock?: PhaseLockPayload;
}

interface ServerLiveStateMessage {
  type: "live_state";
  data: {
    session_id: string;
    playlist: LiveTrackSummary[];
    engine_state: {
      state: LiveEngineState;
      position_sec: number;
      current_track: LiveTrackSummary | null;
      next_track: LiveTrackSummary | null;
      seconds_to_crossfade: number;
      playlist_remaining: number;
    };
  };
}

interface ServerEngineEvent {
  type:
    | "track_started"
    | "approaching_crossfade"
    | "crossfade_triggered"
    | "crossfade_finished"
    | "track_ended"
    | "session_ended"
    | "playlist_running_low"
    | "endless_warning";
  track?: LiveTrackSummary;
  next_track?: LiveTrackSummary;
  from_track?: LiveTrackSummary;
  to_track?: LiveTrackSummary;
  seconds_remaining?: number;
  /**
   * v2.5.2 — authoritative crossfade-trigger position (seconds within the
   * track). Carried by ``track_started`` and ``approaching_crossfade`` so
   * the frontend can derive a live-ticking countdown from the deck's
   * ``currentTime`` instead of freezing on the single-emit
   * ``seconds_remaining`` value. ``null`` / undefined when the engine
   * hasn't computed a target yet (e.g. last track, or duration unknown).
   */
  cf_point_sec?: number | null;
  /** v2.6.0 endless-mode warning payload (cap reached, no candidates). */
  reason?: string;
  message?: string;
  /**
   * v3.0 — phase-lock anchors. Carried by ``track_started`` so the
   * frontend can pre-position (or at least pre-render) the incoming
   * deck before the actual ``crossfade`` engine_command lands, and by
   * ``approaching_crossfade`` so any pre-fade UI (countdown, deck
   * preview) gets the same anchors the audio scheduler will use.
   */
  phase_lock?: PhaseLockPayload;
}

interface ServerEndlessModeMessage {
  type: "endless_mode";
  enabled: boolean;
}

// v2.7 — YouTube Live Chat status updates emitted by the backend. The
// state machine is intentionally small: the frontend only renders a
// pill colour + tooltip, so a flat string is enough.
interface ServerYouTubeStatusMessage {
  type: "youtube_status";
  state:
    | "connected"          // poller running, broadcast attached
    | "no_broadcast"       // creds OK but no active YT live event
    | "quota_exceeded"     // backend is backing off; no action needed
    | "disconnected"       // token revoked / broadcast ended
    | "error";
  broadcast?: { id: string; title: string };
  reason?: string;
}

interface ServerLiveMessage {
  type: "live_message";
  role: "user" | "assistant";
  content: string;
}

interface ServerDjChat {
  type: "dj_chat";
  text: string;
}

// v3.7.0 — greeting overlay trigger. Emitted by the backend once per
// chatter per stream (first message; author already sanitized
// server-side). ``kind: "returning"`` is reserved for channel-level
// regulars — the contract ships with the enum so that lands without a
// breaking change.
interface ServerChatGreeting {
  type: "chat_greeting";
  author: string;
  kind: "first" | "returning";
}

interface ServerError {
  type: "error";
  message: string;
}

// v3.0.1 — phase-lock fallback warning shape on the wire. Keys mirror
// ``LiveEngineBrowser._maybe_emit_critic_warning``'s payload exactly;
// the snake_case → camelCase mapping happens in the WS handler below
// so the rest of the React tree consumes a clean ``CriticWarning``.
interface ServerCriticWarning {
  type: "critic_warning";
  kind: "phase_lock_fallback";
  reason: CriticWarningReason;
  outgoing_track?: { id?: string; display_name?: string };
  incoming_track?: { id?: string; display_name?: string };
  phrase_tier?: string;
  message?: string;
}

type ServerEvent =
  | ServerLiveStateMessage
  | ServerEngineEvent
  | ServerEngineCommand
  | ServerLiveMessage
  | ServerDjChat
  | ServerChatGreeting
  | ServerEndlessModeMessage
  | ServerYouTubeStatusMessage
  | ServerCriticWarning
  | ServerError;

const COMMAND_TEXT: Record<LiveCommand["type"], string> = {
  skip: "skip",
  stay: "stay",
  more_energetic: "more energetic",
  wind_down: "wind down",
};

// v3.0.1 — upper bound on retained ``critic_warning`` entries. Anything
// older falls off the head of the array on insert. 10 is generous: a
// healthy catalog never emits any, and even a sloppy one rarely sees
// more than 2-3 active at once.
const CRITIC_WARNINGS_MAX = 10;

const PLAYBACK_POS_INTERVAL_MS = 250;

// v2.7.3 — WS auto-reconnect with exponential backoff. Mirrors the
// SSE retry policy in ``lib/render-stream.ts`` (5 attempts, capped
// backoff). The hook used to give up on the first close — on a
// uvicorn ``--reload`` cycle every /live tab silently went dead and
// needed a full refresh. Now the hook retries on any non-4001 close,
// resets state when a reopen succeeds, and surfaces an honest
// "exhausted" flag the UI can render a Refresh button against.
export const MAX_WS_RETRIES = 5;
const WS_BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 15_000] as const;

/**
 * v2.7.2 — options object accepted by ``useLiveSession``.
 *
 * ``viewer: true`` switches the hook into a read-only consumer of the
 * session's engine event bus (used by the OBS Browser Source embed
 * route). Viewer mode:
 *  - connects to ``/api/sessions/{id}/live/viewer`` instead of the
 *    primary ``/live/stream`` endpoint,
 *  - suppresses all outbound messages (``playback_pos``, ``user_msg``,
 *    ``perception``, ``set_endless_mode``, ``quit``) — the backend
 *    handler ignores anything a viewer sends anyway, but we save the
 *    network round-trips,
 *  - still plays audio locally so OBS captures it from the Browser
 *    Source, and still derives state from incoming events so the
 *    visualizer renders correctly.
 */
export interface UseLiveSessionOptions {
  viewer?: boolean;
}

/**
 * v3.9 — cap on consecutive skip-on-load-failure advances. Each
 * skippable load failure (decode error / decode timeout) sends a
 * synthetic ``track_ended`` so the backend moves past the unplayable
 * track; if several tracks in a row fail, the problem is the audio
 * pipeline (wedged decoder, OOM), not the tracks — stop burning
 * through the playlist and leave recovery to the server-side stall
 * watchdog. Reset whenever a source schedules successfully.
 */
const MAX_CONSECUTIVE_LOAD_SKIPS = 3;

export function useLiveSession(
  sessionId: string | null,
  options: UseLiveSessionOptions = {},
): UseLiveSessionApi {
  const viewerMode = options.viewer === true;
  // ``viewerMode`` is fixed for the lifetime of the hook (an embed
  // page never flips to operator), but stable closures (ensureBufferDeck,
  // setEndlessModeWS, …) are created once and need a ref to see it.
  const viewerModeRef = useRef(viewerMode);
  viewerModeRef.current = viewerMode;
  // ── Refs (audio + WS plumbing) ──────────────────────────────────────────
  const wsRef = useRef<WebSocket | null>(null);
  const activeDeckRef = useRef<"a" | "b">("a");
  const audioCtxRef = useRef<AudioContext | null>(null);
  // v3.4 — BufferDeck refs replace the prior `<audio>` + GainNode +
  // BiquadFilterNode trio per side. Each BufferDeck owns its own gain
  // and filter internally and exposes a sample-accurate
  // scheduleSource(buffer, when, offset, rate) primitive.
  const deckARef = useRef<BufferDeck | null>(null);
  const deckBRef = useRef<BufferDeck | null>(null);
  // Buffer cache: keyed by stream URL, holds decoded PCM. Filled by
  // preload during APPROACHING_CF + by direct load during a hard cut /
  // first track. Cleared on unmount.
  const bufferCacheRef = useRef<BufferCache | null>(null);
  // Compatibility shim for VisualLayer.tsx — exposes the active deck's
  // virtual position behind the same { currentTime } shape the prior
  // HTMLAudioElement ref provided. Updated every playback_pos tick.
  const audioRef = useRef<VisualAudioShim | null>(null);
  const currentTrackIdRef = useRef<string | null>(null);
  // W2/W3 (beatmatch feedback loop) — accumulates the human pitch-bend
  // correction (ms) for the CURRENT transition; rides on the next
  // beatmatch_measurement emit, then resets. W3 wires the buttons that fill it.
  const pitchBendMsRef = useRef<number>(0);
  // v3.9 — stream URL of the most recently preloaded next track, so
  // cache pruning can keep {current, next} and evict everything else.
  const lastPreloadedUrlRef = useRef<string | null>(null);
  // v3.9 — consecutive skip-on-load-failure counter (see
  // MAX_CONSECUTIVE_LOAD_SKIPS). Reset on every successful schedule.
  const loadFailureStreakRef = useRef(0);

  // ── State (derived from engine events) ──────────────────────────────────
  const [state, setState] = useState<LiveEngineState>("idle");
  const [connected, setConnected] = useState(false);
  const [currentTrack, setCurrentTrack] = useState<LiveTrackSummary | null>(null);
  const [explicitNextTrack, setExplicitNextTrack] = useState<LiveTrackSummary | null>(null);
  const [playlist, setPlaylist] = useState<LiveTrackSummary[]>([]);
  const [playlistRemaining, setPlaylistRemaining] = useState(0);
  const [log, setLog] = useState<LiveCommandLogEntry[]>([]);
  const [djChat, setDjChat] = useState<DjChatEntry[]>([]);
  // v3.0.1 — see ``CriticWarning`` type. Cap at 10 (CRITIC_WARNINGS_MAX
  // below). The dismissCriticWarning callback drops by id.
  const [criticWarnings, setCriticWarnings] = useState<CriticWarning[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);
  const [currentTrackTime, setCurrentTrackTime] = useState(0);
  const [currentTrackDuration, setCurrentTrackDuration] = useState(0);
  // v2.5.2 — authoritative crossfade-trigger time within the active deck's
  // track, populated by ``track_started`` / ``approaching_crossfade`` events
  // (the engine sends ``cf_point_sec``). The displayed countdown is
  // derived from this minus the live ``currentTrackTime`` so it ticks down
  // every 250 ms instead of freezing on a single emit. Falls back to a
  // legacy seconds_remaining value when the server hasn't sent
  // ``cf_point_sec`` yet (older backend).
  const [cfTargetSec, setCfTargetSec] = useState<number | null>(null);
  const [legacySecsToCf, setLegacySecsToCf] = useState<number | null>(null);
  // v2.6.0 endless mode mirror state. Defaults to false; server confirms
  // on every `set_endless_mode` write via an `endless_mode` echo event.
  const [endlessMode, setEndlessMode] = useState(false);
  // True from PLAYLIST_RUNNING_LOW until the next track_started — drives
  // the "picking continuation track…" banner on /live.
  const [playlistRunningLow, setPlaylistRunningLow] = useState(false);
  // v3.7.0 — chat greeting feed for the on-stream overlay. Server emits
  // one chat_greeting per chatter per stream; we keep a short tail and
  // let GreetingOverlay own display timing/coalescing. The seq ref
  // gives each entry a stable identity (burst events share one ts).
  const [greetings, setGreetings] = useState<Greeting[]>([]);
  const greetingSeqRef = useRef(0);
  // v2.7.3 — WS reconnect bookkeeping. ``wsRetryAttempt`` is the
  // current retry count (0 when connected or doing the very first
  // open). ``wsExhausted`` flips once we've burned through
  // ``MAX_WS_RETRIES`` without a successful reopen — at that point the
  // UI offers a manual Reconnect button. ``reconnectKey`` is bumped by
  // ``reconnectNow`` to force the WS effect to re-run from scratch.
  const [wsRetryAttempt, setWsRetryAttempt] = useState(0);
  const [wsExhausted, setWsExhausted] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
  // v2.7 — YouTube Live Chat status pill state. Defaults to "off" so
  // we don't render the pill at all on plain `/live` sessions where YT
  // isn't configured server-side.
  const [youtube, setYoutube] = useState<{
    state: "off" | "connected" | "no_broadcast" | "quota_exceeded" | "disconnected" | "error";
    broadcastTitle?: string;
    reason?: string;
  }>({ state: "off" });

  // v2.7.0 — HTTP probe so the pill renders immediately on /live mount
  // without waiting for a WS event. The live WS only emits
  // ``youtube_status`` for sessions that successfully open the live
  // channel, which can race with the page's first paint. Probing
  // ``/api/youtube/status`` directly gives a deterministic initial
  // state: 404 → feature disabled server-side (leave state "off");
  // 200 + connected:false → "disconnected" (pill renders, clicking
  // starts OAuth); 200 + connected:true → "connected".
  // Server-emitted WS events still win — they're the source of truth
  // for live broadcast attachment — but this seeds the initial render.
  useEffect(() => {
    if (typeof window === "undefined") return;
    let cancelled = false;
    void import("./api").then(({ getYouTubeStatus }) => {
      getYouTubeStatus()
        .then((status) => {
          if (cancelled) return;
          if ("connected" in status && status.connected) {
            setYoutube({
              state: "connected",
              broadcastTitle: status.channel_title,
            });
          } else if ("connected" in status && !status.connected) {
            setYoutube({ state: "disconnected" });
          }
        })
        .catch(() => {
          // Network / 404 — leave as "off" so the pill stays hidden.
        });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Derive next track from the playlist + current track position. This is
  // robust to event-ordering races where ``track_started`` fires before any
  // ``approaching_crossfade`` (so the explicit next_track payload is null).
  // Falls back to the explicit value carried by ``approaching_crossfade`` /
  // ``live_state`` when the current track isn't found in the playlist (e.g.
  // before ``live_state`` lands).
  const currentPosition = useMemo(() => {
    if (!currentTrack || playlist.length === 0) return 0;
    const idx = playlist.findIndex((t) => t.id === currentTrack.id);
    return idx >= 0 ? idx + 1 : 0;
  }, [currentTrack, playlist]);
  const nextTrack = useMemo<LiveTrackSummary | null>(() => {
    if (!currentTrack || playlist.length === 0) return explicitNextTrack;
    const idx = playlist.findIndex((t) => t.id === currentTrack.id);
    if (idx < 0) return explicitNextTrack;
    return idx + 1 < playlist.length ? playlist[idx + 1] : null;
  }, [currentTrack, playlist, explicitNextTrack]);

  // v2.5.2 — live-ticking countdown.
  //
  // Bug A1 (v2.5.1): the backend's ``approaching_crossfade`` event fires
  // exactly once when crossing the warn threshold, so the previous code
  // (which set ``secondsToCrossfade`` from the event payload alone) was
  // frozen at whatever value arrived first — typically 30 s, sometimes 0
  // when the threshold raced with the trigger. The fix derives the
  // displayed countdown from ``cfTargetSec - currentTrackTime`` so it
  // ticks down every 250 ms (matching the playback_pos interval).
  //
  // Fallback chain:
  //   1. Authoritative: ``cfTargetSec - currentTrackTime``.
  //   2. Legacy server snapshot (``seconds_to_crossfade`` from
  //      ``live_state`` or ``seconds_remaining`` from
  //      ``approaching_crossfade``) when an older backend doesn't yet
  //      send ``cf_point_sec``.
  //   3. Zero (no crossfade scheduled — last track or duration unknown).
  const secondsToCrossfade = useMemo<number>(() => {
    if (cfTargetSec !== null && cfTargetSec > 0) {
      return Math.max(0, cfTargetSec - currentTrackTime);
    }
    if (legacySecsToCf !== null) return Math.max(0, legacySecsToCf);
    return 0;
  }, [cfTargetSec, currentTrackTime, legacySecsToCf]);

  const appendLog = useCallback((entry: LiveCommandLogEntry) => {
    setLog((l) => [...l, entry]);
  }, []);

  // ── Audio plumbing helpers (declared first so the useEffectEvent
  //    wrappers below can reference them in module-source order — the v7
  //    react-hooks/immutability rule rejects forward references). ─────────

  const ensureAudioContext = useCallback(() => {
    if (audioCtxRef.current) return audioCtxRef.current;
    if (typeof window === "undefined") return null;
    const Ctor =
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext ?? window.AudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor();
    audioCtxRef.current = ctx;
    // Diagnostic — Chromium suspends the AudioContext silently when the
    // tab is backgrounded/idle. The HTMLAudioElement keeps advancing
    // currentTime (so the playback_pos pings + crossfade timer keep
    // ticking) but the gain nodes stop passing samples to the speakers
    // — exactly the "music gone, everything else fine" symptom. Log the
    // state changes and try to auto-resume so the user gets sound back
    // without having to refresh. Guarded for jsdom test envs whose
    // AudioContext mock omits ``addEventListener``.
    if (typeof ctx.addEventListener === "function") {
      ctx.addEventListener("statechange", () => {
        appendLog({
          role: "assistant",
          text: `[audioctx] state=${ctx.state}`,
          ts: Date.now(),
        });
        if (ctx.state === "suspended") {
          try {
            const p = ctx.resume();
            if (p && typeof p.then === "function") {
              p.then(
                () => {
                  appendLog({
                    role: "assistant",
                    text: "[audioctx] auto-resume ok",
                    ts: Date.now(),
                  });
                },
                (err) => {
                  appendLog({
                    role: "assistant",
                    text: `[audioctx] auto-resume failed: ${
                      (err as { name?: string })?.name ?? "Error"
                    }`,
                    ts: Date.now(),
                  });
                },
              );
            }
          } catch {
            /* ignore — best effort */
          }
        }
      });
    }
    return ctx;
  }, [appendLog]);

  /**
   * v3.4 — lazily create a BufferDeck for side ``which``. The deck owns
   * its own gain + filter chain wired into the AudioContext. Per the
   * spec, sources (AudioBufferSourceNode) are created on demand inside
   * the deck's scheduleSource() — they're single-use, so we never
   * cache a live source on the deck object itself.
   *
   * Returns null if the AudioContext can't be created (SSR / test
   * environment without Web Audio). All call sites must null-check.
   */
  const ensureBufferDeck = useCallback(
    (which: "a" | "b"): BufferDeck | null => {
      const refObj = which === "a" ? deckARef : deckBRef;
      if (refObj.current) return refObj.current;
      const ctx = ensureAudioContext();
      if (!ctx) return null;
      try {
        const deck = new BufferDeck(ctx, which === "a" ? 1 : 0);
        refObj.current = deck;
        return deck;
      } catch (err) {
        // Test environments / older AudioContext mocks may lack
        // createBiquadFilter or createBufferSource. Surface so the
        // diagnostic panel can flag the substrate failure rather
        // than silently going mute.
        appendLog({
          role: "assistant",
          text: `[deck ${which}] BufferDeck init failed: ${
            (err as { name?: string })?.name ?? "Error"
          }`,
          ts: Date.now(),
        });
        return null;
      }
    },
    [ensureAudioContext, appendLog],
  );

  /**
   * v3.4 — lazily create the BufferCache. Same audio context as the
   * decks so decoded PCM is at the right sample rate for sample-
   * accurate scheduling against the playback clock.
   */
  const ensureBufferCache = useCallback((): BufferCache | null => {
    if (bufferCacheRef.current) return bufferCacheRef.current;
    const ctx = ensureAudioContext();
    if (!ctx) return null;
    const cache = new BufferCache(ctx);
    bufferCacheRef.current = cache;
    return cache;
  }, [ensureAudioContext]);

  /**
   * Common natural-end-of-track handler — forwards a synthetic
   * track_ended WS message to the backend so its LiveEngineBrowser
   * can advance the cursor. Only fires when ``which`` is still the
   * active deck (the inactive deck's source may also end if the
   * outgoing track simply plays past its crossfade tail — that's
   * not a session-advancing event).
   */
  const onDeckEnded = useCallback((which: "a" | "b") => {
    if (activeDeckRef.current !== which) return;
    if (viewerModeRef.current) return;
    const ws = wsRef.current;
    const tid = currentTrackIdRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !tid) return;
    try {
      ws.send(JSON.stringify({ type: "track_ended", track_id: tid }));
    } catch {
      /* ignore — backend has the endgame safeguard as a fallback */
    }
  }, []);

  /**
   * v3.9 — bound the decode cache to the tracks still in play. Called
   * after every successful schedule with the NEW current track; keeps
   * that plus the most recently preloaded next track and evicts the
   * rest. Without this an endless session accumulates one ~50–100 MB
   * PCM buffer per played track until the renderer can't allocate and
   * every further decode fails (the 2026-08-01 30 h collapse).
   */
  const pruneBufferCache = useCallback((currentUrl: string) => {
    const cache = bufferCacheRef.current;
    if (!cache) return;
    const keep = [currentUrl];
    if (lastPreloadedUrlRef.current) keep.push(lastPreloadedUrlRef.current);
    cache.prune(keep);
  }, []);

  /**
   * v3.9 — skip-on-load-failure. When a track's buffer can't be
   * decoded (decode error or decode timeout — see
   * ``isSkippableLoadFailure``), waiting cannot fix it: pre-v3.9 the
   * deck simply went inert and the whole session hung until the
   * server-side stall watchdog noticed, minutes later, once per track.
   * Instead, send the backend the same synthetic ``track_ended`` a
   * natural end would, so it advances past the unplayable track right
   * away. Fetch-stage failures are NOT skipped (transient network, and
   * the E2E substrate's 404ing mock streams rely on staying inert),
   * and after MAX_CONSECUTIVE_LOAD_SKIPS consecutive skips we stop —
   * a streak that long means the audio pipeline itself is wedged.
   */
  const reportLoadFailure = useCallback((trackId: string, err: unknown) => {
    if (!isSkippableLoadFailure(err)) return;
    if (viewerModeRef.current) return;
    loadFailureStreakRef.current += 1;
    if (loadFailureStreakRef.current > MAX_CONSECUTIVE_LOAD_SKIPS) {
      console.warn(
        `[live] ${loadFailureStreakRef.current} consecutive load failures — ` +
          "audio pipeline looks wedged; not skipping further (stall watchdog takes over)",
      );
      return;
    }
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: "track_ended", track_id: trackId }));
    } catch {
      /* ignore — backend stall watchdog remains the backstop */
    }
  }, []);

  const loadIntoActiveDeck = useCallback(
    async (track: LiveTrackSummary) => {
      const which = activeDeckRef.current;
      const ctx = ensureAudioContext();
      const deck = ensureBufferDeck(which);
      const cache = ensureBufferCache();
      if (!ctx || !deck || !cache) return;

      // v3.4 — restore the deck to known state BEFORE scheduling the
      // new source. Gain to 1 (the new active track should be audible);
      // filter cutoff back to 20 Hz pass-through (clears any pending
      // bass_swap automation from a previous transition); kills the
      // currently-playing source if any. Equivalent to the prior
      // gain/filter/playbackRate reset block but expressed at the deck
      // abstraction layer.
      deck.stop();
      deck.resetAutomation(1);

      // Decode the track into PCM. This is the latency-bearing step
      // (~0.5–2 s on slow hardware for a 4-min MP3) so the call site
      // SHOULD have preloaded via cache.load() earlier. For a fresh
      // load (no preload) we incur the latency now — same UX as the
      // prior path (HTMLAudioElement also had to buffer before
      // play()).
      let buffer: AudioBuffer;
      try {
        buffer = await cache.load(streamUrl(track.id));
      } catch (err) {
        // v3.9 — decode failures/timeouts skip the track via a
        // synthetic track_ended (see reportLoadFailure); fetch-stage
        // failures still leave the deck INERT — the E2E substrate
        // drives the UI against 404ing mock streams and relies on
        // that, and stale-playlist 404s are prevented upstream (the
        // live WS validates the playlist against the catalog before
        // the engine starts).
        console.warn("[live] buffer load failed on load:", err);
        reportLoadFailure(track.id, err);
        return;
      }

      // Schedule with sample-accurate forward-lookahead — see
      // SCHEDULE_LOOKAHEAD_SEC doc. start() returns immediately; the
      // audio thread picks up the scheduled event on its next render
      // quantum.
      const when = ctx.currentTime + SCHEDULE_LOOKAHEAD_SEC;
      try {
        deck.scheduleSource(buffer, when, 0, 1.0, track.id, () =>
          onDeckEnded(which),
        );
        setAutoplayBlocked(false);
        // v3.9 — a healthy schedule ends any skip-on-failure streak.
        loadFailureStreakRef.current = 0;
      } catch (err) {
        // Most likely failure mode: AudioContext suspended (autoplay
        // policy). Surface the click-to-start overlay so the user
        // gesture can resume() the context.
        const name = (err as { name?: string })?.name ?? "";
        if (name === "InvalidStateError" || name === "NotAllowedError" || name === "") {
          console.warn("[live] schedule blocked on load:", err);
          setAutoplayBlocked(true);
        } else {
          console.warn("[live] scheduleSource failed (non-recoverable):", err);
        }
      }

      currentTrackIdRef.current = track.id;
      // v3.9 — this track is now the working set; drop stale PCM.
      pruneBufferCache(streamUrl(track.id));
      // Refresh the VisualLayer shim immediately so the UI sees the
      // new duration on the next render rather than waiting for the
      // 250 ms playback_pos tick.
      audioRef.current = {
        currentTime: deck.position(),
        duration: deck.duration(),
        paused: !deck.isPlaying(),
      };
    },
    [
      ensureAudioContext,
      ensureBufferDeck,
      ensureBufferCache,
      onDeckEnded,
      reportLoadFailure,
      pruneBufferCache,
    ],
  );

  const hardCutToTrack = useCallback(
    async (track: LiveTrackSummary) => {
      // For a hard cut we can simply load the new src into the active
      // deck — there's no blend to preserve.
      await loadIntoActiveDeck(track);
    },
    [loadIntoActiveDeck],
  );

  const crossfadeToNext = useCallback(
    async (
      track: LiveTrackSummary,
      crossfadeSec: number,
      phaseLock?: PhaseLockPayload,
      outgoingTrack?: LiveTrackSummary | null,
    ) => {
      // v3.4 — sample-accurate verify-then-commit crossfade:
      //   1. Decode the incoming track into a PCM AudioBuffer (likely
      //      already preloaded — see preloadIncoming() called from the
      //      APPROACHING_CF handler).
      //   2. Pick ONE future AudioContext time `when` to land everything:
      //      the new source's start(), the outgoing gain ramp, the
      //      incoming gain ramp, AND any bass_swap HPF automation.
      //      Because all four hit the same audio-thread clock at the
      //      same sample, the kicks of outgoing and incoming align by
      //      construction — eliminating the 10–50 ms "cabalgar" we saw
      //      with the prior HTMLAudioElement substrate (where MP3
      //      seek quantisation + play()-await wall-clock latency
      //      conspired to drift the two decks apart).
      //   3. Schedule, then atomically swap activeDeckRef.

      const fromWhich = activeDeckRef.current;
      const toWhich = fromWhich === "a" ? "b" : "a";
      const ctx = ensureAudioContext();
      const fromDeck = ensureBufferDeck(fromWhich);
      const toDeck = ensureBufferDeck(toWhich);
      const cache = ensureBufferCache();
      if (!ctx || !fromDeck || !toDeck || !cache) return;

      // v3.8 — drift transition profile (beatless genres: aural / ambient).
      // When EITHER endpoint maps to a dj_mix=False profile the backend
      // sends ``transition_style: "drift"`` and the profile's longer
      // crossfade_sec (24s) on the engine_command. Drift is one long
      // equal-power crossfade at NATIVE rate and nothing else — every DJ
      // move that reads as an artefact on frequency-domain content
      // (tempo-match wow/flutter + pitch shift, grid-warp built on
      // hallucinated beatgrids, bass-swap filter sweep heard as a dropout
      // on a pad) is bypassed. The three drift invariants are enforced
      // point-by-point below:
      //   1. anchors ignored     → incomingAnchorSec pinned to 0
      //   2. playbackRate 1.0    → incomingRate 1.0, no beat_rate_schedule
      //   3. no filter touch     → the whole HPF block is skipped
      // and the gain ramp uses equal-power curves over the RECEIVED
      // crossfade_sec (not phase_lock.xfade_sec). Absent / unknown
      // transition_style leaves isDrift false, so the legacy phase-lock /
      // bass_swap / linear paths run bit-for-bit as before.
      const isDrift = phaseLock?.transition_style === "drift";

      // v3.1 carried forward — tempo-match the incoming deck to the
      // outgoing's BPM via AudioBufferSourceNode.playbackRate
      // (sample-accurate equivalent of the prior HTMLMediaElement
      // playbackRate + preservesPitch). Backend supplies
      // incoming_rate = outgoing_bpm / incoming_bpm clamped to
      // [1/1.5, 1.5]; 1.0 / missing means leave at native rate. Drift
      // (invariant 2) forces native rate regardless of any incoming_rate
      // a mixed-profile payload might still carry.
      const incomingRate =
        !isDrift &&
        typeof phaseLock?.incoming_rate === "number" &&
        phaseLock.incoming_rate > 0
          ? phaseLock.incoming_rate
          : 1.0;

      // v3.5 — feed-forward beat-lock grid-warp. When the backend produced
      // a per-bar rate schedule (tight 4/4 grids), it supersedes the single
      // static incomingRate: we start the source at the FIRST segment's
      // rate (so the very first bar is already locked) and apply the rest
      // as playbackRate automation below. Absent for loose grids, where we
      // keep the static incomingRate. Drift (invariant 2) ignores any
      // schedule outright — a beatless pad has no bars to lock to.
      const rateSchedule = phaseLock?.beat_rate_schedule;
      const hasRateSchedule =
        !isDrift && Array.isArray(rateSchedule) && rateSchedule.length > 0;
      const initialRate = hasRateSchedule ? rateSchedule[0].rate : incomingRate;

      // Pre-stretched anchor: the catalog-time second within the
      // incoming track where the engine wants the crossfade to begin.
      // Becomes the `offset` argument to AudioBufferSourceNode.start —
      // sample-accurate by spec, unaffected by MP3 frame boundaries
      // because the buffer is plain PCM at the context's sample rate.
      // Drift (invariant 1) ignores the anchor entirely: no downbeat to
      // land on, so the incoming track simply starts from its head.
      const incomingAnchorSec =
        !isDrift && typeof phaseLock?.incoming_anchor_sec === "number"
          ? phaseLock.incoming_anchor_sec
          : 0;

      // Latency-bearing step (~0.5–2 s on slow hardware). If the
      // preload hit, this resolves synchronously from cache; if it
      // missed (e.g. crossfade fired faster than expected), we wait
      // here. Either way no audio drift accumulates because everything
      // downstream is scheduled against ctx.currentTime AFTER this
      // await resolves.
      let buffer: AudioBuffer;
      try {
        buffer = await cache.load(streamUrl(track.id));
      } catch (err) {
        // v3.9 — same skip semantics as loadIntoActiveDeck: an
        // undecodable incoming track can't be crossfaded into, ever;
        // tell the backend so it advances past it instead of letting
        // the outgoing deck run dry into a wedge.
        console.warn("[live] buffer load failed on crossfade:", err);
        reportLoadFailure(track.id, err);
        return;
      }

      // The one true reference time. Every audio-thread event we
      // schedule below references THIS value, so they all hit the
      // same sample. SCHEDULE_LOOKAHEAD_SEC is the spec-recommended
      // slack so the render thread has at least one quantum to pick
      // up the scheduled events.
      //
      // v3.6 bug 3 — align `when` to the OUTGOING deck's anchor downbeat
      // (backend `outgoing_anchor_sec`) instead of an arbitrary clock
      // instant. The incoming source starts at `when` on its own downbeat
      // (the offset arg), so landing `when` on the outgoing downbeat puts
      // the two in phase by construction. Previously we used a bare
      // `currentTime + lookahead`, pinning the incoming downbeat to a random
      // point in the outgoing bar → off from the first kick even with
      // grid-warp active (grid-warp fixes tempo slope, not phase intercept).
      const _xfTiming = computeCrossfadeWhen({
        ctxNow: ctx.currentTime,
        lookaheadSec: SCHEDULE_LOOKAHEAD_SEC,
        outgoingPosSec: fromDeck.position(),
        outgoingAnchorSec: phaseLock?.outgoing_anchor_sec,
      });
      const when = _xfTiming.when;
      // v3.6-diag — measure the residual few-ms late offset reported by ear.
      // If `clamped` is true the ideal downbeat landing was in the PAST and we
      // bumped to ctxNow+lookahead (missed-downbeat → constant late offset);
      // if false, any residual is more likely output latency. outputLatency /
      // baseLatency are the AudioContext's hardware/render delay (uncompensated
      // today). Grep the browser console for [xfade-timing].
      try {
        const ac = ctx as AudioContext & {
          outputLatency?: number;
          baseLatency?: number;
        };
        console.log(
          "[xfade-timing]",
          JSON.stringify({
            secondsUntilDownbeat: _xfTiming.secondsUntilDownbeat,
            clamped: _xfTiming.clamped,
            lookaheadSec: SCHEDULE_LOOKAHEAD_SEC,
            outputLatency: ac.outputLatency ?? null,
            baseLatency: ac.baseLatency ?? null,
          }),
        );
      } catch {
        /* console unavailable — ignore */
      }

      // v3.10.2 — land in phase even when we missed the downbeat.
      //
      // `when` above was clamped forward whenever the outgoing deck was
      // already past its anchor, which the live logs say is virtually
      // ALWAYS: 362 of 365 crossfades clamped, spread uniformly across
      // 0-0.3 s. That spread is the signature of the backend's ~4 Hz
      // position polling (it cannot notice the crossfade point sooner
      // than its next ping), not of jitter. We were then starting the
      // incoming source at the very top of its bar while the outgoing
      // was already `residualMs` into its own — a few tens of ms out of
      // phase on every transition. At 70 BPM that is up to a third of a
      // beat: an audible flam, and the residual the W2 loop has been
      // dutifully measuring and reporting all along without anything
      // ever acting on it.
      //
      // Skipping the same distance into the incoming track costs nothing
      // and puts both decks equally far into their bars. The alternative
      // — waiting for the NEXT downbeat — would delay every blend by a
      // whole bar and shift its position within the phrase.
      const startOffsetSec = lateCompensatedOffset(
        incomingAnchorSec,
        _xfTiming.residualMs,
        initialRate,
      );
      // Every plan the backend built is measured from the DOWNBEAT, not
      // from whenever we got around to scheduling, so the grid-warp
      // schedule and the bass-swap drop below anchor here.
      const downbeatTime = idealLandingTime(when, _xfTiming.residualMs);

      // W2 (beatmatch feedback loop) — stream this transition's measured
      // residual + (optional) human pitch-bend to the backend, which appends
      // it to measurements.jsonl for the beatmatch-learn loop. The profile
      // mirrors the backend's (key_pair, bpm_bucket) indexing. Best-effort:
      // a failed send must never disrupt the crossfade.
      try {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          const { profile, keyPair, bpmBucket } = buildMeasurementProfile(
            { bpm: outgoingTrack?.bpm, camelot_key: outgoingTrack?.camelot_key },
            { camelot_key: track.camelot_key },
          );
          ws.send(
            JSON.stringify({
              type: "beatmatch_measurement",
              profile,
              key_pair: keyPair,
              bpm_bucket: bpmBucket,
              offset_ms: _xfTiming.residualMs,
              pitch_bend_ms: pitchBendMsRef.current,
            }),
          );
        }
      } catch {
        /* WS send failed — drop this measurement, keep mixing */
      }
      // Reset the pitch-bend accumulator for the next transition (W3 fills it).
      pitchBendMsRef.current = 0;

      // Schedule the incoming source. Returns the same `when` (no
      // surprise — but explicit for readability when the rest of the
      // automation chains off it).
      try {
        toDeck.scheduleSource(
          buffer,
          when,
          startOffsetSec,
          initialRate,
          track.id,
          () => onDeckEnded(toWhich),
        );
        if (hasRateSchedule) {
          // Per-bar pitch-fader ride against the same `when` clock as the
          // source start, so every incoming downbeat lands on an outgoing
          // downbeat for the whole overlap.
          // Anchored at the downbeat, not at `when`: segment times are
          // bar boundaries of the incoming track, and we started
          // `residualMs` past its first one. A segment that now falls in
          // the past applies immediately, which is what "this bar has
          // already begun" means.
          toDeck.applyRateSchedule(rateSchedule, downbeatTime);
        }
        setAutoplayBlocked(false);
        // v3.9 — a healthy schedule ends any skip-on-failure streak.
        loadFailureStreakRef.current = 0;
      } catch (err) {
        const name = (err as { name?: string })?.name ?? "";
        if (name === "InvalidStateError" || name === "NotAllowedError" || name === "") {
          console.warn("[live] schedule blocked on crossfade:", err);
          setAutoplayBlocked(true);
        } else {
          console.warn("[live] scheduleSource failed on crossfade:", err);
        }
        return;
      }

      // v3.3 carried forward — bass_swap HPF automation on the
      // incoming deck's filter. Drop time is computed against the
      // SAME `when` as the source start, scaled by incomingRate
      // (because the deck plays back at that rate so the catalog-time
      // drop maps to wall-clock-time `dropOffset / rate`).
      // SMOOTH_BLEND skips the automation; the deck.resetAutomation()
      // below ensures the filter is at 20 Hz pass-through.
      const filterTo = toDeck.filter;
      if (isDrift) {
        // v3.8 drift (invariant 3) — ZERO BiquadFilter automation. A
        // filter sweep reads as an audio dropout on a beatless pad, so we
        // touch neither frequency.setValueAtTime nor cancelScheduledValues
        // here. Aural transitions are always drift, so no prior bass_swap
        // can have left a pending scheduled value on this deck to clear;
        // the filter stays at its constructor default (20 Hz pass-through).
      } else if (
        filterTo &&
        phaseLock?.transition_style === "bass_swap" &&
        phaseLock.bass_swap
      ) {
        const dropOffsetCatalogSec =
          // From where we ACTUALLY started, not from the anchor: we
          // skipped `residualMs` of the incoming, so that much less of
          // the track remains before the drop.
          phaseLock.bass_swap.drop_at_incoming_sec - startOffsetSec;
        const dropDelaySec = Math.max(
          0,
          dropOffsetCatalogSec / Math.max(0.0001, incomingRate),
        );
        try {
          filterTo.frequency.cancelScheduledValues(when);
          filterTo.frequency.setValueAtTime(
            phaseLock.bass_swap.hpf_cutoff_during_hz,
            when,
          );
          filterTo.frequency.setValueAtTime(
            phaseLock.bass_swap.hpf_cutoff_after_hz,
            when + dropDelaySec,
          );
        } catch {
          /* BiquadFilter scheduling unavailable — degrade silently */
        }
      } else if (filterTo) {
        // SMOOTH_BLEND or no payload — make sure the filter is in its
        // pass-through state in case a previous bass_swap left a
        // pending scheduled value on this deck.
        try {
          filterTo.frequency.cancelScheduledValues(when);
          filterTo.frequency.setValueAtTime(20, when);
        } catch {
          /* ignore */
        }
      }

      // Gain ramps — scheduled at the SAME `when` as the source so
      // the fade-in starts on the FIRST audible sample of the
      // incoming track. This is the core fix for cabalgar: prior
      // substrate scheduled gain at audioCtx.currentTime *after*
      // await play() returned, which was already a handful of ms
      // past the actual first sample.
      const gainFrom = fromDeck.gain;
      const gainTo = toDeck.gain;
      try {
        gainFrom.gain.cancelScheduledValues(when);
        gainTo.gain.cancelScheduledValues(when);
        gainFrom.gain.setValueAtTime(gainFrom.gain.value, when);
        gainTo.gain.setValueAtTime(gainTo.gain.value, when);
        // Drift ramps over the RECEIVED crossfade_sec (the profile
        // override, 24s for aural) — it carries no phase_lock.xfade_sec of
        // its own. Non-drift keeps the phase-lock xfade window when set.
        const xfadeSec = isDrift ? crossfadeSec : phaseLock?.xfade_sec ?? crossfadeSec;
        if (isDrift || (phaseLock && phaseLock.xfade_sec && phaseLock.xfade_sec > 0)) {
          // v3.0 — equal-power cos/sin curves (cos² + sin² = 1) so
          // perceived loudness stays constant across the overlap.
          // Algebra matches agent.phase_lock.phase_locked_crossfade_np.
          // Drift uses the SAME equal-power curves (invariant: one long
          // equal-power crossfade) — just over the profile's crossfade_sec.
          const fadeOut = buildEqualPowerCurve("out");
          const fadeIn = buildEqualPowerCurve("in");
          gainFrom.gain.setValueCurveAtTime(fadeOut, when, xfadeSec);
          gainTo.gain.setValueCurveAtTime(fadeIn, when, xfadeSec);
        } else {
          gainFrom.gain.linearRampToValueAtTime(0, when + xfadeSec);
          gainTo.gain.linearRampToValueAtTime(1, when + xfadeSec);
        }
      } catch {
        /* AudioParam mock without setValueCurveAtTime — fall back
           silently; the source still plays, the fade just won't be
           equal-power. */
      }

      activeDeckRef.current = toWhich;
      currentTrackIdRef.current = track.id;
      // v3.9 — the incoming track is now the working set; drop stale
      // PCM. The outgoing deck's source keeps its own buffer reference
      // until it plays out, so evicting mid-crossfade is inaudible.
      pruneBufferCache(streamUrl(track.id));
      // Refresh the VisualLayer shim — duration/paused are now from
      // the new deck. currentTime updates on the playback_pos tick.
      audioRef.current = {
        currentTime: toDeck.position(),
        duration: toDeck.duration(),
        paused: !toDeck.isPlaying(),
      };
    },
    [
      ensureAudioContext,
      ensureBufferDeck,
      ensureBufferCache,
      onDeckEnded,
      reportLoadFailure,
      pruneBufferCache,
    ],
  );

  /**
   * v3.4 — pre-decode the incoming track ahead of the crossfade so the
   * actual transition scheduling is synchronous. Called from the
   * APPROACHING_CF WS event handler (~30 s lookahead). Safe to call
   * multiple times for the same trackId — the BufferCache
   * de-duplicates concurrent loads. Quiet on failure; the crossfade
   * path will just incur the decode latency synchronously if this
   * never ran.
   */
  const preloadIncoming = useCallback(
    async (track: LiveTrackSummary) => {
      const cache = ensureBufferCache();
      if (!cache) return;
      // v3.9 — record the URL FIRST so a prune racing this preload
      // (e.g. a track advance landing mid-decode) keeps it.
      lastPreloadedUrlRef.current = streamUrl(track.id);
      try {
        await cache.load(streamUrl(track.id));
      } catch {
        /* swallow — next attempt or the crossfade path will retry */
      }
    },
    [ensureBufferCache],
  );

  const stopAllDecks = useCallback(() => {
    // v3.4 — stop() is the BufferDeck equivalent of pause + removeAttribute
    // + load(). It detaches the current source node, clears its onended
    // handler (so we don't fire a synthetic track_ended for a deliberate
    // teardown), and zeroes the deck's internal track state.
    for (const ref of [deckARef, deckBRef]) {
      const deck = ref.current;
      if (deck) {
        try {
          deck.stop();
        } catch {
          /* ignore */
        }
      }
    }
    audioRef.current = null;
    currentTrackIdRef.current = null;
  }, []);

  // ── useEffectEvent wrappers ─────────────────────────────────────────────
  // These read the latest `setState` / helper closures every render but
  // their identity stays stable from the WS effect's perspective, so the
  // effect doesn't re-run on each parent render.

  const handleEngineCommand = useEffectEvent((evt: ServerEngineCommand) => {
    const { command } = evt;
    if (command === "load" && evt.track) {
      void loadIntoActiveDeck(evt.track);
    } else if (command === "skip" && evt.track) {
      void hardCutToTrack(evt.track);
    } else if (command === "crossfade" && evt.to_track) {
      void crossfadeToNext(
        evt.to_track,
        evt.crossfade_sec ?? 12,
        evt.phase_lock,
        evt.from_track ?? currentTrack,
      );
    } else if (command === "stop_deck") {
      // v3.4 — release just the active deck's source so the next ``load``
      // plays into a fresh BufferSource and the previous track's
      // onended fallback can't re-fire (BufferDeck.stop() clears the
      // onended handler before stop()ping the source).
      const which = activeDeckRef.current;
      const deck = which === "a" ? deckARef.current : deckBRef.current;
      if (deck) {
        try {
          deck.stop();
        } catch {
          /* ignore */
        }
      }
    } else if (command === "stop") {
      stopAllDecks();
    }
    // ``queue_swap`` is purely UI metadata in v2.5.1 — the next
    // ``track_started`` will pick up the new track when the engine gets
    // there. No deck action needed now.
  });

  const onServerEvent = useEffectEvent((evt: ServerEvent) => {
    switch (evt.type) {
      case "live_state": {
        setPlaylist(evt.data.playlist || []);
        const es = evt.data.engine_state;
        setState(es.state);
        if (es.current_track) setCurrentTrack(es.current_track);
        if (es.next_track) setExplicitNextTrack(es.next_track);
        // Initial hint — until a ``track_started`` (or
        // ``approaching_crossfade``) lands with ``cf_point_sec``, we use
        // the engine's snapshot value. Once the engine sends an
        // authoritative cf target, the live-ticking derivation takes over.
        setLegacySecsToCf(es.seconds_to_crossfade);
        setPlaylistRemaining(es.playlist_remaining);
        break;
      }
      case "track_started": {
        if (evt.track) {
          setCurrentTrack(evt.track);
          currentTrackIdRef.current = evt.track.id;
          setState("playing");
          setCurrentTrackTime(0);
          setCurrentTrackDuration(0);
          // v2.5.2 — engine emits the authoritative crossfade-trigger
          // time so the countdown can tick live from the deck's
          // ``currentTime``. ``cf_point_sec`` may be null when the engine
          // is on the final track (no successor — no crossfade).
          if (typeof evt.cf_point_sec === "number") {
            setCfTargetSec(evt.cf_point_sec);
          } else {
            setCfTargetSec(null);
          }
          // Reset legacy fallback so a stale seconds_remaining from the
          // previous track doesn't bleed into the new one's countdown.
          setLegacySecsToCf(null);
          // v2.6.0 — clear the endless-mode "running low" banner once
          // we're on the new (appended) track. Engine will re-fire
          // playlist_running_low when this one approaches its own CF.
          setPlaylistRunningLow(false);
        }
        break;
      }
      case "approaching_crossfade": {
        if (evt.next_track) {
          setExplicitNextTrack(evt.next_track);
          // v3.4 — kick off decode of the incoming track now (~30 s
          // before the cf hits). BufferCache de-dupes if this races
          // with the crossfade itself. Decode (~0.5-2 s for a typical
          // MP3) is async fire-and-forget here so we never block the
          // UI event handler on it.
          void preloadIncoming(evt.next_track);
        }
        if (typeof evt.cf_point_sec === "number") {
          setCfTargetSec(evt.cf_point_sec);
        }
        if (typeof evt.seconds_remaining === "number") {
          setLegacySecsToCf(evt.seconds_remaining);
        }
        break;
      }
      case "crossfade_triggered":
        setState("crossfading");
        break;
      case "crossfade_finished":
        setState("playing");
        if (evt.to_track) {
          setCurrentTrack(evt.to_track);
          currentTrackIdRef.current = evt.to_track.id;
          setCurrentTrackTime(0);
          setCurrentTrackDuration(0);
          // ``track_started`` (which carries the new ``cf_point_sec``)
          // follows this event, so just clear the previous target.
          setCfTargetSec(null);
          setLegacySecsToCf(null);
        }
        break;
      case "track_ended":
        // No state change here — track_started for the new track follows.
        break;
      case "playlist_running_low":
        // v2.6.0 endless mode — surfaces the banner while the agent /
        // engine decides on a continuation track. Cleared on the next
        // track_started above (where setPlaylistRunningLow(false) is
        // wired into the existing handler).
        setPlaylistRunningLow(true);
        break;
      case "endless_warning":
        // Non-fatal; the engine has either hit the append cap or run
        // out of in-genre candidates. Surface to the agent via toast
        // semantics — the page wraps this into its UI banner.
        setError(evt.message || "Endless mode: no more candidates.");
        break;
      case "endless_mode":
        // Server-confirmed echo from set_endless_mode. We mirror the
        // boolean exactly so the toggle pill never gets out of sync
        // with the actual engine state.
        setEndlessMode(evt.enabled);
        if (!evt.enabled) setPlaylistRunningLow(false);
        break;
      case "youtube_status":
        // v2.7 — YouTube Live Chat poller state. The server emits one
        // of these on connect (after broadcast discovery) and again on
        // quota_exceeded / disconnected / error. We mirror straight
        // into local state so the pill renders without a refresh.
        setYoutube({
          state: evt.state,
          broadcastTitle: evt.broadcast?.title,
          reason: evt.reason,
        });
        break;
      case "chat_greeting": {
        // v3.7.0 — greeting overlay feed. Author is sanitized
        // server-side; we just append (short tail — GreetingOverlay
        // consumes and paces the display). The id is captured NOW —
        // the state updater runs later, and a same-batch burst would
        // otherwise read the ref after every increment (duplicate ids).
        const greetingId = ++greetingSeqRef.current;
        const greetingAuthor = evt.author;
        const greetingKind = evt.kind;
        setGreetings((prev) => [
          ...prev.slice(-19),
          {
            id: greetingId,
            author: greetingAuthor,
            kind: greetingKind,
            ts: Date.now(),
          },
        ]);
        break;
      }
      case "session_ended":
        setState("ended");
        setExplicitNextTrack(null);
        setCfTargetSec(null);
        setLegacySecsToCf(null);
        // v3.4 — release the buffer sources on both decks so we don't
        // leak ~80 MB of PCM each and the user can navigate away
        // cleanly. The deck wrappers stay (they only own gain +
        // filter, both reusable for the next session).
        for (const ref of [deckARef, deckBRef]) {
          const deck = ref.current;
          if (deck) {
            try {
              deck.stop();
            } catch {
              /* ignore */
            }
          }
        }
        // Drop the decode cache too — a session_ended means no further
        // playback in this session and the buffers could be a sizable
        // chunk of RAM.
        if (bufferCacheRef.current) {
          try {
            bufferCacheRef.current.clear();
          } catch {
            /* ignore */
          }
        }
        break;
      case "engine_command":
        handleEngineCommand(evt);
        break;
      case "live_message":
        appendLog({ role: evt.role, text: evt.content, ts: Date.now() });
        break;
      case "dj_chat":
        // Cap the feed at the most recent 200 entries so a long
        // broadcast with active YT chat doesn't accumulate thousands
        // of stale messages in memory (and force React to reconcile
        // them every render). 200 is generous: the audience /
        // immersive overlay shows the last 4, the booth panel shows
        // the last few — anything older is already off-screen.
        setDjChat((prev) => {
          const next = [...prev, { text: evt.text || "", ts: Date.now() }];
          return next.length > 200 ? next.slice(-200) : next;
        });
        break;
      case "critic_warning": {
        // v3.0.1 — phase-lock fallback notice. The backend already
        // debounces per transition pair (one emit per (cur, next)
        // index pair) so we don't need to dedup here, just append.
        // Reshape the snake_case wire payload into the camelCase
        // ``CriticWarning`` shape the rest of the React tree consumes.
        const ts = Date.now();
        const warning: CriticWarning = {
          id: `cw-${ts}-${Math.random().toString(36).slice(2, 8)}`,
          ts,
          kind: evt.kind,
          reason: evt.reason,
          message: evt.message || evt.reason,
          outgoingTrack: {
            id: evt.outgoing_track?.id ?? null,
            displayName: evt.outgoing_track?.display_name ?? null,
          },
          incomingTrack: {
            id: evt.incoming_track?.id ?? null,
            displayName: evt.incoming_track?.display_name ?? null,
          },
        };
        setCriticWarnings((prev) => {
          const next = [...prev, warning];
          return next.length > CRITIC_WARNINGS_MAX
            ? next.slice(-CRITIC_WARNINGS_MAX)
            : next;
        });
        break;
      }
      case "error":
        setError(evt.message || "Live session error");
        break;
      default:
        break;
    }
  });

  const onConnected = useEffectEvent(() => {
    setConnected(true);
    setError(null);
    // v2.7.3 — reaching open clears any retry bookkeeping. A flapping
    // backend (open / close / open / close) therefore resets the
    // attempt counter on each successful handshake, which is the
    // correct behaviour: the user should see "Reconnecting (1/5)" if
    // it later drops again, not "(3/5)" from a previous outage.
    setWsRetryAttempt(0);
    setWsExhausted(false);
    publishLiveActive(true);
  });
  const onClosed = useEffectEvent(() => {
    setConnected(false);
    publishLiveActive(false);
  });
  const onErrorCallback = useEffectEvent((msg: string) => {
    setError(msg);
  });

  // ── WebSocket lifecycle ─────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return;
    const token = getToken();
    if (!token) return;

    // v2.6.0 canonical path for the primary; v2.7.2 read-only viewer
    // path for OBS Browser Source / embed. The backend keeps
    // ``/ws/live/{id}`` as a deprecated alias for the primary; once
    // every page hits the new endpoint that alias can come out.
    const wsPath = viewerMode
      ? `/api/sessions/${sessionId}/live/viewer`
      : `/api/sessions/${sessionId}/live/stream`;

    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let currentWs: WebSocket | null = null;
    // v2.7.4 — track whether we've already burned our single allowed
    // 4001-reclaim attempt for this effect mount. If a fresh WS open
    // STILL gets displaced, the displacing primary is genuinely
    // alive on the other side and ping-ponging would kick a real
    // user off. Reset by the next effect mount (HMR / reconnectNow).
    let displacedReclaimUsed = false;

    // v2.7.3 — connect / retry loop. Wrapped so the close handler can
    // re-arm itself with backoff. ``attempt`` starts at 0 (the very
    // first open), increments to 1..MAX_WS_RETRIES on each
    // post-failure retry. Successful open resets the counter via
    // ``onConnected``.
    const connect = (attempt: number) => {
      if (cancelled) return;
      const ws = new WebSocket(`${wsBase()}${wsPath}?token=${token}`);
      currentWs = ws;
      wsRef.current = ws;
      let opened = false;

      ws.onopen = () => {
        opened = true;
        if (cancelled) {
          ws.close();
          return;
        }
        onConnected();
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data) as ServerEvent;
          onServerEvent(data);
        } catch {
          // ignore malformed frames
        }
      };

      ws.onerror = () => {
        if (opened && !cancelled) onErrorCallback("WebSocket error");
      };

      ws.onclose = (event) => {
        // Effect cleanup already ran (component unmounted or sessionId
        // changed). Don't schedule a retry — the next mount will own
        // its own connect loop.
        if (cancelled) {
          onClosed();
          return;
        }
        // v2.7.2 — close code 4001 is the backend's "displaced by a
        // newer primary on the same session" signal (see
        // ``ws_manager.displace_existing``). v2.7.4 — instead of
        // surfacing the error immediately, allow ONE delayed reclaim
        // attempt: the displacing tab may have closed in the
        // meantime (common for OBS Browser Sources the operator
        // can't manually refresh — a transient HMR remount used to
        // strand the tab on a permanent 4001 banner). If the
        // reclaim ALSO gets displaced, we give up — clearly there's
        // a genuinely-active other primary and looping would kick
        // it off.
        if (event && event.code === 4001) {
          onClosed();
          if (displacedReclaimUsed) {
            onErrorCallback(
              "Live session moved to another window. Refresh this tab to take it back.",
            );
            setWsRetryAttempt(0);
            return;
          }
          displacedReclaimUsed = true;
          // Longer delay than the standard backoff (the displacing
          // primary needs time to settle before we contend; if we
          // race-reconnect immediately we both ping-pong each
          // other). Banner shows "Reconnecting (1/5)…" via the
          // shared retry-attempt state.
          const delay = 30_000;
          setWsRetryAttempt(1);
          retryTimer = setTimeout(() => {
            retryTimer = null;
            connect(attempt);
          }, delay);
          return;
        }
        // Any other close (backend reload, network blip, idle
        // timeout) is retryable. Schedule the next attempt with
        // capped exponential backoff. After ``MAX_WS_RETRIES``
        // consecutive failures we give up and let the UI prompt the
        // user to retry manually.
        onClosed();
        const next = attempt + 1;
        if (next > MAX_WS_RETRIES) {
          setWsExhausted(true);
          setWsRetryAttempt(0);
          return;
        }
        const delay = WS_BACKOFF_MS[
          Math.min(attempt, WS_BACKOFF_MS.length - 1)
        ];
        setWsRetryAttempt(next);
        retryTimer = setTimeout(() => {
          retryTimer = null;
          connect(next);
        }, delay);
      };
    };

    // Effect-entry reset — clears any "exhausted" state from a prior
    // mount of this hook so the manual ``reconnectNow`` (which bumps
    // ``reconnectKey``) starts from a clean slate.
    setWsExhausted(false);
    setWsRetryAttempt(0);
    connect(0);

    return () => {
      cancelled = true;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      wsRef.current = null;
      const ws = currentWs;
      if (ws) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        } else if (ws.readyState === WebSocket.CONNECTING) {
          ws.addEventListener("open", () => ws.close(), { once: true });
        }
      }
      // Intentionally NOT closing the AudioContext or stopping the decks
      // here. Next.js Fast Refresh re-runs this effect every time the
      // hook source is edited, and the previous destructive cleanup
      // tore down audio mid-session — the user heard the music cut and
      // the track restart from 0. The audio cleanup now lives in a
      // dedicated unmount-only effect below; the browser GCs the
      // ``<audio>`` elements + ``AudioContext`` when the page truly
      // unloads.
    };
    // ``onServerEvent`` / ``onConnected`` etc. are useEffectEvent wrappers —
    // by the rules of React they MUST NOT appear in the dep array.
    // ``reconnectKey`` IS a dep — bumping it via ``reconnectNow`` is the
    // signal to re-run this effect after the user clicks Reconnect.
    // ``viewerMode`` IS a dep (v3.6.2) — the /live page resolves
    // ``?viewer=1`` in a post-mount effect, so the first connect can
    // race it and land on the PRIMARY endpoint. A wrongly-primary
    // OBS Browser Source displaces the operator and its disconnect
    // tears the whole session down (``finally`` → ``engine.stop()``).
    // Re-running the effect on the flip closes the wrong socket and
    // reconnects to ``/live/viewer``.
  }, [sessionId, reconnectKey, viewerMode]);

  // v2.7.3 — manual "try again" trigger surfaced to the UI. Clears the
  // exhausted flag so the banner doesn't render twice during the
  // re-attempt, then bumps reconnectKey to force the WS effect to
  // tear down its (terminated) WS and start a fresh connect loop.
  const reconnectNow = useCallback(() => {
    setWsExhausted(false);
    setWsRetryAttempt(0);
    setReconnectKey((k) => k + 1);
  }, []);

  // W3 (beatmatch feedback loop) — manual pitch-bend. The DJ nudges the
  // incoming (active) deck earlier/later to fix a transition by ear; the
  // momentary playbackRate bump is the audible correction, and the ms are
  // accumulated into pitchBendMsRef to ride on the next beatmatch_measurement
  // emit (W2) as the reinforcement signal. Resets per transition there.
  const nudgePitch = useCallback((direction: "earlier" | "later") => {
    const { accumMs, rate } = applyPitchNudge(direction, pitchBendMsRef.current);
    pitchBendMsRef.current = accumMs;
    const deck = ensureBufferDeck(activeDeckRef.current);
    deck?.nudgeRate(rate);
  }, [ensureBufferDeck]);

  // True-unmount audio cleanup. ``[]`` deps means the cleanup runs only
  // when the hook is unmounted (component truly disposed), not on every
  // WS effect re-run. Strict Mode in dev still mount/unmount/mounts on
  // first render, but at that point no audio is playing so the close()
  // is harmless. Mid-session HMR no longer kills the audio.
  useEffect(() => {
    return () => {
      stopAllDecks();
      if (audioCtxRef.current) {
        try {
          audioCtxRef.current.close();
        } catch {
          /* ignore */
        }
        audioCtxRef.current = null;
      }
    };
  }, [stopAllDecks]);

  // ── Throttled playback_pos ping to the backend + UI time tick ─────────
  // Set up an interval that both pings the backend with the current playback
  // position AND drives the UI progress bar via ``currentTrackTime`` /
  // ``currentTrackDuration``. setState lives in this event-handler-style
  // callback — fully compatible with react-hooks v7 (the rule disallows
  // setState in the *body* of an effect, not inside a setInterval handler).
  // Cleanup pair = canonical v2.4 pattern.
  const tickPlaybackPos = useEffectEvent(() => {
    if (typeof WebSocket === "undefined") return;
    const ws = wsRef.current;
    const which = activeDeckRef.current;
    const deck = which === "a" ? deckARef.current : deckBRef.current;
    if (!deck) return;
    // v3.9.5 — the id MUST come from the same object as the position.
    // Reading it from ``currentTrackIdRef`` (updated by engine events)
    // while reading the clock from the deck let the two disagree during
    // a crossfade: the ping went out carrying the INCOMING track's id
    // with the OUTGOING deck's position. The engine compared that
    // near-end position against the new, shorter track's duration and
    // declared it finished on arrival — 2026-08-23, 'Gentle Drift'
    // (208.6 s, cf point 191.6 s) into 'Warm Tide' (184.9 s), which left
    // both decks audible. The deck owns the source it scheduled, so its
    // own id and its own clock can never describe different tracks.
    const tid = deck.getTrackId() ?? currentTrackIdRef.current;
    // v3.4 — position is derived from the audio-thread clock, so it's
    // accurate to within the render quantum (~2.7 ms @ 48 kHz). Both
    // the catalog-time-into-track for the backend protocol and the
    // duration for the UI progress bar come from the deck.
    const ct = deck.position();
    const dur = deck.duration();
    setCurrentTrackTime((prev) => (Math.abs(prev - ct) > 0.05 ? ct : prev));
    setCurrentTrackDuration((prev) => (Math.abs(prev - dur) > 0.05 ? dur : prev));
    // Keep the VisualLayer compatibility shim fresh on every tick so
    // its beat clock reads sub-frame-accurate currentTime without
    // needing to know about BufferDeck.
    audioRef.current = {
      currentTime: ct,
      duration: dur,
      paused: !deck.isPlaying(),
    };
    if (!ws || ws.readyState !== WebSocket.OPEN || !tid) return;
    // Viewers must NOT send playback_pos — the primary owns the
    // engine's crossfade timer. If two clients both pinged, the
    // engine would race their timings and trigger doubles.
    if (viewerModeRef.current) return;
    try {
      ws.send(
        JSON.stringify({
          type: "playback_pos",
          track_id: tid,
          currentTime: ct,
          // v3.9.7 — the terms ``position()`` is built from. The backend
          // compares them against the rate the transition plan asked for;
          // it cannot derive them, because the position it receives is
          // computed FROM them and so always looks self-consistent.
          deck_rate: deck.getRateAtStart(),
          deck_offset: deck.getOffsetAtStart(),
        }),
      );
    } catch {
      /* ignore — next tick will retry */
    }
  });

  useEffect(() => {
    if (!sessionId) return;
    const id = window.setInterval(() => tickPlaybackPos(), PLAYBACK_POS_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [sessionId]);

  // Resume the AudioContext when the tab becomes visible again, and log
  // every "outside the DOM" event we can hook into so silent stops are
  // attributable. Covered:
  //   - visibilitychange (tab in/out of focus)
  //   - Page Lifecycle freeze/resume (Chrome may freeze backgrounded tabs)
  //   - mediaDevices.devicechange (Bluetooth/USB audio disconnect)
  // ``ctx.resume()`` requires no user gesture once the context has been
  // unlocked at least once — and the LiveStage flow always unlocks via
  // the autoplay overlay or the initial play() call.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const tryResume = (reason: string) => {
      const ctx = audioCtxRef.current;
      if (!ctx) return;
      // Aggressive recovery — call resume() unconditionally (not only on
      // ``suspended``). Chrome has a class of bugs where the renderer
      // goes silent while ``state === "running"`` after visibility
      // toggles or media-server interruptions; resume() is a no-op when
      // already running but kicks the renderer back to life when stuck.
      try {
        const p = ctx.resume();
        if (p && typeof p.then === "function") {
          p.then(
            () =>
              appendLog({
                role: "assistant",
                text: `[audioctx] kicked on ${reason} (state=${ctx.state})`,
                ts: Date.now(),
              }),
            (err) =>
              appendLog({
                role: "assistant",
                text: `[audioctx] kick on ${reason} failed: ${
                  (err as { name?: string })?.name ?? "Error"
                }`,
                ts: Date.now(),
              }),
          );
        }
      } catch {
        /* ignore — best effort */
      }
      // v3.4 — with BufferDeck there's no element-level paused/play
      // dichotomy: the source is either scheduled (will produce audio
      // when ctx.state === "running") or it's not. ctx.resume() above
      // is the entire kick we need. If the deck has no source scheduled
      // we don't try to start one — that's the engine's role via
      // engine_command load / crossfade. This simplification removes a
      // category of HTMLAudioElement-specific recovery that didn't
      // apply to the new substrate.
    };
    const onVisible = () => {
      appendLog({
        role: "assistant",
        text: `[lifecycle] visibilitychange → ${document.visibilityState}`,
        ts: Date.now(),
      });
      if (document.visibilityState === "visible") tryResume("visibilitychange");
    };
    const onFreeze = () => {
      appendLog({
        role: "assistant",
        text: "[lifecycle] page frozen (Chrome tab freezing)",
        ts: Date.now(),
      });
    };
    const onResume = () => {
      appendLog({
        role: "assistant",
        text: "[lifecycle] page resumed from freeze",
        ts: Date.now(),
      });
      tryResume("page resume");
    };
    document.addEventListener("visibilitychange", onVisible);
    document.addEventListener("freeze", onFreeze);
    document.addEventListener("resume", onResume);

    let unsubDevice: (() => void) | null = null;
    const md = navigator?.mediaDevices;
    if (md && typeof md.addEventListener === "function") {
      const onDeviceChange = () => {
        appendLog({
          role: "assistant",
          text: "[mediaDevices] devicechange (audio output may have switched)",
          ts: Date.now(),
        });
      };
      md.addEventListener("devicechange", onDeviceChange);
      unsubDevice = () => md.removeEventListener("devicechange", onDeviceChange);
    }

    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      document.removeEventListener("freeze", onFreeze);
      document.removeEventListener("resume", onResume);
      if (unsubDevice) unsubDevice();
    };
  }, [appendLog]);

  // v3.4 — simplified heartbeat for the BufferDeck substrate.
  //
  // The HTMLAudioElement substrate had a class of "running but silent"
  // failure modes (net=2 ready=2, ctx=running el.paused=false but no
  // samples reaching speakers) that needed tiered ctx.resume() +
  // el.play() + el.load() recovery. BufferDeck has fewer failure
  // surfaces: either the AudioContext is suspended (resume() unsticks
  // it — handled by the visibility-resume effect above) or the source
  // has played out (onended fires → we forward track_ended). There is
  // no equivalent of "stalled HTTP stream mid-track" because the
  // entire buffer is in memory by the time scheduleSource() is called.
  //
  // We keep a lightweight state snapshot every 2 s so any drift in
  // the visible state (ctx state, deck gain, active deck, virtual
  // position) shows up in the diagnostic panel without auto-recovery
  // logic specific to the old substrate.
  useEffect(() => {
    if (!sessionId) return;
    if (typeof window === "undefined") return;
    let last = "";
    const id = window.setInterval(() => {
      const ctx = audioCtxRef.current;
      const a = deckARef.current;
      const b = deckBRef.current;
      const which = activeDeckRef.current;
      const active = which === "a" ? a : b;
      if (!a && !b && !ctx) return;
      const snapshot =
        `ctx=${ctx?.state ?? "?"} ` +
        `gA=${a ? a.gain.gain.value.toFixed(2) : "?"} ` +
        `gB=${b ? b.gain.gain.value.toFixed(2) : "?"} ` +
        `active=${which} ` +
        `playing=${active?.isPlaying() ?? "?"} ` +
        `pos=${active ? active.position().toFixed(1) : "?"}s ` +
        `dur=${active ? active.duration().toFixed(1) : "?"}s`;
      if (snapshot !== last) {
        last = snapshot;
        appendLog({
          role: "assistant",
          text: `[heartbeat] ${snapshot}`,
          ts: Date.now(),
        });
      }
    }, 2000);
    return () => window.clearInterval(id);
  }, [sessionId, appendLog]);

  // ── Public command callbacks ────────────────────────────────────────────
  const sendCommand = useCallback(
    (cmd: LiveCommand) => {
      // Viewers are read-only — silently no-op.
      if (viewerModeRef.current) return;
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const text = COMMAND_TEXT[cmd.type];
      appendLog({ role: "user", text, ts: Date.now() });
      ws.send(JSON.stringify({ type: "user_msg", text }));
    },
    [appendLog],
  );

  const sendUserMessage = useCallback(
    (text: string) => {
      if (viewerModeRef.current) return;
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const trimmed = text.trim();
      if (!trimmed) return;
      appendLog({ role: "user", text: trimmed, ts: Date.now() });
      ws.send(JSON.stringify({ type: "user_msg", text: trimmed }));
    },
    [appendLog],
  );

  const sendRaw = useCallback((message: Record<string, unknown>) => {
    if (viewerModeRef.current) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify(message));
    } catch {
      /* ignore — caller will retry on the next sample */
    }
  }, []);

  const resumePlayback = useCallback(() => {
    // Ensure the AudioContext exists AND is resumed inside the gesture —
    // without this, a click during state==='idle' (no deck loaded yet)
    // wouldn't unlock anything, and the engine's later play() would
    // still trip Chrome's autoplay policy. Creating + resuming during
    // the user gesture marks the document as user-activated for the
    // rest of the session.
    const ctx = ensureAudioContext();
    if (ctx && typeof ctx.resume === "function") {
      try {
        const maybe = ctx.resume();
        if (maybe && typeof (maybe as Promise<void>).catch === "function") {
          (maybe as Promise<void>).catch(() => {
            /* ignore — best effort */
          });
        }
      } catch {
        /* ignore — context may already be running */
      }
    }
    // v3.4 — with the BufferDeck substrate, sources scheduled via
    // start(when) play automatically once the AudioContext is
    // running; there's no per-element play() to chase. The gesture
    // unlocked the context above, so just clear the blocked flag
    // and let the audio thread pick up any already-scheduled sources.
    setAutoplayBlocked(false);
  }, [ensureAudioContext]);

  const quit = useCallback(() => {
    const ws = wsRef.current;
    // Viewers can't ``quit`` the session — they just close their WS
    // and the bus continues without them. Still stop the local decks
    // so an embed page that called quit() actually goes quiet.
    if (!viewerModeRef.current && ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "quit" }));
      } catch {
        /* ignore */
      }
    }
    stopAllDecks();
  }, [stopAllDecks]);

  // v2.6.0 — toggle endless mode. Wire-fires only; the engine's reply
  // (`endless_mode` event) is the source of truth, mirrored into
  // `endlessMode` state by the switch handler above. This keeps the
  // pill from desyncing if the WS write succeeds but the server fails
  // to update the engine for any reason.
  const setEndlessModeWS = useCallback((enabled: boolean) => {
    if (viewerModeRef.current) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(
        JSON.stringify({ type: "set_endless_mode", enabled: !!enabled }),
      );
    } catch {
      /* ignore — WS error already handled elsewhere */
    }
  }, []);

  const dismissCriticWarning = useCallback((id: string) => {
    setCriticWarnings((prev) => prev.filter((w) => w.id !== id));
  }, []);

  return useMemo<UseLiveSessionApi>(
    () => ({
      state,
      connected,
      currentTrack,
      nextTrack,
      secondsToCrossfade,
      playlistRemaining,
      playlist,
      currentPosition,
      currentTrackTime,
      currentTrackDuration,
      log,
      djChat,
      criticWarnings,
      dismissCriticWarning,
      error,
      autoplayBlocked,
      audioRef,
      sendCommand,
      sendUserMessage,
      sendRaw,
      resumePlayback,
      quit,
      endlessMode,
      playlistRunningLow,
      setEndlessMode: setEndlessModeWS,
      greetings,
      youtube,
      wsRetryAttempt,
      wsRetryMax: MAX_WS_RETRIES,
      wsExhausted,
      reconnectNow,
      nudgePitch,
    }),
    [
      state,
      connected,
      currentTrack,
      nextTrack,
      secondsToCrossfade,
      playlistRemaining,
      playlist,
      currentPosition,
      currentTrackTime,
      currentTrackDuration,
      log,
      djChat,
      criticWarnings,
      dismissCriticWarning,
      error,
      autoplayBlocked,
      sendCommand,
      sendUserMessage,
      sendRaw,
      resumePlayback,
      quit,
      endlessMode,
      playlistRunningLow,
      setEndlessModeWS,
      greetings,
      youtube,
      wsRetryAttempt,
      wsExhausted,
      reconnectNow,
      nudgePitch,
    ],
  );
}

// ---------------------------------------------------------------------------
// Cross-component "live session active?" event bus.
//
// The PlayerProvider doesn't know about the live session. Rather than
// extending it with a new flag we use a window event so any component
// (`<MiniPlayer>` first, more later) can subscribe without prop drilling.
// ---------------------------------------------------------------------------

const LIVE_ACTIVE_EVENT = "apollo:live-active";

function publishLiveActive(active: boolean) {
  if (typeof window === "undefined") return;
  try {
    window.dispatchEvent(
      new CustomEvent(LIVE_ACTIVE_EVENT, { detail: { active } }),
    );
  } catch {
    /* ignore */
  }
}

/** Subscribe a sibling component to the "live session active" flag. */
export function useIsLiveActive(): boolean {
  const [active, setActive] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onChange = (ev: Event) => {
      const ce = ev as CustomEvent<{ active: boolean }>;
      // setState here runs inside an event handler (canonical v2.4 — not
      // a setState-in-effect violation).
      setActive(!!ce.detail?.active);
    };
    window.addEventListener(LIVE_ACTIVE_EVENT, onChange);
    return () => window.removeEventListener(LIVE_ACTIVE_EVENT, onChange);
  }, []);
  return active;
}
