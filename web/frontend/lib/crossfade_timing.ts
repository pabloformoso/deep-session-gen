/**
 * Crossfade timing math — extracted from ``crossfadeToNext`` (live.ts) so the
 * "when does the blend land?" decision is pure and unit-testable.
 *
 * v3.6 bug 3 context: the live engine's backend computes a sample-accurate
 * ``outgoing_anchor_sec`` (the outgoing track's downbeat where the crossfade
 * should begin), but the frontend historically scheduled everything at
 * ``ctx.currentTime + SCHEDULE_LOOKAHEAD_SEC`` — an ARBITRARY clock instant.
 * That pins the incoming downbeat to a random point in the outgoing bar, so
 * the mix is off from the very first kick even when grid-warp is active
 * (grid-warp corrects tempo SLOPE, not phase INTERCEPT).
 *
 * The fix: land the crossfade when the outgoing deck actually REACHES its
 * anchor downbeat, so the incoming downbeat (started at ``when``) coincides
 * with the outgoing downbeat by construction.
 */

export interface CrossfadeWhenInput {
  /** AudioContext.currentTime right now (seconds, context clock). */
  ctxNow: number;
  /** Spec lookahead slack so the render thread can pick up the event. */
  lookaheadSec: number;
  /**
   * Current playback position of the OUTGOING deck, in catalog seconds
   * (``fromDeck.position()``).
   */
  outgoingPosSec: number;
  /**
   * Backend-computed outgoing downbeat (catalog seconds) where the crossfade
   * should begin. ``undefined`` when no phase-lock payload (legacy / loose
   * grid) — then we fall back to the plain lookahead instant.
   */
  outgoingAnchorSec?: number;
}

export interface CrossfadeWhenResult {
  /** The AudioContext time to land the crossfade at. */
  when: number;
  /**
   * Ideal seconds-from-now to the outgoing downbeat (anchor - pos), BEFORE
   * the floor clamp. Negative means the deck is already past its anchor.
   * ``null`` when there was no anchor (fallback path).
   */
  secondsUntilDownbeat: number | null;
  /**
   * True when the ideal ``when`` landed in the past and was clamped up to
   * ``ctxNow + lookahead`` — i.e. we MISSED the target downbeat and aterrizó
   * on the next available instant. A persistent clamp is the likely cause of
   * a constant few-ms late offset (we keep just missing the downbeat).
   */
  clamped: boolean;
  /**
   * The residual offset in MILLISECONDS that the beatmatch loop learns from:
   * how far past the ideal downbeat the blend actually lands (`when - ideal`,
   * in ms, never negative). 0 when not clamped (we hit the downbeat) or when
   * there was no anchor. This is the measured signal streamed to the backend
   * as `offset_ms` (W2). Positive = the incoming downbeat is late.
   */
  residualMs: number;
}

/**
 * Decide when to land the crossfade, with diagnostics.
 *
 * v3.6: ``when`` is aligned to the outgoing deck's anchor downbeat so the
 * incoming downbeat (started at ``when`` via the start() offset) coincides
 * with it by construction. Returns ``clamped``/``secondsUntilDownbeat`` so
 * the caller can log WHY a residual offset exists (missed-downbeat clamp vs
 * output latency) instead of guessing.
 */
export function computeCrossfadeWhen(
  input: CrossfadeWhenInput,
): CrossfadeWhenResult {
  const { ctxNow, lookaheadSec, outgoingPosSec, outgoingAnchorSec } = input;
  const floor = ctxNow + lookaheadSec;
  // No phase-lock payload (legacy / loose grid) → plain lookahead instant.
  if (typeof outgoingAnchorSec !== "number") {
    return {
      when: floor,
      secondsUntilDownbeat: null,
      clamped: false,
      residualMs: 0,
    };
  }
  // Land the blend when the OUTGOING deck actually reaches its anchor
  // downbeat. Distance from now to that downbeat is (anchor - pos).
  const secondsUntilDownbeat = outgoingAnchorSec - outgoingPosSec;
  const ideal = ctxNow + secondsUntilDownbeat;
  // Never schedule in the past: if the outgoing deck is already at/past its
  // anchor, fall back to the lookahead floor (the next-best landing).
  const when = Math.max(ideal, floor);
  const clamped = ideal < floor;
  // The residual the loop learns from: how far past the ideal the blend lands.
  const residualMs = Math.max(0, (when - ideal) * 1000);
  return { when, secondsUntilDownbeat, clamped, residualMs };
}

/**
 * The audio-thread instant the incoming downbeat *should* have landed on.
 *
 * ``when`` is the instant we actually schedule at. When the outgoing deck
 * was already past its anchor, ``when`` was clamped forward and the two
 * differ by ``residualMs``. Every plan the backend built — the grid-warp
 * segment times, the bass-swap drop — is measured from the DOWNBEAT, not
 * from whenever we happened to get around to scheduling, so those plans
 * must be anchored here rather than at ``when``.
 *
 * Returns a time in the PAST relative to ``when`` when clamped, which is
 * exactly right: a Web Audio ``setValueAtTime`` in the past applies
 * immediately, which is what "this bar already started" means.
 */
export function idealLandingTime(when: number, residualMs: number): number {
  if (!Number.isFinite(residualMs) || residualMs <= 0) return when;
  return when - residualMs / 1000;
}

/**
 * Where to start the incoming track so it is in phase despite landing late.
 *
 * The bug this fixes: the incoming source was always started at its own
 * downbeat (``incomingAnchorSec``), even when ``when`` had been clamped
 * because the outgoing deck was already past its anchor. The outgoing was
 * then ``residualMs`` INTO its bar while the incoming was at the very top
 * of its own — the two decks a few tens of ms out of phase on every
 * single transition. Measured live 2026-09-07: 362 of 365 crossfades
 * clamped, spread uniformly across 0–0.3 s, which is the signature of the
 * backend's ~4 Hz position polling, not of jitter. At 70 BPM a beat is
 * 0.857 s, so 0.3 s is a third of a beat — an audible flam.
 *
 * The fix costs nothing: skip the same amount into the incoming track.
 * Both decks are then equally far into their respective bars, so they are
 * phase-locked by construction and the blend still lands immediately —
 * no waiting a whole bar for the next downbeat.
 *
 * ``rate`` converts output seconds to SOURCE seconds: the deck plays the
 * incoming at ``rate`` to match the outgoing's tempo, so ``residualMs`` of
 * output time consumes ``residualMs * rate`` of the buffer.
 */
export function lateCompensatedOffset(
  incomingAnchorSec: number,
  residualMs: number,
  rate: number,
): number {
  const base = Number.isFinite(incomingAnchorSec) ? incomingAnchorSec : 0;
  if (!Number.isFinite(residualMs) || residualMs <= 0) return base;
  // A non-positive or non-finite rate would map the correction to
  // nonsense; fall back to no compensation rather than seek somewhere
  // arbitrary in the buffer.
  if (!Number.isFinite(rate) || rate <= 0) return base;
  return base + (residualMs / 1000) * rate;
}

/**
 * One pitch-bend nudge (W3 reinforcement input). Pure: given the current
 * accumulated correction and a direction, returns the new accumulator and the
 * playbackRate multiplier to apply briefly to the incoming deck.
 *
 * Sign convention (matches beatmatch-learn-knowledge + offset_ms): nudging the
 * incoming track EARLIER/forward (because it entered late) is POSITIVE and
 * speeds it up momentarily (rate > 1). Nudging LATER is negative (rate < 1).
 *
 * @param direction  "earlier" (+) or "later" (-)
 * @param accumMs    correction accumulated so far this transition (ms)
 * @param stepMs     ms added per press (default 5)
 * @param rateBump   playbackRate delta applied during the nudge (default 0.02)
 */
export function applyPitchNudge(
  direction: "earlier" | "later",
  accumMs: number,
  stepMs = 5,
  rateBump = 0.02,
): { accumMs: number; rate: number } {
  const sign = direction === "earlier" ? 1 : -1;
  return {
    accumMs: accumMs + sign * stepMs,
    rate: 1 + sign * rateBump,
  };
}

/**
 * Build the transition profile key the beatmatch loop indexes on, from the
 * outgoing/incoming track BPM + Camelot key. Mirrors the backend's
 * `(camelot_key_pair, bpm_bucket)` indexing so a profile means the same thing
 * on both sides. Pure + deterministic for unit testing.
 *
 * bpm_bucket uses the OUTGOING bpm, floored to a 2-BPM band
 * (`floor/2*2`-`+2`), e.g. 121.8 → "120-122". Unknown key/bpm → "?".
 */
export function buildMeasurementProfile(
  outgoing: { bpm?: number | null; camelot_key?: string | null },
  incoming: { camelot_key?: string | null },
): { profile: string; keyPair: string; bpmBucket: string } {
  const ok = outgoing.camelot_key ?? "?";
  const ik = incoming.camelot_key ?? "?";
  const keyPair = `${ok}->${ik}`;
  let bpmBucket = "?";
  if (typeof outgoing.bpm === "number" && Number.isFinite(outgoing.bpm)) {
    const lo = Math.floor(outgoing.bpm / 2) * 2;
    bpmBucket = `${lo}-${lo + 2}`;
  }
  return { profile: `${keyPair}|bpm${bpmBucket}`, keyPair, bpmBucket };
}
