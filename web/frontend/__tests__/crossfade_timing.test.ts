/**
 * v3.6 bug 3 regression — crossfade `when` must align to the outgoing
 * downbeat, not an arbitrary clock instant.
 *
 * Found live 2026-06-26: transition Deep Dive→Twilight Echo sounded
 * "desfasado desde el primer golpe" DESPITE grid_warp active and a clean
 * incoming grid, because the frontend pinned the incoming downbeat to
 * `ctx.currentTime + lookahead` and never used the backend's
 * `outgoing_anchor_sec`. See memory project_v36_beatmatch_endbar_bug.
 *
 * RED-FIRST: the desired-behaviour tests are xfail(strict) until the v3.6
 * fix lands in computeCrossfadeWhen / live.ts.
 */
import { describe, it, expect } from "vitest";

import {
  computeCrossfadeWhen,
  buildMeasurementProfile,
  applyPitchNudge,
  idealLandingTime,
  lateCompensatedOffset,
} from "../lib/crossfade_timing";

describe("applyPitchNudge (W3 reinforcement input)", () => {
  it("earlier nudge is positive and speeds up (rate > 1)", () => {
    const r = applyPitchNudge("earlier", 0);
    expect(r.accumMs).toBe(5);
    expect(r.rate).toBeGreaterThan(1);
  });

  it("later nudge is negative and slows down (rate < 1)", () => {
    const r = applyPitchNudge("later", 0);
    expect(r.accumMs).toBe(-5);
    expect(r.rate).toBeLessThan(1);
  });

  it("accumulates across presses", () => {
    let acc = 0;
    acc = applyPitchNudge("earlier", acc).accumMs; // +5
    acc = applyPitchNudge("earlier", acc).accumMs; // +5
    acc = applyPitchNudge("later", acc).accumMs; //  -5
    expect(acc).toBe(5);
  });

  it("honors custom step and rate bump", () => {
    const r = applyPitchNudge("earlier", 10, 8, 0.05);
    expect(r.accumMs).toBe(18);
    expect(r.rate).toBeCloseTo(1.05, 6);
  });
});

describe("computeCrossfadeWhen — fallback (no phase-lock)", () => {
  it("returns ctxNow + lookahead when no outgoing anchor is given", () => {
    const r = computeCrossfadeWhen({
      ctxNow: 10.0,
      lookaheadSec: 0.1,
      outgoingPosSec: 123.4,
      outgoingAnchorSec: undefined,
    });
    expect(r.when).toBeCloseTo(10.1, 6);
    expect(r.secondsUntilDownbeat).toBeNull();
    expect(r.clamped).toBe(false);
    expect(r.residualMs).toBe(0);
  });
});

describe("computeCrossfadeWhen — residualMs (W2 measured signal)", () => {
  it("is 0 when the downbeat is hit (not clamped)", () => {
    const r = computeCrossfadeWhen({
      ctxNow: 50.0,
      lookaheadSec: 0.1,
      outgoingPosSec: 293.04,
      outgoingAnchorSec: 293.54, // 0.5s ahead → comfortably future, no clamp
    });
    expect(r.clamped).toBe(false);
    expect(r.residualMs).toBe(0);
  });

  it("equals the clamp gap in ms when the downbeat was missed", () => {
    // Deck 0.3s PAST its anchor → ideal is 0.3s in the past; clamp bumps to
    // ctxNow+lookahead. residual = when - ideal = 0.3s + lookahead = 0.4s.
    const r = computeCrossfadeWhen({
      ctxNow: 50.0,
      lookaheadSec: 0.1,
      outgoingPosSec: 293.84,
      outgoingAnchorSec: 293.54,
    });
    expect(r.clamped).toBe(true);
    expect(r.residualMs).toBeCloseTo(400, 0); // (0.3 + 0.1) * 1000
  });

  it("is never negative", () => {
    const r = computeCrossfadeWhen({
      ctxNow: 0,
      lookaheadSec: 0.05,
      outgoingPosSec: 0,
      outgoingAnchorSec: 100,
    });
    expect(r.residualMs).toBeGreaterThanOrEqual(0);
  });
});

describe("buildMeasurementProfile", () => {
  it("builds key_pair + 2-BPM bucket from outgoing/incoming tracks", () => {
    const p = buildMeasurementProfile(
      { bpm: 121.8, camelot_key: "8A" },
      { camelot_key: "8B" },
    );
    expect(p.keyPair).toBe("8A->8B");
    expect(p.bpmBucket).toBe("120-122");
    expect(p.profile).toBe("8A->8B|bpm120-122");
  });

  it("floors bpm to the lower even band edge", () => {
    expect(buildMeasurementProfile({ bpm: 123.0 }, {}).bpmBucket).toBe("122-124");
    expect(buildMeasurementProfile({ bpm: 120.0 }, {}).bpmBucket).toBe("120-122");
  });

  it("uses '?' for unknown key / bpm", () => {
    const p = buildMeasurementProfile({}, {});
    expect(p.keyPair).toBe("?->?");
    expect(p.bpmBucket).toBe("?");
    expect(p.profile).toBe("?->?|bpm?");
  });

  it("handles null fields", () => {
    const p = buildMeasurementProfile(
      { bpm: null, camelot_key: null },
      { camelot_key: null },
    );
    expect(p.profile).toBe("?->?|bpm?");
  });
});

describe("computeCrossfadeWhen — v3.6 downbeat alignment", () => {
  it(
    "waits until the outgoing deck reaches its anchor downbeat",
    () => {
      // Outgoing deck is at 293.04s; backend wants the crossfade to begin
      // at the 293.54s downbeat — i.e. 0.5s of audio from now. The blend
      // must land ~0.5s in the future (plus lookahead slack), NOT at
      // ctxNow + lookahead (which would be only 0.1s).
      const r = computeCrossfadeWhen({
        ctxNow: 50.0,
        lookaheadSec: 0.1,
        outgoingPosSec: 293.04,
        outgoingAnchorSec: 293.54,
      });
      // Desired: when ≈ ctxNow + (anchor - pos) = 50.0 + 0.5 = 50.5
      expect(r.when).toBeCloseTo(50.5, 3);
      expect(r.secondsUntilDownbeat).toBeCloseTo(0.5, 6);
      expect(r.clamped).toBe(false);
    },
  );

  it(
    "never schedules in the past — clamps to at least ctxNow + lookahead",
    () => {
      // Outgoing deck already PAST its anchor (anchor - pos = -0.3s). We
      // must not return a `when` before ctxNow + lookahead. This invariant
      // must hold in BOTH the buggy and fixed versions (it's a safety
      // floor, not the alignment behaviour), so it's a normal passing test
      // — it guards the fix from ever scheduling in the past.
      const r = computeCrossfadeWhen({
        ctxNow: 50.0,
        lookaheadSec: 0.1,
        outgoingPosSec: 293.84,
        outgoingAnchorSec: 293.54,
      });
      expect(r.when).toBeGreaterThanOrEqual(50.1 - 1e-9);
      // And it reports the clamp fired — the diagnostic that distinguishes a
      // missed-downbeat constant offset from output latency.
      expect(r.clamped).toBe(true);
      expect(r.secondsUntilDownbeat).toBeCloseTo(-0.3, 6);
    },
  );

  it(
    "incoming downbeat coincides with outgoing downbeat by construction",
    () => {
      // The whole point: the incoming source starts at `when` (its own
      // downbeat via the offset arg), so `when` must equal the wall-clock
      // moment the outgoing deck is AT its downbeat. Distance from now to
      // that moment is (anchor - pos).
      const ctxNow = 12.345;
      const lookaheadSec = 0.1;
      const outgoingPosSec = 100.0;
      const outgoingAnchorSec = 101.875; // 1.875s = one bar @ 128 BPM
      const r = computeCrossfadeWhen({
        ctxNow,
        lookaheadSec,
        outgoingPosSec,
        outgoingAnchorSec,
      });
      const secondsUntilOutgoingDownbeat = outgoingAnchorSec - outgoingPosSec;
      expect(r.when - ctxNow).toBeCloseTo(secondsUntilOutgoingDownbeat, 3);
      expect(r.clamped).toBe(false);
    },
  );
});


/**
 * v3.10.2 — landing in phase when the downbeat was already missed.
 *
 * Measured live 2026-09-07 over an 18 h set: 362 of 365 crossfades came
 * back `clamped: true`, with the lateness spread uniformly across
 * 0-0.3 s. Uniform-over-a-fixed-window is the signature of POLLING
 * GRANULARITY, not jitter — the backend samples playback position at
 * ~4 Hz, so it cannot notice the crossfade point any sooner. The
 * backend log agrees: `fired at pos=209.8s, cf_point=209.6s`.
 *
 * The old code started the incoming source at the very top of its bar
 * regardless, while the outgoing was already `residualMs` into its own.
 * At 70 BPM a beat is 0.857 s, so 0.3 s is a third of a beat.
 */
describe("lateCompensatedOffset — phase despite a missed downbeat", () => {
  it("does not move the offset when the downbeat was hit", () => {
    expect(lateCompensatedOffset(12.5, 0, 1.0)).toBe(12.5);
  });

  it("skips into the incoming by exactly how late the landing was", () => {
    // 200 ms late at native rate → start 200 ms past the incoming downbeat.
    expect(lateCompensatedOffset(12.5, 200, 1.0)).toBeCloseTo(12.7, 6);
  });

  it("converts output seconds to SOURCE seconds via the playback rate", () => {
    // The deck plays the incoming at 0.9x to match the outgoing's tempo,
    // so 200 ms of output time consumes only 180 ms of the buffer.
    expect(lateCompensatedOffset(12.5, 200, 0.9)).toBeCloseTo(12.68, 6);
    // ...and at 1.1x it consumes more.
    expect(lateCompensatedOffset(12.5, 200, 1.1)).toBeCloseTo(12.72, 6);
  });

  it.each([0, -1, NaN, Infinity])(
    "falls back to the bare anchor for an unusable rate (%s)",
    (rate) => {
      expect(lateCompensatedOffset(12.5, 200, rate as number)).toBe(12.5);
    },
  );

  it("ignores a negative or non-finite residual", () => {
    expect(lateCompensatedOffset(12.5, -50, 1.0)).toBe(12.5);
    expect(lateCompensatedOffset(12.5, NaN, 1.0)).toBe(12.5);
  });

  it("survives a missing anchor without seeking somewhere arbitrary", () => {
    expect(lateCompensatedOffset(NaN, 200, 1.0)).toBeCloseTo(0.2, 6);
  });
});

describe("idealLandingTime — plans anchor at the downbeat, not at `when`", () => {
  it("is `when` itself when nothing was clamped", () => {
    expect(idealLandingTime(50.0, 0)).toBe(50.0);
  });

  it("is in the PAST by the residual when clamped", () => {
    // A grid-warp segment scheduled here already applies — which is what
    // "this bar has already begun" means.
    expect(idealLandingTime(50.1, 200)).toBeCloseTo(49.9, 6);
  });
});

describe("the two decks end up equally far into their bars", () => {
  it.each([
    ["a hit downbeat", 293.54, 0],
    ["one ping late", 293.94, 0.1],
    ["a full ping late", 293.94, 0.3],
  ])("%s", (_label, anchor, lateness) => {
    const outgoingPos = anchor + lateness; // deck already past its anchor
    const t = computeCrossfadeWhen({
      ctxNow: 50.0,
      lookaheadSec: 0.1,
      outgoingPosSec: outgoingPos,
      outgoingAnchorSec: anchor,
    });
    const rate = 1.0;
    const startOffset = lateCompensatedOffset(12.5, t.residualMs, rate);

    // How far into its OWN bar each deck is at the instant the blend
    // lands: what it had already passed, plus what it plays between now
    // and `when`. (Note this is non-zero even for a "hit" downbeat —
    // SCHEDULE_LOOKAHEAD_SEC itself pushes `when` past the anchor, so the
    // compensation covers the lookahead slack too.)
    const outgoingIntoBar = outgoingPos - anchor + (t.when - 50.0);
    const incomingIntoBar = startOffset - 12.5;

    // The invariant the whole fix exists to hold.
    expect(incomingIntoBar).toBeCloseTo(outgoingIntoBar, 6);
  });

  it("the old behaviour violated it — this is what was wrong", () => {
    const t = computeCrossfadeWhen({
      ctxNow: 50.0,
      lookaheadSec: 0.1,
      outgoingPosSec: 293.94,
      outgoingAnchorSec: 293.54,
    });
    expect(t.clamped).toBe(true);
    // OLD: incoming started at the anchor -> 0 into its bar...
    const oldIncomingIntoBar = 12.5 - 12.5;
    // ...while the outgoing was 0.4 s past its anchor already and plays
    // another 0.1 s before the blend lands: 0.5 s into its bar.
    const outgoingIntoBar = 0.4 + (t.when - 50.0);
    expect(outgoingIntoBar).toBeCloseTo(0.5, 6);
    expect(oldIncomingIntoBar).not.toBeCloseTo(outgoingIntoBar, 2);
    // NEW: the gap closes.
    const newIncomingIntoBar =
      lateCompensatedOffset(12.5, t.residualMs, 1.0) - 12.5;
    expect(newIncomingIntoBar).toBeCloseTo(outgoingIntoBar, 6);
  });
});
