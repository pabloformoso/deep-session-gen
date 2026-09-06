/**
 * The seek bar's arithmetic.
 *
 * Two states the naive version gets wrong, both visible rather than crashy:
 * an audio element reports duration 0 until metadata loads (a bar you cannot
 * drag), and `progressSec` briefly exceeds `durationSec` at the end of a file
 * (a range input pinned at the far right, reading as a bar that never
 * finishes).
 */
import { describe, expect, it } from "vitest";
import { scrubberState } from "@/components/ember/Scrubber";

describe("scrubberState", () => {
  it("uses the element's duration once it has one", () => {
    expect(scrubberState(30, 180)).toEqual({ max: 180, value: 30, seekable: true });
  });

  it("falls back to the take's own length before metadata loads", () => {
    // ACE already told us how long it is; the bar has a real length from the
    // first frame instead of snapping into place a second later.
    expect(scrubberState(0, 0, 180)).toEqual({ max: 180, value: 0, seekable: true });
  });

  it("prefers the element over the fallback once both exist", () => {
    // The file is the truth; the metas are an estimate.
    expect(scrubberState(10, 175, 180).max).toBe(175);
  });

  it("clamps progress past the end", () => {
    expect(scrubberState(181, 180).value).toBe(180);
  });

  it("clamps a negative progress", () => {
    expect(scrubberState(-5, 180).value).toBe(0);
  });

  it("is not seekable with no duration anywhere — and yields no NaN", () => {
    const s = scrubberState(0, 0);
    expect(s.seekable).toBe(false);
    expect(Number.isNaN(s.max)).toBe(false);
    expect(Number.isNaN(s.value)).toBe(false);
  });

  it("survives the infinities a streaming source reports", () => {
    // `durationSec` is Infinity for a stream with no known length.
    const s = scrubberState(5, Infinity, null);
    expect(s.seekable).toBe(false);
    expect(Number.isFinite(s.value)).toBe(true);
  });
});
