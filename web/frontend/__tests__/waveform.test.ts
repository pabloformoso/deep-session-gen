/**
 * The waveform's two pure pieces.
 *
 * `/live` has drawn this shape since v2.7.2 but only as a picture. Making it
 * seekable adds the arithmetic below, and both halves have a failure that is
 * visible rather than crashy: bars that collapse to a flat row when peaks are
 * missing, and a click that seeks past the end (which restarts some browsers'
 * media elements instead of stopping).
 */
import { describe, expect, it } from "vitest";
import { BARS, barHeights, seekFraction } from "@/components/ember/Waveform";

describe("barHeights", () => {
  it("uses the real peaks when there are enough", () => {
    const peaks = Array.from({ length: BARS }, (_, i) => i / BARS);
    const h = barHeights(peaks);
    expect(h).toHaveLength(BARS);
    expect(h[0]).toBeCloseTo(3);          // silence is still a visible sliver
    expect(h[BARS - 1]).toBeGreaterThan(h[0]);
  });

  it("falls back rather than collapsing to a flat row", () => {
    // A take generated five minutes ago has no peaks; a row of identical
    // stubs would read as "this file is silent".
    const h = barHeights(undefined);
    expect(h).toHaveLength(BARS);
    expect(new Set(h.map((x) => Math.round(x))).size).toBeGreaterThan(3);
  });

  it("falls back when the peaks are too few to fill the bars", () => {
    // Half an envelope drawn across the full width would misplace every peak.
    const h = barHeights([0.5, 0.9]);
    expect(h).toHaveLength(BARS);
    expect(h[0]).not.toBeCloseTo(3 + 0.5 * 29);
  });

  it("never returns a negative or NaN height", () => {
    for (const p of [undefined, [], Array(BARS).fill(0)]) {
      for (const v of barHeights(p as number[] | undefined)) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThan(0);
      }
    }
  });
});

describe("seekFraction", () => {
  it("maps a click to its position along the track", () => {
    expect(seekFraction(150, 100, 200)).toBeCloseTo(0.25);
  });

  it("clamps a click that lands outside the element", () => {
    // A pointer can report a position slightly past the edge, and seeking
    // beyond the end restarts some media elements rather than stopping.
    expect(seekFraction(50, 100, 200)).toBe(0);
    expect(seekFraction(400, 100, 200)).toBe(1);
  });

  it("returns 0 for a zero-width element instead of dividing by it", () => {
    expect(seekFraction(120, 100, 0)).toBe(0);
  });
});
