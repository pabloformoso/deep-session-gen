"use client";
/**
 * A waveform you can click to move through the song.
 *
 * `/live` has drawn this shape since v2.7.2 — 80 bars from the catalog's
 * `waveform_peaks`, with a synthetic sine fallback so a legacy entry never
 * collapses into a flat row. That one is display-only: it shows where the
 * playhead is and you cannot move it. This is the same picture made seekable,
 * which is what judging a take actually needs.
 *
 * **Peaks are optional on purpose.** The 512 imported tracks carry them
 * (`compute_waveform_peaks` at build time); a take that ACE generated five
 * minutes ago does not, because it has not been through the ingest. Rather
 * than show nothing there, the synthetic pattern keeps the control usable —
 * the bar heights are then decoration, but the POSITION is real.
 */
import { usePlayer } from "@/lib/player";

export const BARS = 80;

/**
 * Bar heights in px. Real peaks when there are enough of them, otherwise the
 * same synthetic pattern `/live` falls back to, so the two surfaces look like
 * one app rather than two.
 */
export function barHeights(peaks: number[] | undefined, bars = BARS): number[] {
  const valid = Array.isArray(peaks) && peaks.length >= bars;
  return Array.from({ length: bars }, (_, k) =>
    valid
      ? 3 + (peaks![k] ?? 0) * 29
      : 3 + Math.abs(Math.sin(k * 0.4) * 17) + ((k * 17) % 5),
  );
}

/**
 * Where a click landed, as a fraction of the track. Clamped: a pointer can
 * report a position slightly outside the element it is over, and seeking past
 * the end restarts some browsers' media elements rather than stopping.
 */
export function seekFraction(clientX: number, left: number, width: number): number {
  if (!(width > 0)) return 0;
  return Math.max(0, Math.min(1, (clientX - left) / width));
}

export function Waveform({
  peaks,
  fallbackDuration = null,
  label = "Seek",
}: {
  peaks?: number[];
  /** Used until the audio element reports its own length. */
  fallbackDuration?: number | null;
  label?: string;
}) {
  const { progressSec, durationSec, seek } = usePlayer();

  const total =
    Number.isFinite(durationSec) && durationSec > 0
      ? durationSec
      : Number.isFinite(fallbackDuration ?? NaN) && (fallbackDuration ?? 0) > 0
        ? (fallbackDuration as number)
        : 0;
  const frac = total > 0 ? Math.max(0, Math.min(1, progressSec / total)) : 0;
  const playIdx = Math.floor(frac * BARS);
  const heights = barHeights(peaks);

  const onSeek = (clientX: number, el: HTMLElement) => {
    if (!(total > 0)) return;
    const r = el.getBoundingClientRect();
    seek(seekFraction(clientX, r.left, r.width) * total);
  };

  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={Math.round(total)}
      aria-valuenow={Math.round(progressSec)}
      data-testid="waveform"
      onClick={(e) => onSeek(e.clientX, e.currentTarget)}
      onKeyDown={(e) => {
        // Keyboard seeking, because a bar chart is not focusable by default
        // and a control you can only reach with a mouse is half a control.
        if (e.key === "ArrowRight") seek(progressSec + 5);
        if (e.key === "ArrowLeft") seek(progressSec - 5);
      }}
      className={
        "flex items-end gap-[1px] h-9 select-none outline-none " +
        (total > 0 ? "cursor-pointer" : "cursor-default opacity-50")
      }
    >
      {heights.map((h, k) => (
        <span
          key={k}
          style={{ height: `${h}px` }}
          className={
            "flex-1 min-w-0 " +
            (k === playIdx
              ? "bg-cream"
              : k < playIdx
                ? "bg-ember"
                : "bg-line2")
          }
        />
      ))}
    </div>
  );
}
