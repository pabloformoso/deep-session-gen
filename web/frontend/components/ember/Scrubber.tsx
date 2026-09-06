"use client";
/**
 * A seek bar for whatever is playing, rendered where you are looking at it.
 *
 * **The mechanics already existed.** `PlayerProvider` has published
 * `progressSec`, `durationSec` and `seek` from the start, and `MiniPlayer`
 * draws exactly this control at the bottom of the window. What was missing is
 * that a take is judged from inside the generation dialog, and the bar was two
 * hundred pixels below it, attached to a player the eye is not on. So this is
 * the same control, placed at the take.
 *
 * It matters more here than in the catalog: nobody listens to a three-minute
 * take end to end to decide whether it is any good — they jump to the middle,
 * hear the chorus, and move on.
 */
import { usePlayer } from "@/lib/player";

/** Elapsed, total and whether dragging is possible. Pure, so it is tested. */
export function scrubberState(
  progressSec: number,
  durationSec: number,
  fallbackDuration?: number | null,
): { max: number; value: number; seekable: boolean } {
  // The audio element reports 0 until metadata loads. The take's OWN duration
  // (from ACE's metas) covers that gap, so the bar has a real length from the
  // first frame instead of snapping into place a second later.
  const raw =
    Number.isFinite(durationSec) && durationSec > 0
      ? durationSec
      : Number.isFinite(fallbackDuration ?? NaN) && (fallbackDuration ?? 0) > 0
        ? (fallbackDuration as number)
        : 0;
  const max = raw > 0 ? raw : 0;
  // Clamped both ends: `progressSec` briefly exceeds `durationSec` at the end
  // of a file, and a range input with value > max renders pinned at the far
  // right, which reads as a bar that never finishes.
  const value = max > 0 ? Math.max(0, Math.min(progressSec, max)) : 0;
  return { max, value, seekable: max > 0 };
}

function clock(sec: number): string {
  if (!Number.isFinite(sec)) return "—";
  const t = Math.max(0, Math.round(sec));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

export function Scrubber({
  fallbackDuration = null,
  label = "Seek",
}: {
  /** The item's own known length, used until the audio reports its own. */
  fallbackDuration?: number | null;
  label?: string;
}) {
  const { progressSec, durationSec, seek } = usePlayer();
  const { max, value, seekable } = scrubberState(
    progressSec,
    durationSec,
    fallbackDuration,
  );

  return (
    <div className="flex items-center gap-2" data-testid="take-scrubber">
      <span className="font-mono text-[10px] text-faint w-9 text-right tabular-nums">
        {clock(value)}
      </span>
      <input
        type="range"
        min={0}
        max={max || 1}
        step={0.1}
        value={value}
        disabled={!seekable}
        aria-label={label}
        data-testid="take-seek"
        onChange={(e) => seek(Number(e.target.value))}
        className="flex-1 accent-ember h-1 cursor-pointer disabled:cursor-default disabled:opacity-40"
      />
      <span className="font-mono text-[10px] text-faint w-9 tabular-nums">
        {seekable ? clock(max) : "—"}
      </span>
    </div>
  );
}
