import { Page, expect } from "@playwright/test";

/**
 * PhaseBar renders each phase as a span with text-neon + font-bold when active.
 * This helper waits for the given label to be the active (font-bold) node.
 * Labels mirror app/session/[id]/page.tsx's PHASES array (with "checkpoint" → "ckpt").
 */
export type PhaseLabel =
  | "genre"
  | "planning"
  | "ckpt1"
  | "critique"
  | "ckpt2"
  | "editing"
  | "validating"
  | "rating"
  | "complete";

export async function expectPhase(page: Page, label: PhaseLabel, timeout = 25_000): Promise<void> {
  // The active phase has both text-neon AND font-bold classes. The completed
  // phases have text-neon/50 (opacity), which does not match "text-neon" by
  // exact class match — so we match on font-bold only among the phase spans.
  const active = page.locator(".font-bold", { hasText: new RegExp(`^${label}$`, "i") });
  await expect(active).toBeVisible({ timeout });
}

/** Wait for the phase bar to advance *past* `label` (phase no longer active). */
export async function expectPhaseNotActive(page: Page, label: PhaseLabel): Promise<void> {
  const active = page.locator(".font-bold", { hasText: new RegExp(`^${label}$`, "i") });
  await expect(active).toHaveCount(0, { timeout: 15_000 });
}
