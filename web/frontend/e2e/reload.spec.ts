import { test, expect } from "@playwright/test";
import { signedInOnDashboard } from "./fixtures/auth";
import { expectPhase } from "./fixtures/phase";

/**
 * C3 — mid-pipeline reload. Drive through to checkpoint1 (Planner done),
 * reload the page, and confirm phase + playlist come back from the server
 * state, not a client-side cache.
 */
test("C3: mid-pipeline reload restores state from /api/sessions/{id}", async ({ page, request }) => {
  await signedInOnDashboard(page, request);
  await page.getByRole("button", { name: /new session/i }).click();
  await page.waitForURL(/\/session\/[0-9a-f-]+/);

  await page.getByPlaceholder(/60-minute cyberpunk set/i).fill("60-minute techno, peak time");
  await page.getByRole("button", { name: /^send$/i }).click();
  await expectPhase(page, "ckpt1");
  await expect(page.locator("text=/Track 1/i")).toBeVisible();

  await page.reload();

  await expectPhase(page, "ckpt1");
  await expect(page.locator("text=/Track 1/i")).toBeVisible();
});
