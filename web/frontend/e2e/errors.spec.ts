import { test, expect } from "@playwright/test";
import { signedInOnDashboard } from "./fixtures/auth";

/**
 * C1 — garbage genre triggers the backend's error path (mock_pipeline returns
 * None for "garbage"/"xyzzy") and the UI stays usable for retry.
 */
test.describe("C — error paths", () => {
  test("C1: unresolved genre → error banner shown; input re-enabled for retry", async ({ page, request }) => {
    await signedInOnDashboard(page, request);
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);

    const input = page.getByPlaceholder(/60-minute cyberpunk set/i);
    await input.fill("xyzzy garbage");
    await page.getByRole("button", { name: /^send$/i }).click();

    await expect(page.locator("text=/Could not confirm genre/i")).toBeVisible();
    // Input should be usable again for a retry (phase went back to init/genre)
    await expect(page.getByPlaceholder(/60-minute cyberpunk set/i)).toBeEnabled();
  });

  /**
   * C2 — phase failure mid-Planner. The mock pipeline raises a RuntimeError
   * when the prompt contains "crash" (sentinel surfaced via mood='crash').
   * The WS handler catches it, emits a graceful `error` event, and the user
   * can navigate back to the dashboard and start a fresh session.
   */
  test("C2: planner crash → error banner; user can recover via dashboard", async ({ page, request }) => {
    await signedInOnDashboard(page, request);
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);

    await page.getByPlaceholder(/60-minute cyberpunk set/i)
      .fill("60-minute techno set, please crash the planner");
    await page.getByRole("button", { name: /^send$/i }).click();

    await expect(page.locator("text=/RuntimeError|simulated planner crash/i")).toBeVisible();

    // Recovery: back to dashboard and create a new session
    await page.getByRole("link", { name: /dashboard/i }).click();
    await page.waitForURL(/\/dashboard$/);
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);
    await expect(page.getByPlaceholder(/60-minute cyberpunk set/i)).toBeEnabled();
  });
});
