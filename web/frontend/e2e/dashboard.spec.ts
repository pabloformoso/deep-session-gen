import { test, expect } from "@playwright/test";
import { signedInOnDashboard } from "./fixtures/auth";

/**
 * A3–A4 — dashboard create/list/delete without touching the pipeline.
 */
test.describe("A — dashboard", () => {
  test("A3: create session navigates to /session/{id} with phase=Init/Genre", async ({ page, request }) => {
    await signedInOnDashboard(page, request);
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);
    // Phase bar visible — initial phase span should be present somewhere
    await expect(page.locator("text=/GENRE/i").first()).toBeVisible();
  });

  test("A4: newly-created session appears in list; delete removes it without reload", async ({ page, request }) => {
    await signedInOnDashboard(page, request);

    // Create one via UI, then come back to the dashboard
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);
    await page.getByRole("link", { name: /dashboard/i }).click();
    await page.waitForURL(/\/dashboard$/);

    const cards = page.locator(".bg-surface.border");
    await expect(cards.first()).toBeVisible();
    const initialCount = await cards.count();
    expect(initialCount).toBeGreaterThan(0);

    // Hover to reveal the delete button (it has opacity-0 until group-hover)
    await cards.first().hover();
    const delBtn = cards.first().getByRole("button", { name: /✕/ });
    await delBtn.click({ force: true });

    await expect(cards).toHaveCount(initialCount - 1);
  });
});
