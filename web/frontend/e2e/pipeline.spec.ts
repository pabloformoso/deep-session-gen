import { test, expect } from "@playwright/test";
import { signedInOnDashboard } from "./fixtures/auth";
import { expectPhase } from "./fixtures/phase";

/**
 * B1–B8 — the pipeline critical path. Each step asserts:
 *   (1) phase-bar state after the transition
 *   (2) the next-phase input is rendered AND not disabled (no UI blocker).
 *
 * Runs serially inside a single browser context so the WebSocket stays open
 * and session state stays coherent end-to-end.
 */
test.describe.serial("B — pipeline transitions", () => {
  test("B1-B8: genre → planner → ckpt1 → critic → ckpt2 → editor → validate → rating → complete", async ({
    page,
    request,
  }) => {
    await signedInOnDashboard(page, request);

    // ── create session → land on /session/{id}
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);

    // ── B1: genre confirmed → phase advances to planning
    const genreInput = page.getByPlaceholder(/60-minute cyberpunk set/i);
    await expect(genreInput).toBeEnabled();
    await genreInput.fill("60-minute cyberpunk set, dark and intense");
    await page.getByRole("button", { name: /^send$/i }).click();

    // Backend auto-runs planner after genre confirm, so we go straight past
    // "planning" to "ckpt1". Assert the final state and the Planner's output.
    // ── B2: planner done → ckpt1; playlist populated; approve button enabled
    await expectPhase(page, "ckpt1");
    await expect(page.getByRole("button", { name: /run the critic/i })).toBeEnabled();
    await expect(page.locator("text=/Track 1/i")).toBeVisible();

    // ── B3: checkpoint 1 approve → critique starts; working indicator shown
    await page.getByRole("button", { name: /run the critic/i }).click();
    // Critic is fast under mock — expect the transition to ckpt2 to land quickly.

    // ── B4: critic done → ckpt2; verdict visible; continue button enabled
    //   ⟵ this is the regression that kicked off the plan: previously the
    //   phase_complete payload didn't flip session.phase to checkpoint2.
    await expectPhase(page, "ckpt2");
    await expect(page.getByRole("button", { name: /continue to editor/i })).toBeEnabled();
    await expect(page.locator("text=APPROVED").first()).toBeVisible();

    // ── B5: ckpt2 approve → editing; editor input focused + enabled
    await page.getByRole("button", { name: /continue to editor/i }).click();
    await expectPhase(page, "editing");
    const editorInput = page.getByPlaceholder(/swap track 3|build my-set/i);
    await expect(editorInput).toBeEnabled();

    // ── B6: editor command "build e2e-smoke" → validating → rating
    await editorInput.fill("build e2e-smoke");
    await page.getByRole("button", { name: /^run$/i }).click();

    // ── B7: validator done → rating; 1-5 buttons + notes field enabled
    await expectPhase(page, "rating");
    for (const n of [1, 2, 3, 4, 5]) {
      await expect(page.getByRole("button", { name: new RegExp(`^${n}$`) })).toBeEnabled();
    }
    await expect(page.getByPlaceholder(/notes/i)).toBeEditable();

    // ── B8: rate & finish → complete banner visible; no input left
    await page.getByRole("button", { name: /^5$/ }).click();
    await page.getByRole("button", { name: /save & finish/i }).click();
    await expectPhase(page, "complete");
    await expect(page.locator("text=/Session complete/i")).toBeVisible();

    // Back to dashboard still works
    await page.getByRole("link", { name: /dashboard/i }).click();
    await page.waitForURL(/\/dashboard$/);
    await expect(page.getByRole("button", { name: /new session/i })).toBeVisible();
  });
});
