import { test, expect, Request } from "@playwright/test";
import { signedInOnDashboard, registerViaApi } from "./fixtures/auth";
import { expectPhase } from "./fixtures/phase";

/**
 * D1–D4 — pin the regressions we already fixed:
 *   D1: no infinite /api/sessions/{id} polling (fd69eb1 / 6ec6e11)
 *   D2: single WS connect (no "closed before established" churn)
 *   D3: no duplicate-key React warnings on the playlist
 *   D4: register → /api/auth/me works (passlib/bcrypt regression)
 */

test.describe("D — regression guards", () => {
  test("D1+D2+D3: one session fetch, one WS connect, no duplicate-key warnings", async ({ page, request }) => {
    const sessionFetches: Request[] = [];
    page.on("request", (req) => {
      if (/\/api\/sessions\/[0-9a-f-]+$/.test(req.url())) sessionFetches.push(req);
    });

    const consoleWarnings: string[] = [];
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      const text = msg.text();
      if (msg.type() === "warning") consoleWarnings.push(text);
      if (msg.type() === "error") consoleErrors.push(text);
    });

    const wsConnects: string[] = [];
    page.on("websocket", (ws) => wsConnects.push(ws.url()));

    await signedInOnDashboard(page, request);
    await page.getByRole("button", { name: /new session/i }).click();
    await page.waitForURL(/\/session\/[0-9a-f-]+/);

    // Drive through the pipeline so we accumulate a few state changes. If the
    // old polling bug comes back, we'll see many /api/sessions/{id} calls.
    await page.getByPlaceholder(/60-minute cyberpunk set/i).fill("60-minute techno set, dark");
    await page.getByRole("button", { name: /^send$/i }).click();
    await expectPhase(page, "ckpt1");
    await page.getByRole("button", { name: /run the critic/i }).click();
    await expectPhase(page, "ckpt2");

    // Give any rogue interval a chance to fire
    await page.waitForTimeout(1500);

    // D1: initial fetch-on-mount only. If the infinite poll regresses (6ec6e11)
    // this climbs quickly — 1500ms would give a ~15s stale-time poll enough
    // chances to misbehave. The fixed-state bound catches that class of bug.
    expect(sessionFetches.length).toBeLessThanOrEqual(2);

    // D2: WS should not thrash. Dev mode's React strict-mode double mount can
    // legitimately open 2 connections per mount; the prior regression was a
    // continuous reopen loop (dozens). 6 is a comfortable ceiling that still
    // catches a thrash regression.
    expect(wsConnects.length).toBeLessThanOrEqual(6);

    // D3: no "Encountered two children with the same key" warnings
    const keyWarn = [...consoleWarnings, ...consoleErrors].filter((m) =>
      /two children with the same key/i.test(m),
    );
    expect(keyWarn).toEqual([]);
  });

  test("D4: register → /api/auth/me succeeds (passlib/bcrypt pin)", async ({ request }) => {
    const user = await registerViaApi(request);
    const res = await request.get("http://localhost:8801/api/auth/me", {
      headers: { Authorization: `Bearer ${user.token}` },
    });
    expect(res.ok()).toBeTruthy();
    const me = await res.json();
    expect(me.username).toBe(user.username);
    expect(me.email).toBe(user.email);
  });
});
