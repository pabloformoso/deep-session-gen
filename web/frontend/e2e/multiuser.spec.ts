import { test, expect } from "@playwright/test";
import { registerViaApi, installToken } from "./fixtures/auth";

/**
 * C4 — cross-user isolation. User A's session id, opened with User B's token,
 * must 404 + redirect to dashboard.
 */
test("C4: user B cannot view user A's session (404 → /dashboard)", async ({ page, request }) => {
  // User A creates a session via API
  const userA = await registerViaApi(request);
  const createRes = await request.post("http://localhost:8801/api/sessions", {
    headers: { Authorization: `Bearer ${userA.token}` },
  });
  expect(createRes.ok()).toBeTruthy();
  const { id: sessionId } = await createRes.json();

  // User B registers, then tries to open user A's session URL
  const userB = await registerViaApi(request);
  await installToken(page, userB);
  await page.goto(`/session/${sessionId}`);

  // Session page fetches /api/sessions/{id} on mount and redirects on 404.
  await page.waitForURL(/\/dashboard$/);
});
