import { Page, APIRequestContext, expect } from "@playwright/test";
import { randomUUID } from "crypto";

export interface E2EUser {
  username: string;
  email: string;
  password: string;
  token: string;
  id: number;
}

const API_BASE = process.env.APOLLO_E2E_API ?? "http://localhost:8801";

/** Register a fresh user via the REST API. Faster than driving the UI. */
export async function registerViaApi(request: APIRequestContext): Promise<E2EUser> {
  const username = `e2e-${randomUUID().slice(0, 8)}`;
  const email = `${username}@example.com`;
  const password = "pw12345";
  const res = await request.post(`${API_BASE}/api/auth/register`, {
    data: { username, email, password },
  });
  expect(res.ok(), `register failed: ${await res.text()}`).toBeTruthy();
  const body = await res.json();
  return { username, email, password, token: body.access_token, id: body.user.id };
}

/** Inject an auth token into localStorage so the page boots already-signed-in. */
export async function installToken(page: Page, user: E2EUser): Promise<void> {
  await page.addInitScript(
    ({ token, u }) => {
      localStorage.setItem("apollo_token", token);
      localStorage.setItem("apollo_user", JSON.stringify(u));
    },
    { token: user.token, u: { id: user.id, username: user.username, email: user.email } },
  );
}

/** Register + install token + navigate to dashboard. One call for the common setup. */
export async function signedInOnDashboard(
  page: Page,
  request: APIRequestContext,
): Promise<E2EUser> {
  const user = await registerViaApi(request);
  await installToken(page, user);
  await page.goto("/dashboard");
  await expect(page.getByRole("button", { name: /new session/i })).toBeVisible();
  return user;
}
