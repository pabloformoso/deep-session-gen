import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import * as api from "../api";
import { saveAuth, clearAuth } from "../auth";

type MockFetch = ReturnType<typeof vi.fn>;

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 204 ? "No Content" : "OK",
    json: async () => body,
  } as Response;
}

describe("lib/api", () => {
  let fetchMock: MockFetch;

  beforeEach(() => {
    localStorage.clear();
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("login posts credentials and returns token", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { access_token: "tok", user: { id: 1, username: "a", email: "a@t.io" } }),
    );
    const res = await api.login("alice", "pw");
    expect(res.access_token).toBe("tok");

    const call = fetchMock.mock.calls[0];
    expect(call[0]).toBe("/api/auth/login");
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({ username: "alice", password: "pw" });
  });

  it("attaches Authorization header when token is stored", async () => {
    saveAuth("tok42", { id: 1, username: "a", email: "a@t.io" });
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { id: 1, username: "a", email: "a@t.io" }));
    await api.me();
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers.Authorization).toBe("Bearer tok42");
  });

  it("omits Authorization when no token", async () => {
    clearAuth();
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { access_token: "t", user: {} }));
    await api.login("a", "pw");
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers.Authorization).toBeUndefined();
  });

  it("throws with server-supplied detail on error", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(400, { detail: "Username already taken" }));
    await expect(api.register("a", "a@t.io", "pw")).rejects.toThrow("Username already taken");
  });
});
