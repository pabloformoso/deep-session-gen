import { describe, it, expect, beforeEach } from "vitest";
import { saveAuth, clearAuth, getToken, getUser, isLoggedIn } from "../auth";

describe("lib/auth", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("saveAuth writes both token and user", () => {
    saveAuth("tok123", { id: 1, username: "alice", email: "a@t.io" });
    expect(getToken()).toBe("tok123");
    expect(getUser()).toEqual({ id: 1, username: "alice", email: "a@t.io" });
  });

  it("clearAuth removes both", () => {
    saveAuth("tok123", { id: 1, username: "a", email: "a@t.io" });
    clearAuth();
    expect(getToken()).toBeNull();
    expect(getUser()).toBeNull();
  });

  it("isLoggedIn reflects token presence", () => {
    expect(isLoggedIn()).toBe(false);
    saveAuth("tok", { id: 1, username: "a", email: "a@t.io" });
    expect(isLoggedIn()).toBe(true);
  });

  it("getUser returns null when nothing stored", () => {
    expect(getUser()).toBeNull();
  });
});
