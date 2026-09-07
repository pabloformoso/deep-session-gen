import { getToken } from "./auth";
import type { Catalog, Playlist, PlaylistDetail, SessionState } from "./types";

const BASE = `${process.env.NEXT_PUBLIC_API_BASE ?? ""}/api`;

async function req<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken();
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((options.headers as Record<string, string>) ?? {}),
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// Auth
export const register = (username: string, email: string, password: string) =>
  req<{ access_token: string; user: { id: number; username: string; email: string } }>(
    "/auth/register",
    { method: "POST", body: JSON.stringify({ username, email, password }) },
  );

export const login = (username: string, password: string) =>
  req<{ access_token: string; user: { id: number; username: string; email: string } }>(
    "/auth/login",
    { method: "POST", body: JSON.stringify({ username, password }) },
  );

export const me = () =>
  req<{ id: number; username: string; email: string }>("/auth/me");

// Sessions
export const createSession = () =>
  req<SessionState>("/sessions", { method: "POST" });

// v2.6.0 — Brief flow. POSTs the brief + optional environment, the
// server parses it (Haiku) and kicks off planning as a background task.
// Returns the empty session shape plus a `parsed` block the Brief page
// reads once for the "understood as" preview.
export type ParsedBrief = {
  genre: string | null;
  duration_min: number | null;
  mood: string | null;
  venue: string | null;
  energy: string | null;
  tempo: string | null;
};
export const createSessionWithBrief = (
  brief: string,
  environment?: string,
) =>
  req<SessionState & { parsed: ParsedBrief }>("/sessions", {
    method: "POST",
    body: JSON.stringify({ brief, environment }),
  });

export const listSessions = () =>
  req<SessionState[]>("/sessions");

export const getSession = (id: string) =>
  req<SessionState>(`/sessions/${id}`);

export const deleteSession = (id: string) =>
  req<void>(`/sessions/${id}`, { method: "DELETE" });

export const rateSession = (
  id: string,
  rating: number,
  notes?: string,
  transition_ratings?: unknown[],
) =>
  req<{ ok: boolean }>(`/sessions/${id}/rate`, {
    method: "POST",
    body: JSON.stringify({ rating, notes, transition_ratings }),
  });

// v2.6.0 — Curate apply/ignore endpoints.
export const applyNote = (sessionId: string, noteId: string) =>
  req<SessionState>(`/sessions/${sessionId}/notes/${noteId}/apply`, {
    method: "POST",
  });

export const ignoreNote = (sessionId: string, noteId: string) =>
  req<{ handled: string[] }>(`/sessions/${sessionId}/notes/${noteId}/ignore`, {
    method: "POST",
  });

// v2.6.0 — Editor deterministic gestures (drag, trash, picker).
export const reorderSessionTracks = (sessionId: string, order: number[]) =>
  req<SessionState>(`/sessions/${sessionId}/tracks/reorder`, {
    method: "POST",
    body: JSON.stringify({ order }),
  });

export const deleteSessionTrack = (sessionId: string, position: number) =>
  req<SessionState>(`/sessions/${sessionId}/tracks/${position}`, {
    method: "DELETE",
  });

export const insertSessionTrack = (
  sessionId: string,
  at: number,
  trackId: string,
) =>
  req<SessionState>(`/sessions/${sessionId}/tracks/insert`, {
    method: "POST",
    body: JSON.stringify({ at, track_id: trackId }),
  });

// v2.6.0 — Render kickoff + status (subscribed-to-stream lives in
// `lib/render-stream.ts` because EventSource is GET-only).
export type StartRenderResponse = {
  jobId: string;
  streamUrl: string;
  status: "started" | "already_running";
};
export const startRender = (sessionId: string) =>
  req<StartRenderResponse>(`/sessions/${sessionId}/render`, { method: "POST" });

export type RenderStatusResponse = {
  phase: string;
  running: boolean;
  stage?: string | null;
  pct?: number;
  etaSeconds?: number | null;
  assets?: Record<string, string> | null;
  chapters?: Array<{ tMs: number; title: string; camelot: string | null }> | null;
  error?: string | null;
};
export const getRenderStatus = (sessionId: string) =>
  req<RenderStatusResponse>(`/sessions/${sessionId}/render/status`);

// v2.6.0 — SSE editor command. EventSource is GET-only and can't carry a
// body, so we use `fetch` and parse the SSE stream from the response body
// reader manually. Each `data:` frame is JSON-decoded and forwarded to
// `onEvent`. The `done` named event closes the stream; `error` does too.
export type EditorStreamHandlers = {
  onEvent: (event: Record<string, unknown>) => void;
  onDone: () => void;
  onError: (message: string) => void;
};

export async function streamEditorCommand(
  sessionId: string,
  text: string,
  handlers: EditorStreamHandlers,
): Promise<void> {
  const token = getToken();
  const res = await fetch(`${BASE}/sessions/${sessionId}/editor_command`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    handlers.onError(`HTTP ${res.status}`);
    return;
  }
  if (!res.body) {
    handlers.onError("Empty response");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent: string | null = null;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let lineEnd: number;
      while ((lineEnd = buffer.indexOf("\n")) !== -1) {
        const rawLine = buffer.slice(0, lineEnd);
        buffer = buffer.slice(lineEnd + 1);
        const line = rawLine.replace(/\r$/, "");

        if (line === "") {
          currentEvent = null;
          continue;
        }
        if (line.startsWith(":")) continue; // SSE comment / heartbeat
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
          continue;
        }
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (currentEvent === "done") {
            handlers.onDone();
            return;
          }
          if (currentEvent === "error") {
            try {
              const payload = JSON.parse(data) as { message?: string };
              handlers.onError(payload.message ?? "Error");
            } catch {
              handlers.onError(data || "Error");
            }
            return;
          }
          try {
            handlers.onEvent(JSON.parse(data));
          } catch {
            // skip malformed frames
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// Catalog
export const getCatalog = (genre?: string) => {
  const qs = genre ? `?genre=${encodeURIComponent(genre)}` : "";
  return req<Catalog>(`/catalog${qs}`);
};

// Audio streaming — `<audio>` can't set Authorization headers, so the JWT
// goes in the query string (same trick as the WebSocket auth).
export const streamUrl = (trackId: string): string => {
  const token = getToken() ?? "";
  return `${BASE}/tracks/${encodeURIComponent(trackId)}/stream?token=${encodeURIComponent(token)}`;
};

// Locally generated cover art. Same query-string token trick as
// streamUrl — an <img> can't set Authorization either. Only tracks
// Apollo generated itself have one; everything imported from Suno keeps
// its remote suno.cover_url, so callers must fall back.
export const coverUrl = (trackId: string): string => {
  const token = getToken() ?? "";
  return `${BASE}/tracks/${encodeURIComponent(trackId)}/cover?token=${encodeURIComponent(token)}`;
};

// Playlists (v2.2.1)
export const listPlaylists = () => req<Playlist[]>("/playlists");

export const createPlaylist = (name: string) =>
  req<Playlist>("/playlists", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

export const getPlaylist = (id: number) =>
  req<PlaylistDetail>(`/playlists/${id}`);

export const renamePlaylist = (id: number, name: string) =>
  req<Playlist>(`/playlists/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });

export const deletePlaylist = (id: number) =>
  req<void>(`/playlists/${id}`, { method: "DELETE" });

export const addTracks = (id: number, trackIds: string[]) =>
  req<{ playlist_id: number; track_count: number }>(
    `/playlists/${id}/tracks`,
    { method: "POST", body: JSON.stringify({ track_ids: trackIds }) },
  );

export const removeTrack = (id: number, trackId: string) =>
  req<void>(`/playlists/${id}/tracks/${encodeURIComponent(trackId)}`, {
    method: "DELETE",
  });

export const reorderTracks = (id: number, trackIds: string[]) =>
  req<{ id: number; track_ids: string[]; updated_at: string }>(
    `/playlists/${id}/order`,
    { method: "PUT", body: JSON.stringify({ track_ids: trackIds }) },
  );

// Per-user track ratings (1–5). DELETE is idempotent so the UI can fire it
// on every "click filled star" without first checking server state.
export const setRating = (trackId: string, rating: number) =>
  req<{ track_id: string; rating: number }>(
    `/tracks/${encodeURIComponent(trackId)}/rating`,
    { method: "PUT", body: JSON.stringify({ rating }) },
  );

export const clearRating = (trackId: string) =>
  req<void>(`/tracks/${encodeURIComponent(trackId)}/rating`, {
    method: "DELETE",
  });

// ── v2.7: YouTube Live Chat integration ──────────────────────────────────────
// Status endpoint is plain GET; the OAuth start is a 302 redirect we open in
// a popup window. Disconnect drops the persisted credentials server-side.
export type YouTubeStatus =
  | { connected: false }
  | { connected: true; channel_id: string; channel_title: string };

export const getYouTubeStatus = () =>
  req<YouTubeStatus>("/youtube/status").catch((e) => {
    // The endpoint 404s when GOOGLE_CLIENT_ID/SECRET aren't configured —
    // that's the "feature disabled" path, not an error. Treat it as
    // disconnected so the pill stays hidden.
    if (/not configured/i.test(String((e as Error).message))) {
      return { connected: false as const };
    }
    throw e;
  });

/** URL the frontend opens in a popup to kick off OAuth. The backend
 *  redirects to Google with a signed state token, then back here on
 *  success — the live WS picks up the new credentials on next reconnect. */
export const youtubeOAuthStartUrl = (): string => {
  const token = getToken();
  // The start endpoint requires auth via Bearer; we can't set a header on
  // a top-level window.open, so we pass the token as a query string. The
  // backend reads `auth_query_token` (existing helper used by WS upgrades
  // + render SSE) when the header isn't present.
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${BASE}/youtube/oauth/start${q}`;
};

export const disconnectYouTube = () =>
  req<{ disconnected: boolean }>("/youtube/disconnect", { method: "POST" });
