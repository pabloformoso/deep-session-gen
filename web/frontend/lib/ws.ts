"use client";
import { useEffect, useRef, useCallback } from "react";
import { getToken } from "./auth";
import type { ServerEvent } from "./types";

function deriveWsBase(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_BASE;
  if (explicit) return explicit;
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;
  if (apiBase) return apiBase.replace(/^http/, "ws");
  // Next doesn't proxy WebSockets, so the browser connects directly to the
  // backend. Default to the canonical dev port (matches the /api rewrite in
  // next.config.ts). For prod or non-default ports set NEXT_PUBLIC_WS_BASE.
  return "ws://localhost:8000";
}

const WS_BASE = deriveWsBase();

export function useSessionWS(
  sessionId: string | null,
  onEvent: (event: ServerEvent) => void,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!sessionId) return;
    const token = getToken();
    if (!token) return;

    const ws = new WebSocket(`${WS_BASE}/ws/sessions/${sessionId}?token=${token}`);
    wsRef.current = ws;
    let opened = false;
    let cancelled = false;

    ws.onopen = () => {
      opened = true;
      if (cancelled) ws.close();
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as ServerEvent;
        onEventRef.current(data);
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      // Suppress handshake-time errors (StrictMode double-mount, backend
      // reload wiping the in-memory session store). Only surface errors
      // after a successful open.
      if (opened && !cancelled) {
        onEventRef.current({ type: "error", message: "WebSocket error" });
      }
    };

    return () => {
      cancelled = true;
      wsRef.current = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else if (ws.readyState === WebSocket.CONNECTING) {
        // Defer close until the handshake completes; closing mid-handshake
        // triggers the "closed before connection is established" warning.
        ws.addEventListener("open", () => ws.close(), { once: true });
      }
    };
  }, [sessionId]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
