"use client";
import { useEffect, useRef, useCallback } from "react";
import { getToken } from "./auth";
import type { ServerEvent } from "./types";

function deriveWsBase(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_BASE;
  if (explicit) return explicit;
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;
  if (apiBase) return apiBase.replace(/^http/, "ws");
  // Fallback matches the next.config.ts /api rewrite target: in dev the
  // backend is on 8800 and Next doesn't proxy WebSockets. For prod set
  // NEXT_PUBLIC_WS_BASE explicitly.
  return "ws://localhost:8800";
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

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as ServerEvent;
        onEventRef.current(data);
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      onEventRef.current({ type: "error", message: "WebSocket error" });
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
