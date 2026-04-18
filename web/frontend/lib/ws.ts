"use client";
import { useEffect, useRef, useCallback } from "react";
import { getToken } from "./auth";
import type { ServerEvent } from "./types";

const WS_BASE = "ws://localhost:8000";

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
