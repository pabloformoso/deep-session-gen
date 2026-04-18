"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { getSession } from "@/lib/api";
import { useSessionWS } from "@/lib/ws";
import { getUser } from "@/lib/auth";
import type { SessionState, Phase, ServerEvent } from "@/lib/types";
import AgentStream, { LogEntry } from "@/components/AgentStream";
import PlaylistPanel from "@/components/PlaylistPanel";
import CriticPanel from "@/components/CriticPanel";

// ---------------------------------------------------------------------------
// Phase status bar
// ---------------------------------------------------------------------------
const PHASES: Phase[] = ["genre", "planning", "checkpoint1", "critique", "checkpoint2", "editing", "validating", "rating", "complete"];

function PhaseBar({ current }: { current: Phase }) {
  const idx = PHASES.indexOf(current);
  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1">
      {PHASES.map((p, i) => (
        <div key={p} className="flex items-center gap-1 flex-shrink-0">
          <span
            className={`text-[10px] tracking-widest uppercase transition-colors ${
              i < idx ? "text-neon/50" : i === idx ? "text-neon font-bold" : "text-muted"
            }`}
          >
            {p.replace(/_/g, " ").replace("checkpoint", "ckpt")}
          </span>
          {i < PHASES.length - 1 && <span className="text-border text-xs">›</span>}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Phase-specific input areas
// ---------------------------------------------------------------------------
function GenreInput({ onSubmit, disabled }: { onSubmit: (v: string) => void; disabled: boolean }) {
  const [value, setValue] = useState("");
  const submit = () => { if (value.trim()) { onSubmit(value.trim()); setValue(""); } };
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted">Describe your session — genre, duration, mood.</p>
      <p className="text-xs text-muted/60">Example: "Build a 60-minute deep house set, late night vibes"</p>
      <div className="flex gap-2">
        <input
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          placeholder="60-minute cyberpunk set, dark and intense..."
          disabled={disabled}
          className="flex-1 bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors disabled:opacity-40"
          autoFocus
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="bg-neon text-[#0a0a0f] px-4 py-2 rounded text-xs font-bold uppercase tracking-widest hover:bg-neon-dim transition-colors disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}

function CheckpointActions({
  phase,
  onApprove,
  onFeedback,
  disabled,
}: {
  phase: "checkpoint1" | "checkpoint2";
  onApprove: () => void;
  onFeedback: (msg: string) => void;
  disabled: boolean;
}) {
  const [feedback, setFeedback] = useState("");
  const label = phase === "checkpoint1"
    ? "Playlist looks good — run the Critic"
    : "Continue to Editor";

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">
        {phase === "checkpoint1"
          ? "Review the playlist above. Approve to run the Critic, or send feedback to the Planner."
          : "Review the critique. Approve to open the Editor, or send specific fixes to the Planner."}
      </p>
      <div className="flex gap-2">
        <input
          value={feedback}
          onChange={e => setFeedback(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && feedback.trim()) { onFeedback(feedback.trim()); setFeedback(""); } }}
          placeholder="Optional feedback..."
          disabled={disabled}
          className="flex-1 bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors disabled:opacity-40"
        />
        <button
          onClick={onApprove}
          disabled={disabled}
          className="bg-neon text-[#0a0a0f] px-4 py-2 rounded text-xs font-bold uppercase tracking-widest hover:bg-neon-dim transition-colors disabled:opacity-40 whitespace-nowrap"
        >
          {label}
        </button>
      </div>
    </div>
  );
}

function EditorInput({ onSubmit, disabled }: { onSubmit: (v: string) => void; disabled: boolean }) {
  const [value, setValue] = useState("");
  const submit = () => { if (value.trim()) { onSubmit(value.trim()); setValue(""); } };
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted">
        Edit the set — swap tracks, reorder, or type <span className="text-neon">build &lt;name&gt;</span> to render.
      </p>
      <div className="flex gap-2">
        <input
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          placeholder='swap track 3 with deep-house--midnight-groove  |  build my-set'
          disabled={disabled}
          className="flex-1 bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] font-mono focus:outline-none focus:border-neon transition-colors disabled:opacity-40"
          autoFocus
        />
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="bg-neon text-[#0a0a0f] px-4 py-2 rounded text-xs font-bold uppercase tracking-widest hover:bg-neon-dim transition-colors disabled:opacity-40"
        >
          Run
        </button>
      </div>
    </div>
  );
}

function RatingInput({ onSubmit, sessionName }: { onSubmit: (r: number, n: string) => void; sessionName: string | null }) {
  const [rating, setRating] = useState(0);
  const [notes, setNotes] = useState("");
  return (
    <div className="space-y-3">
      {sessionName && (
        <p className="text-neon text-xs">
          ✓ Built: <span className="font-bold">output/{sessionName}/</span>
        </p>
      )}
      <p className="text-xs text-muted">Rate this session (1–5):</p>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map(n => (
          <button
            key={n}
            onClick={() => setRating(n)}
            className={`w-8 h-8 rounded text-sm transition-colors ${
              n <= rating ? "bg-neon text-[#0a0a0f] font-bold" : "bg-surface border border-border text-muted"
            }`}
          >
            {n}
          </button>
        ))}
      </div>
      <input
        value={notes}
        onChange={e => setNotes(e.target.value)}
        placeholder="Notes (optional)..."
        className="w-full bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors"
      />
      <button
        onClick={() => rating > 0 && onSubmit(rating, notes)}
        disabled={rating === 0}
        className="bg-neon text-[#0a0a0f] px-4 py-2 rounded text-xs font-bold uppercase tracking-widest hover:bg-neon-dim transition-colors disabled:opacity-40"
      >
        Save & Finish
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main session page
// ---------------------------------------------------------------------------
export default function SessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.id as string;

  const user = getUser();
  const [session, setSession] = useState<SessionState | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [wsReady, setWsReady] = useState(false);
  const pendingTextRef = useRef("");

  // Flush accumulated text delta as a single log entry
  const flushText = useCallback(() => {
    if (pendingTextRef.current) {
      setLogEntries(prev => [...prev, { type: "text", content: pendingTextRef.current }]);
      pendingTextRef.current = "";
    }
  }, []);

  const appendLog = useCallback((entry: LogEntry) => {
    flushText();
    setLogEntries(prev => [...prev, entry]);
  }, [flushText]);

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "text_delta":
        pendingTextRef.current += event.content;
        setStreaming(true);
        break;
      case "tool_call":
        flushText();
        setStreaming(false);
        setLogEntries(prev => [...prev, {
          type: "tool_call",
          content: `${event.name}(${JSON.stringify(event.input)})`,
        }]);
        break;
      case "tool_result":
        setLogEntries(prev => [...prev, { type: "tool_result", content: event.result }]);
        break;
      case "phase_start":
        flushText();
        setStreaming(false);
        setLogEntries(prev => [...prev, { type: "system", content: `── ${event.phase.replace(/_/g, " ").toUpperCase()} ──` }]);
        break;
      case "phase_complete":
        flushText();
        setStreaming(false);
        setSession(prev => prev ? { ...prev, phase: event.phase as Phase, ...(event.data as object) } : prev);
        break;
      case "state":
        setSession(event.data);
        setWsReady(true);
        break;
      case "error":
        flushText();
        setStreaming(false);
        appendLog({ type: "error", content: event.message });
        break;
    }
  }, [flushText, appendLog]);

  const { send } = useSessionWS(sessionId, handleEvent);

  useEffect(() => {
    if (!user) { router.push("/login"); return; }
    getSession(sessionId).then(setSession).catch(() => router.push("/dashboard"));
  }, [sessionId, user, router]);

  const sendMsg = useCallback((type: string, content?: string) => {
    flushText();
    setStreaming(true);
    send({ type, ...(content !== undefined ? { content } : {}) });
  }, [send, flushText]);

  async function handleRate(rating: number, notes: string) {
    const { rateSession } = await import("@/lib/api");
    await rateSession(sessionId, rating, notes);
    setSession(prev => prev ? { ...prev, phase: "complete" } : prev);
    appendLog({ type: "system", content: "Session saved to memory. Thanks!" });
  }

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-muted text-xs animate-pulse">Loading session...</p>
      </div>
    );
  }

  const phase = session.phase as Phase;

  return (
    <div className="min-h-screen flex flex-col h-screen overflow-hidden">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-surface flex-shrink-0">
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="text-muted hover:text-neon text-xs transition-colors">
            ← Dashboard
          </Link>
          <span className="text-muted">|</span>
          <span className="font-pixel text-neon text-[10px] glow">APOLLO</span>
        </div>
        <PhaseBar current={phase} />
        <div className="w-24" />
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Agent stream */}
        <div className="flex flex-col flex-1 min-w-0 border-r border-border">
          <div className="flex-1 overflow-hidden p-3">
            <AgentStream entries={logEntries} isStreaming={streaming} />
          </div>

          {/* Phase input area */}
          <div className="border-t border-border p-3 bg-surface flex-shrink-0">
            {!wsReady && phase === "init" && (
              <p className="text-muted text-xs animate-pulse">Connecting...</p>
            )}

            {wsReady && phase === "init" && (
              <GenreInput onSubmit={v => sendMsg("genre_intent", v)} disabled={false} />
            )}

            {phase === "genre" && (
              <GenreInput onSubmit={v => sendMsg("genre_intent", v)} disabled={streaming} />
            )}

            {phase === "checkpoint1" && (
              <CheckpointActions
                phase="checkpoint1"
                onApprove={() => sendMsg("checkpoint_approve")}
                onFeedback={v => sendMsg("genre_intent", v)}
                disabled={streaming}
              />
            )}

            {phase === "checkpoint2" && (
              <CheckpointActions
                phase="checkpoint2"
                onApprove={() => sendMsg("checkpoint2_approve")}
                onFeedback={v => sendMsg("genre_intent", v)}
                disabled={streaming}
              />
            )}

            {phase === "editing" && (
              <EditorInput onSubmit={v => sendMsg("editor_command", v)} disabled={streaming} />
            )}

            {(phase === "planning" || phase === "critique" || phase === "validating") && (
              <p className="text-muted text-xs animate-pulse">Agent working...</p>
            )}

            {phase === "rating" && (
              <RatingInput onSubmit={handleRate} sessionName={session.session_name} />
            )}

            {phase === "complete" && (
              <p className="text-neon text-xs">
                ✓ Session complete — output saved to{" "}
                <span className="font-bold">output/{session.session_name}/</span>
              </p>
            )}
          </div>
        </div>

        {/* Right: Playlist + Critic stacked */}
        <div className="w-80 flex flex-col flex-shrink-0">
          <div className="flex-1 border-b border-border overflow-hidden">
            <PlaylistPanel
              tracks={session.playlist}
              onReorder={() => {}} // visual only; actual reorder goes via Editor agent
            />
          </div>
          <div className="h-64 overflow-hidden">
            <CriticPanel
              verdict={session.critic_verdict}
              problems={session.critic_problems}
              structured={session.structured_problems}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
