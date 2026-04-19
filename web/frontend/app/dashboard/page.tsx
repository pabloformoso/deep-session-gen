"use client";
import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { createSession, listSessions, deleteSession } from "@/lib/api";
import { getUser, clearAuth } from "@/lib/auth";
import type { SessionState } from "@/lib/types";

function formatPhase(phase: string) {
  return phase.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function phaseColor(phase: string) {
  if (phase === "complete") return "text-neon";
  if (phase === "rating") return "text-purple";
  if (["validating", "building"].includes(phase)) return "text-yellow-400";
  if (phase === "critique") return "text-orange-400";
  return "text-muted";
}

export default function DashboardPage() {
  const router = useRouter();
  const [user] = useState(() => getUser());
  const [sessions, setSessions] = useState<SessionState[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setSessions(await listSessions());
    } catch {
      clearAuth();
      router.push("/login");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    if (!user) { router.push("/login"); return; }
    load();
  }, [user, load, router]);

  async function handleCreate() {
    const s = await createSession();
    router.push(`/session/${s.id}`);
  }

  async function handleDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    await deleteSession(id);
    setSessions(prev => prev.filter(s => s.id !== id));
  }

  return (
    <div className="min-h-screen p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-pixel text-neon text-base glow tracking-widest">APOLLO</h1>
          <p className="text-muted text-xs mt-1">Welcome, {user?.username}</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleCreate}
            className="bg-neon text-[#0a0a0f] px-4 py-2 rounded text-xs font-bold tracking-widest uppercase hover:bg-neon-dim transition-colors"
          >
            + New Session
          </button>
          <button
            onClick={() => { clearAuth(); router.push("/login"); }}
            className="text-muted text-xs hover:text-[#e2e2ff] transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>

      {/* Sessions list */}
      {loading ? (
        <p className="text-muted text-xs animate-pulse">Loading sessions...</p>
      ) : sessions.length === 0 ? (
        <div className="border border-dashed border-border rounded p-12 text-center">
          <p className="text-muted text-sm mb-4">No sessions yet.</p>
          <button
            onClick={handleCreate}
            className="text-neon text-xs hover:underline"
          >
            Create your first session →
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {sessions.map(s => (
            <div
              key={s.id}
              onClick={() => router.push(`/session/${s.id}`)}
              className="bg-surface border border-border rounded p-4 cursor-pointer hover:border-neon transition-colors group flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <div>
                  <p className="text-sm text-[#e2e2ff] font-bold">
                    {s.session_name || s.genre || "Untitled Session"}
                  </p>
                  <p className="text-xs text-muted mt-0.5">
                    {s.genre ?? "no genre"} · {s.duration_min ?? "?"}min · {new Date(s.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`text-xs ${phaseColor(s.phase)}`}>
                  {formatPhase(s.phase)}
                </span>
                <button
                  onClick={e => handleDelete(s.id, e)}
                  className="text-muted hover:text-danger text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ✕
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
