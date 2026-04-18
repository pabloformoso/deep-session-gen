"use client";
import type { StructuredProblem } from "@/lib/types";

interface CriticPanelProps {
  verdict: string | null;
  problems: string[];
  structured: StructuredProblem[];
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const styles: Record<string, string> = {
    APPROVED: "bg-neon/10 text-neon border-neon/30",
    NEEDS_FIXES: "bg-yellow-400/10 text-yellow-400 border-yellow-400/30",
    REJECT: "bg-danger/10 text-danger border-danger/30",
  };
  const style = styles[verdict] ?? "bg-muted/10 text-muted border-muted/30";
  return (
    <span className={`border rounded px-2 py-0.5 text-xs font-bold tracking-widest uppercase ${style}`}>
      {verdict}
    </span>
  );
}

export default function CriticPanel({ verdict, problems, structured }: CriticPanelProps) {
  if (!verdict) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-muted text-xs">Waiting for critique...</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <span className="text-xs text-muted uppercase tracking-widest">Critic</span>
        <VerdictBadge verdict={verdict} />
      </div>

      <div className="flex-1 p-3 space-y-3">
        {problems.length === 0 ? (
          <p className="text-neon text-xs">No issues found — set is solid.</p>
        ) : (
          problems.map((p, i) => {
            const sp = structured.find(s => p.includes(`${s.pos_from}`) && p.includes(`${s.pos_to}`));
            return (
              <div key={i} className="border border-border rounded p-2 text-xs animate-slide-up">
                {sp && (
                  <div className="flex items-center gap-2 mb-1 text-muted">
                    <span className="text-purple">pos {sp.pos_from}→{sp.pos_to}</span>
                    <span className="text-neon">{sp.key_pair}</span>
                    <span>{sp.bpm_diff > 0 ? `Δ${sp.bpm_diff} BPM` : ""}</span>
                  </div>
                )}
                <p className="text-[#e2e2ff]/80">{p}</p>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
