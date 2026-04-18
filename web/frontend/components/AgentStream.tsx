"use client";
import { useEffect, useRef } from "react";

interface LogEntry {
  type: "text" | "tool_call" | "tool_result" | "system" | "error";
  content: string;
}

interface AgentStreamProps {
  entries: LogEntry[];
  isStreaming: boolean;
}

export type { LogEntry };

export default function AgentStream({ entries, isStreaming }: AgentStreamProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  function entryColor(type: LogEntry["type"]) {
    switch (type) {
      case "tool_call":   return "text-purple";
      case "tool_result": return "text-[#e2e2ff]/60";
      case "system":      return "text-neon";
      case "error":       return "text-danger";
      default:            return "text-[#e2e2ff]";
    }
  }

  function entryPrefix(type: LogEntry["type"]) {
    switch (type) {
      case "tool_call":   return "  [tool] ";
      case "tool_result": return "  → ";
      case "system":      return "── ";
      case "error":       return "✗ ";
      default:            return "";
    }
  }

  return (
    <div className="bg-[#0a0a0f] border border-border rounded h-full overflow-y-auto p-4 font-mono text-xs leading-relaxed">
      {entries.length === 0 && (
        <p className="text-muted">Waiting for agent...</p>
      )}
      {entries.map((e, i) => (
        <div key={i} className={`whitespace-pre-wrap ${entryColor(e.type)} animate-fade-in`}>
          <span className="opacity-50">{entryPrefix(e.type)}</span>
          {e.content}
        </div>
      ))}
      {isStreaming && (
        <span className="text-neon animate-blink">█</span>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
