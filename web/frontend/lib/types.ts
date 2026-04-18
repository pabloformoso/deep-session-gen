export type Phase =
  | "init"
  | "genre"
  | "planning"
  | "checkpoint1"
  | "critique"
  | "checkpoint2"
  | "editing"
  | "building"
  | "validating"
  | "rating"
  | "complete";

export interface Track {
  id: string;
  display_name: string;
  bpm: number | null;
  camelot_key: string | null;
  duration_sec: number | null;
  genre: string | null;
}

export interface StructuredProblem {
  pos_from: number;
  pos_to: number;
  key_pair: string;
  bpm_diff: number;
  text: string;
}

export interface SessionState {
  id: string;
  user_id: number;
  phase: Phase;
  genre: string | null;
  duration_min: number | null;
  mood: string | null;
  playlist: Track[];
  session_name: string | null;
  critic_verdict: string | null;
  critic_problems: string[];
  structured_problems: StructuredProblem[];
  validator_status: string | null;
  validator_issues: string[];
  created_at: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
}

// WebSocket event types from server
export type ServerEvent =
  | { type: "text_delta"; content: string }
  | { type: "tool_call"; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string }
  | { type: "phase_start"; phase: Phase }
  | { type: "phase_complete"; phase: Phase; data: unknown }
  | { type: "state"; data: SessionState }
  | { type: "error"; message: string };
