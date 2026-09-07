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
  | "complete"
  // v2.6.0 — render lifecycle. `rendering` while the SSE stream is open;
  // `failed` on subprocess non-zero exit. `complete` still marks success.
  | "rendering"
  | "failed"
  // v2.6.0 — live broadcast state. Editor endpoints reject during this
  // phase to prevent mutating a playlist mid-set.
  | "performing";

export interface Suno {
  title?: string;
  artist?: string;
  year?: string;
  prompt?: string;
  lyrics?: string;
  tags?: string;
  cover_url?: string;
  suno_id?: string;
  disambiguated?: boolean;
}

export interface Track {
  id: string;
  display_name: string;
  bpm: number | null;
  camelot_key: string | null;
  duration_sec: number | null;
  genre: string | null;
  genre_folder?: string;
  file?: string;
  variant_of?: string | null;
  suno?: Suno;
  /**
   * Relative URL of a LOCAL cover, set by /api/catalog only for tracks
   * Apollo generated itself. Null for the Suno-imported majority, which
   * carry a remote `suno.cover_url` instead — always fall back.
   */
  cover_url?: string | null;
  user_rating?: number | null;
  /**
   * v2.7.2 — 80 normalised RMS peaks (range 0..1) representing the
   * track's amplitude envelope. Populated by ``main.py --build-catalog``
   * via ``compute_waveform_peaks``. Optional because legacy entries
   * predate the field; the UI falls back to a synthetic pattern
   * when absent.
   */
  waveform_peaks?: number[];
}

export interface Catalog {
  tracks: Track[];
  genres: string[];
}

export interface StructuredProblem {
  pos_from: number;
  pos_to: number;
  key_pair: string;
  bpm_diff: number;
  text: string;
}

// v2.6.0 — server-mapped CriticNote produced by `web/backend/notes.py`.
// Replaces the client-side `adaptProblem` that used to live in
// `app/curate/page.tsx`.
export type NoteSeverity = "fix" | "tip" | "ok";
export type NoteStatus = "pending" | "applied" | "ignored";

export interface CriticNote {
  id: string;
  severity: NoteSeverity;
  /** Track position or range, e.g. "3" or "2–5". */
  target: string;
  headline: string;
  body: string;
  suggestion: string | null;
  status: NoteStatus;
}

// v2.6.0 — energy arc emitted alongside the playlist by
// `web/backend/arc.py`. `points` are aligned with `playlist[i]`.
export interface SessionArc {
  flat: boolean;
  max: number;
  peak_pos: number;
  points: number[];
}

export interface SessionState {
  id: string;
  user_id: number;
  phase: Phase;
  genre: string | null;
  duration_min: number | null;
  mood: string | null;
  // v2.5.0 — environment perception. Free-text description of where the
  // set will be played; the Planner uses it as a soft signal (see
  // `agent/run.py::_PLANNER_SYSTEM`). Null on legacy sessions or when the
  // user opted to leave it unspecified.
  environment: string | null;
  playlist: Track[];
  session_name: string | null;
  critic_verdict: string | null;
  critic_problems: string[];
  structured_problems: StructuredProblem[];
  // v2.6.0 — server-mapped CriticNotes with stable ids + apply/ignore
  // status. Empty until the critique phase has run.
  notes?: CriticNote[];
  // v2.6.0 — list of note ids the user has acted on (applied or ignored).
  handled?: string[];
  // v2.6.0 — energy arc points + flat/peak metadata.
  arc?: SessionArc | null;
  // v2.6.0 — 0–100 set-health score recomputed after every critique +
  // editor mutation. `null` until first critique runs.
  set_health?: number | null;
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
  | { type: "tool_progress"; name: string; stage: string; message: string }
  | { type: "tool_result"; name: string; result: string }
  | { type: "phase_start"; phase: Phase }
  | { type: "phase_complete"; phase: Phase; data: unknown }
  | { type: "state"; data: SessionState }
  | { type: "error"; message: string };

// Playlists (v2.2.1)
export interface Playlist {
  id: number;
  user_id: number;
  name: string;
  created_at: string;
  updated_at: string;
  track_count: number;
}

/**
 * A playlist track row may be a full hydrated `Track` from the catalog or a
 * stub `{ id, display_name, missing: true }` if the catalog no longer
 * resolves the id (eg. it was renamed during a `--build-catalog` rebuild).
 */
export interface PlaylistTrack extends Partial<Track> {
  id: string;
  display_name: string;
  missing?: boolean;
}

export interface PlaylistDetail {
  id: number;
  user_id: number;
  name: string;
  created_at: string;
  updated_at: string;
  tracks: PlaylistTrack[];
}
