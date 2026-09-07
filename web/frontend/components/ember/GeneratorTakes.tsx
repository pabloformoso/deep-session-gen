"use client";
/**
 * Apollo — one generated take, and everything that can be done to it.
 *
 * Lifted VERBATIM out of `GeneratorDialog` in G6, because the generations
 * library (`/generations`) is the same take row in a different frame: the
 * wizard shows the batch you just asked for, the feed shows every batch you
 * ever asked for, and a take must behave identically in both. Nothing here
 * changed shape in the move — the two G6-only affordances are OPTIONAL props
 * the dialog does not pass, so its DOM is byte-for-byte what it was:
 *
 *   - `actions`          — extra controls left of Score (the feed's
 *                          Discard / Restore; nothing in the wizard).
 *   - `published` +      — this take went into the catalog in an earlier
 *     `publishedTrackId`   session, per the store, which is something the
 *                          wizard can never know: it only ever sees the
 *                          publishes it made itself. The two are separate
 *                          because the disposition and the id come from
 *                          different columns and either can be missing.
 *
 * The row owns three independent, read-in-order flows, each a hook in
 * `lib/generator.ts`: Score (read-only, never blocks), Edit (spawns a
 * chained card nested in this row — the DOM tree IS the lineage), and
 * Publish (one-way, confirmed, writes the catalog).
 */
import * as React from "react";
import {
  POLL_INTERVAL_MS,
  buildCritiqueRequest,
  buildEditRequest,
  buildPublishRequest,
  canEditTake,
  canPublishTake,
  canScoreTake,
  chainAppended,
  chainedTaskFor,
  editRangeError,
  editSourceLabel,
  scoreChips,
  scoreVerdict,
  suggestDisplayName,
  takeAudioUrl,
  useGeneratorTask,
  useTakeEdit,
  useTakePublish,
  useTakeScore,
  variantOptionsFor,
  type ChainedTask,
  type EditMode,
  type PublishResponse,
  type ScoreChip,
  type Take,
} from "@/lib/generator";
import { usePlayer, type Playable } from "@/lib/player";
import { Btn, Crumb } from "./primitives";
import { Banner, Spinner } from "./feedback";
import { Scrubber } from "./Scrubber";
import { Waveform } from "./Waveform";

/** G3 — the three edits, in the operator's words (API spec §3.3). */
const EDIT_MODES: ReadonlyArray<[EditMode, string, string]> = [
  [
    "repaint",
    "Repaint a stretch",
    "Regenerates only the seconds you name; the rest of the take is left alone.",
  ],
  [
    "cover",
    "Cover the whole take",
    "Keeps the shape, re-performs it. Low strength stays close to the original.",
  ],
  [
    "complete",
    "Continue the take",
    "Carries the piece on from where it ends.",
  ],
];

export function formatClock(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const total = Math.max(0, Math.round(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ── Small field wrappers ──────────────────────────────────────────────────

export const FIELD_CLS =
  "bg-transparent border border-line2 px-3 py-2 text-ember-text font-sans " +
  "text-sm outline-none placeholder:text-faint focus:border-ember-text " +
  "disabled:opacity-50";

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <Crumb>{label}</Crumb>
      {children}
      {hint && <span className="text-[11px] text-mute leading-[1.45]">{hint}</span>}
    </label>
  );
}

// ── Takes ─────────────────────────────────────────────────────────────────

export function playableFor(
  take: Take,
  taskId: string,
  genre: string,
  label: string,
): Playable {
  const metas = take.metas ?? {};
  return {
    id: `ace:${taskId}:${take.index}`,
    display_name: label,
    bpm: typeof metas.bpm === "number" ? metas.bpm : null,
    // ACE reports keyscale ("C major"); the Camelot conversion is G2's job.
    camelot_key: null,
    duration_sec: typeof metas.duration === "number" ? metas.duration : null,
    genre,
    stream_url: takeAudioUrl(take.file),
  };
}

const NO_METADATA_TITLE =
  "This take came back without a BPM or a key, so the catalog would have to " +
  "guess them — and guessed metadata is how it acquired its poisoned BPMs.";

const EDIT_TITLE =
  "Repaint, cover or continue this take. The result arrives as its own card " +
  "under this one, and can be published or edited again.";

const SCORE_TITLE =
  "Measure this take against the genre's reference tracks, and read what " +
  "the numbers say. Nothing is written, and nothing is blocked.";

/** Chip colours: the bench's verdict for a metric, or its quieter
 *  advisory tier, which is reported and never fails. */
const CHIP_TONE: Record<ScoreChip["tone"], string> = {
  in: "text-ember border-ember",
  out: "text-warn border-warn",
  advisory: "text-mute border-line2",
  unknown: "text-faint border-line2",
};

function ScoreChips({ chips }: { chips: ScoreChip[] }) {
  return (
    <div className="flex flex-wrap gap-x-2 gap-y-1">
      {chips.map((chip) => (
        <span
          key={chip.key}
          data-testid="generator-score-chip"
          data-tone={chip.tone}
          title={
            chip.band
              ? `${chip.label} ${chip.value} · reference band ${chip.band}`
              : `${chip.label} ${chip.value}`
          }
          className={
            "font-mono text-[10px] border px-1.5 py-0.5 " + CHIP_TONE[chip.tone]
          }
        >
          {chip.label} {chip.value}
          {chip.band ? (
            <span className="text-faint"> · band {chip.band}</span>
          ) : null}
        </span>
      ))}
    </div>
  );
}

export function TakeRow({
  take,
  playable,
  queue,
  genres,
  defaultGenre,
  publishedNames,
  onPublished,
  label,
  depth,
  actions,
  published,
  publishedTrackId,
}: {
  take: Take;
  playable: Playable;
  queue: Playable[];
  /** Real genre folders — a take lands in one of them, never a new one. */
  genres: string[];
  defaultGenre: string;
  /** Names already published, offered as `variant of` (source first). */
  publishedNames: string[];
  onPublished: (displayName: string, result?: PublishResponse) => void;
  /** "Take 1", or "Take 1 · repaint 2" inside a chained card. */
  label: string;
  /** How many edits deep this row sits — 0 is the original batch. */
  depth: number;
  /** G6 — controls only the feed has (Discard / Restore). */
  actions?: React.ReactNode;
  /** G6 — the STORE says this take is already in the catalog. */
  published?: boolean;
  /** G6 — and the id it went in as, when the store kept one. */
  publishedTrackId?: string | null;
}) {
  const { play, pause, currentTrack, isPlaying } = usePlayer();
  const { state: pub, open, cancel, publish } = useTakePublish();
  const {
    state: ed,
    open: openEdit,
    cancel: cancelEdit,
    change,
    submit: submitEdit,
  } = useTakeEdit();
  const { state: sc, score } = useTakeScore();
  const active = currentTrack?.id === playable.id;
  const playingThis = active && isPlaying;
  const metas = take.metas ?? {};
  const duration = typeof metas.duration === "number" ? metas.duration : null;
  const chips: string[] = [
    metas.bpm != null ? `${metas.bpm} BPM` : "— BPM",
    metas.keyscale ? String(metas.keyscale) : "— key",
    formatClock(metas.duration),
    take.seed_value != null ? `seed ${take.seed_value}` : "seed —",
  ];

  const [name, setName] = React.useState("");
  const [genreFolder, setGenreFolder] = React.useState(defaultGenre);
  const [variantOf, setVariantOf] = React.useState("");
  /** The edits released FROM this take, in the order they were asked for. */
  const [chain, setChain] = React.useState<ChainedTask[]>([]);

  const publishable = canPublishTake(take);
  const busy = pub.phase === "publishing";
  const confirming =
    pub.phase === "confirm" || pub.phase === "publishing" || pub.phase === "failed";
  // A take the store already knows as published is as inert as one this row
  // published a moment ago — the catalog entry exists either way.
  const storedPublish = Boolean(published) || Boolean(publishedTrackId);
  const alreadyPublished = pub.phase === "published" || storedPublish;

  const editable = canEditTake(take);
  const editing = ed.phase !== "idle";
  const editBusy = ed.phase === "submitting";
  const rangeError = editRangeError(ed.form, duration);
  const publishedName = pub.result?.display_name ?? null;

  const scorable = canScoreTake(take);
  const scoring = sc.phase === "scoring";
  const bandChips = sc.result ? scoreChips(sc.result) : [];

  // Defaults are computed when the panel OPENS, not at mount: a take
  // published from an earlier row changes what this one should suggest.
  const startConfirm = () => {
    const base = publishedNames[0] ?? null;
    setName(base ?? suggestDisplayName(take.prompt));
    setGenreFolder(defaultGenre);
    setVariantOf(base ?? "");
    open();
  };

  const onPublish = async () => {
    const result = await publish(
      buildPublishRequest(take, {
        displayName: name,
        genreFolder,
        variantOf: variantOf || null,
      }),
    );
    if (result) onPublished(result.display_name, result);
  };

  const onScore = () =>
    void score(buildCritiqueRequest(take, { genreFolder: defaultGenre }));

  const onEdit = async () => {
    // Read the form BEFORE awaiting: a successful submit closes the panel
    // and resets it, and the card's lineage is written from what was sent.
    const mode = ed.form.mode;
    const res = await submitEdit(
      buildEditRequest(take, ed.form, { genreFolder: defaultGenre }),
    );
    if (!res) return;
    const source = editSourceLabel(label, publishedName);
    setChain((prev) => chainAppended(prev, chainedTaskFor(res, mode, source)));
  };

  return (
    <li
      data-testid="generator-take"
      className="flex flex-col border-b border-line"
    >
      <div className="flex items-center gap-3 py-3">
        <button
          type="button"
          onClick={() => (playingThis ? pause() : play(playable, queue))}
          data-testid="generator-take-play"
          aria-label={`${playingThis ? "Pause" : "Play"} ${label}`}
          className={
            "w-9 h-9 flex-shrink-0 flex items-center justify-center border text-sm " +
            (active
              ? "border-ember text-ember"
              : "border-line2 text-ember-text hover:border-ember hover:text-ember")
          }
        >
          {playingThis ? "❚❚" : "▶"}
        </button>

        <div className="min-w-0 flex-1">
          <div className="font-display italic text-lg leading-tight">
            {label}
          </div>
          <div className="flex flex-wrap gap-x-2 gap-y-1 mt-1">
            {chips.map((c, i) => (
              <span
                key={i}
                className="font-mono text-[10px] text-mute border border-line2 px-1.5 py-0.5"
              >
                {c}
              </span>
            ))}
          </div>
          {take.result_parse_error && (
            <div className="font-mono text-[10px] text-warn mt-1">
              metadata unreadable — audio still plays
            </div>
          )}
          {/* Only for the take actually playing: a bar on every row would be
              four dead controls and one live one. Nobody listens to three
              minutes to judge a take — they jump to the middle. */}
          {active && (
            <div className="mt-2 flex flex-col gap-1">
              {/* A take has no `waveform_peaks` — it has not been through the
                  ingest — so the bars are the synthetic pattern `/live` uses.
                  The heights are decoration there; the POSITION is real. */}
              <Waveform fallbackDuration={duration} label={`Seek ${label}`} />
              <Scrubber fallbackDuration={duration} label={`Seek ${label} precisely`} />
            </div>
          )}
        </div>

        {actions}

        {/* Score is the read-only sibling: it measures, it never writes,
            and it stays available whatever else the row is doing. */}
        <Btn
          kind="ghost"
          onClick={onScore}
          disabled={!scorable || scoring}
          data-testid="generator-score"
          title={
            scorable
              ? SCORE_TITLE
              : "This take carries no audio path, so there is nothing to measure."
          }
          className="px-3 py-[7px] text-[11px] flex-shrink-0"
        >
          {scoring ? (
            <>
              <Spinner /> Scoring
            </>
          ) : sc.result ? (
            "Score again"
          ) : (
            "Score"
          )}
        </Btn>

        {/* Edit is a sibling of Publish, and steps aside while that take
            is being written to the catalog — one take, one request. */}
        <Btn
          kind="ghost"
          onClick={openEdit}
          disabled={!editable || busy || editing}
          data-testid="generator-edit"
          title={
            editable
              ? EDIT_TITLE
              : "This take carries no audio path, so there is nothing to edit."
          }
          className="px-3 py-[7px] text-[11px] flex-shrink-0"
        >
          Edit
        </Btn>

        {alreadyPublished ? (
          <Btn
            kind="ghost"
            disabled
            data-testid="generator-publish"
            title="Already in the catalog."
            className="px-3 py-[7px] text-[11px] flex-shrink-0"
          >
            Published
          </Btn>
        ) : (
          <Btn
            kind="ghost"
            onClick={startConfirm}
            disabled={!publishable || pub.phase !== "idle"}
            data-testid="generator-publish"
            title={
              publishable
                ? "Add this take to the catalog."
                : NO_METADATA_TITLE
            }
            className="px-3 py-[7px] text-[11px] flex-shrink-0"
          >
            Publish to catalog
          </Btn>
        )}
      </div>

      {sc.phase !== "idle" && (
        <div
          data-testid="generator-score-panel"
          className="flex flex-col gap-2 border border-line2 p-3 mb-3"
        >
          {/* The label carries the contract: this panel informs a
              decision, it does not stand in front of one. */}
          <Crumb tone="ember">bench score · informs, never blocks</Crumb>

          {sc.error && (
            <Banner tone="error">
              <span
                data-testid="generator-score-error"
                className="normal-case tracking-normal font-sans text-[12px]"
              >
                {sc.error}
              </span>
            </Banner>
          )}

          {/* First score only: a bordered box with nothing in it would
              read as a result, and this one takes a download. */}
          {scoring && !sc.result && (
            <span className="text-[12px] text-mute leading-[1.45]">
              Measuring this take against the {defaultGenre} references…
            </span>
          )}

          {sc.result && (
            <>
              <span
                data-testid="generator-score-verdict"
                className="text-[12px] text-ember-text leading-[1.45]"
              >
                {scoreVerdict(sc.result)}
              </span>

              {bandChips.length > 0 && <ScoreChips chips={bandChips} />}

              {sc.result.note && (
                <span
                  data-testid="generator-score-note"
                  className="text-[11px] text-mute leading-[1.45]"
                >
                  {sc.result.note}
                </span>
              )}

              {sc.result.critique && (
                <p
                  data-testid="generator-score-critique"
                  className="text-[12px] text-mute leading-[1.5] m-0"
                >
                  {sc.result.critique}
                </p>
              )}
            </>
          )}
        </div>
      )}

      {editing && (
        <div
          data-testid="generator-edit-panel"
          className="flex flex-col gap-3 border border-line2 p-3 mb-3"
        >
          {ed.error && (
            <Banner tone={ed.errorStatus === 409 ? "warn" : "error"}>
              <span
                data-testid="generator-edit-error"
                className="normal-case tracking-normal font-sans text-[12px]"
              >
                {ed.error}
              </span>
            </Banner>
          )}

          <Field
            label="what to change"
            hint={EDIT_MODES.find(([m]) => m === ed.form.mode)?.[2]}
          >
            <select
              value={ed.form.mode}
              onChange={(e) => change({ mode: e.target.value as EditMode })}
              data-testid="generator-edit-mode"
              className={FIELD_CLS}
              disabled={editBusy}
            >
              {EDIT_MODES.map(([value, title]) => (
                <option key={value} value={value}>
                  {title}
                </option>
              ))}
            </select>
          </Field>

          {ed.form.mode === "repaint" && (
            <div className="grid grid-cols-2 gap-3">
              <Field
                label={
                  duration
                    ? `from · second (0–${Math.floor(duration)})`
                    : "from · second"
                }
              >
                <input
                  type="number"
                  min={0}
                  max={duration ?? undefined}
                  step={1}
                  value={ed.form.start}
                  onChange={(e) => change({ start: Number(e.target.value) })}
                  data-testid="generator-edit-start"
                  className={FIELD_CLS}
                  disabled={editBusy}
                />
              </Field>
              <Field label="to · second" hint="−1 regenerates through to the end.">
                <input
                  type="number"
                  min={-1}
                  max={duration ?? undefined}
                  step={1}
                  value={ed.form.end}
                  onChange={(e) => change({ end: Number(e.target.value) })}
                  data-testid="generator-edit-end"
                  className={FIELD_CLS}
                  disabled={editBusy}
                />
              </Field>
            </div>
          )}

          {ed.form.mode === "cover" && (
            <Field
              label={`strength · ${ed.form.strength.toFixed(2)}`}
              hint="Low keeps the original close; high lets it wander."
            >
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={ed.form.strength}
                onChange={(e) => change({ strength: Number(e.target.value) })}
                data-testid="generator-edit-strength"
                className="accent-ember"
                disabled={editBusy}
              />
            </Field>
          )}

          <Field
            label="prompt override"
            hint="Leave empty to keep this take's own prompt."
          >
            <textarea
              value={ed.form.prompt}
              onChange={(e) => change({ prompt: e.target.value })}
              rows={2}
              data-testid="generator-edit-prompt"
              placeholder={take.prompt || "same style, more energy"}
              className={FIELD_CLS + " resize-y"}
              disabled={editBusy}
            />
          </Field>

          {rangeError && (
            <span
              data-testid="generator-edit-range-error"
              className="text-[11px] text-warn leading-[1.45]"
            >
              {rangeError}
            </span>
          )}

          <div className="flex justify-end gap-2">
            <Btn
              kind="ghost"
              type="button"
              onClick={cancelEdit}
              disabled={editBusy}
              data-testid="generator-edit-cancel"
              className="px-3 py-1.5 text-[11px]"
            >
              Cancel
            </Btn>
            <Btn
              type="button"
              onClick={() => void onEdit()}
              disabled={editBusy || rangeError !== null}
              data-testid="generator-edit-submit"
              className="px-4 py-[7px] text-[11px]"
            >
              {editBusy ? (
                <>
                  <Spinner /> Sending
                </>
              ) : (
                "Send the edit"
              )}
            </Btn>
          </div>
        </div>
      )}

      {confirming && (
        <div
          data-testid="generator-publish-confirm"
          className="flex flex-col gap-3 border border-line2 p-3 mb-3"
        >
          {pub.error && (
            <Banner tone="error">
              <span
                data-testid="generator-publish-error"
                className="normal-case tracking-normal font-sans text-[12px]"
              >
                {pub.error}
              </span>
            </Banner>
          )}

          <div className="grid grid-cols-2 gap-3">
            <Field
              label="display name"
              hint="Becomes the WAV's filename and the track's name in every set."
            >
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                data-testid="generator-publish-name"
                className={FIELD_CLS}
                disabled={busy}
              />
            </Field>
            <Field label="genre folder">
              <select
                value={genreFolder}
                onChange={(e) => setGenreFolder(e.target.value)}
                data-testid="generator-publish-genre"
                className={FIELD_CLS}
                disabled={busy}
              >
                {genres.length === 0 && <option value="">No genres found</option>}
                {genres.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          {publishedNames.length > 0 && (
            <Field
              label="variant of"
              hint="A second take of the same piece links to the first, so the no-repeat rules treat them as one."
            >
              <select
                value={variantOf}
                onChange={(e) => setVariantOf(e.target.value)}
                data-testid="generator-publish-variant"
                className={FIELD_CLS}
                disabled={busy}
              >
                <option value="">a new piece</option>
                {publishedNames.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
          )}

          <div className="flex justify-end gap-2">
            <Btn
              kind="ghost"
              type="button"
              onClick={cancel}
              disabled={busy}
              data-testid="generator-publish-cancel"
              className="px-3 py-1.5 text-[11px]"
            >
              Cancel
            </Btn>
            <Btn
              type="button"
              onClick={() => void onPublish()}
              disabled={busy || !name.trim() || !genreFolder}
              data-testid="generator-publish-submit"
              className="px-4 py-[7px] text-[11px]"
            >
              {busy ? (
                <>
                  <Spinner /> Publishing
                </>
              ) : (
                "Publish"
              )}
            </Btn>
          </div>
        </div>
      )}

      {pub.phase === "published" && pub.result && (
        <div
          data-testid="generator-published"
          className="flex flex-col gap-1 border border-line2 p-3 mb-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] text-ember border border-ember px-1.5 py-0.5">
              {pub.result.track_id}
            </span>
            <span className="font-mono text-[10px] text-mute">
              {pub.result.camelot_key} · {pub.result.bpm} BPM
              {pub.result.variant_of ? ` · take of ${pub.result.variant_of}` : ""}
            </span>
          </div>
          <span className="text-[11px] text-mute leading-[1.45]">
            {pub.result.note}
          </span>
        </div>
      )}

      {/* G6 — published before this row existed. The store kept the id and
          the disposition and nothing else, so the chip is the whole story.
          `pub.result` wins when this row did the publishing itself: that
          block carries the key, the BPM and the ingest's note. */}
      {!pub.result && storedPublish && (
        <div
          data-testid="generator-published"
          className="flex flex-wrap items-center gap-2 border border-line2 p-3 mb-3"
        >
          {publishedTrackId && (
            <span className="font-mono text-[10px] text-ember border border-ember px-1.5 py-0.5">
              {publishedTrackId}
            </span>
          )}
          <span className="font-mono text-[10px] text-mute">in the catalog</span>
        </div>
      )}

      {/* The lineage IS the nesting: an edit of this take lives inside its
          row, and an edit of that one goes a level deeper still. */}
      {chain.map((chained) => (
        <ChainedTaskCard
          key={chained.task.task_id}
          chained={chained}
          genres={genres}
          defaultGenre={defaultGenre}
          publishedNames={variantOptionsFor(publishedName, publishedNames)}
          onPublished={onPublished}
          depth={depth + 1}
        />
      ))}
    </li>
  );
}

// ── Chained card: one edit, polled exactly like an original ───────────────

/**
 * An edit's task card, rendered under the take it came from.
 *
 * It is a normal generation card in every respect — same poller, same ETA
 * countdown, same degraded-blip handling — because on the backend an edit
 * IS a normal task. The only thing that makes it an edit is the lineage
 * this component prints, which lives on the page and nowhere else.
 */
export function ChainedTaskCard({
  chained,
  genres,
  defaultGenre,
  publishedNames,
  onPublished,
  depth,
}: {
  chained: ChainedTask;
  genres: string[];
  defaultGenre: string;
  publishedNames: string[];
  onPublished: (displayName: string, result?: PublishResponse) => void;
  depth: number;
}) {
  // The task handle is adopted as the INITIAL state (the card is keyed by
  // its task id and never re-points), so polling starts without an effect.
  const { state, etaCountdown } = useGeneratorTask(
    POLL_INTERVAL_MS,
    chained.task,
  );

  const takeLabel = React.useCallback(
    (index: number) => `${chained.source} · ${chained.mode} ${index + 1}`,
    [chained.source, chained.mode],
  );

  const playables = React.useMemo(
    () =>
      state.takes.map((t, i) =>
        playableFor(
          t,
          state.taskId ?? chained.task.task_id,
          defaultGenre,
          takeLabel(i),
        ),
      ),
    [state.takes, state.taskId, chained.task.task_id, defaultGenre, takeLabel],
  );

  return (
    <div
      data-testid="generator-chained-card"
      className="border-l-2 border-line2 pl-3 mb-3 flex flex-col gap-2"
    >
      <div className="flex items-baseline justify-between gap-2">
        <span data-testid="generator-chained-lineage">
          <Crumb tone="ember">{chained.lineage}</Crumb>
        </span>
        {state.phase === "pending" && (
          <span className="flex items-center gap-2 font-mono text-[10px] text-mute uppercase tracking-mono">
            <Spinner />
            <span data-testid="generator-chained-eta">
              {etaCountdown == null
                ? "eta unknown"
                : etaCountdown === 0
                  ? "any second now"
                  : `~${etaCountdown}s left`}
            </span>
            {state.degraded && (
              <span
                className="text-faint"
                data-testid="generator-chained-degraded"
              >
                · reconnecting
              </span>
            )}
          </span>
        )}
      </div>

      {state.phase === "failed" && state.error && (
        <Banner tone="error">
          <span
            data-testid="generator-chained-error"
            className="normal-case tracking-normal font-sans text-[12px]"
          >
            {state.error}
          </span>
        </Banner>
      )}

      {state.takes.length > 0 && (
        <ul className="list-none m-0 p-0 flex flex-col">
          {state.takes.map((t, i) => (
            <TakeRow
              key={`${state.taskId}-${t.index}-${i}`}
              take={t}
              playable={playables[i]}
              queue={playables}
              genres={genres}
              defaultGenre={defaultGenre}
              publishedNames={publishedNames}
              onPublished={onPublished}
              label={takeLabel(i)}
              depth={depth}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
