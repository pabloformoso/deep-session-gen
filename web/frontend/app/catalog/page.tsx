"use client";
/**
 * Apollo v2.6.0 — Catalog (Library of tracks).
 *
 * Ember design-system port of the legacy catalog. Same data + behaviour
 * (genre filter, favorites toggle, free-text search, optimistic rating
 * patch, detail drawer, add-to-playlist menu, in-page player) — only the
 * visual layer changes. ``data-testid`` hooks are preserved verbatim so
 * the existing E2E suite keeps working without spec edits.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { clearRating, getCatalog, setRating } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { usePlayer } from "@/lib/player";
import StarRating from "@/components/StarRating";
import AddToPlaylistMenu from "@/components/AddToPlaylistMenu";
import { Shell } from "@/components/ember/Shell";
import { Crumb, Stripe } from "@/components/ember/primitives";
import type { Track } from "@/lib/types";
import { GenerateSongs } from "@/components/ember/GenerateSongs";
import { Waveform } from "@/components/ember/Waveform";

function formatDuration(sec: number | null | undefined) {
  if (!sec) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function CatalogPage() {
  const router = useRouter();
  const { user, hydrated } = useAuth();
  const [tracks, setTracks] = useState<Track[]>([]);
  const [genres, setGenres] = useState<string[]>([]);
  const [genre, setGenre] = useState<string>("");
  const [search, setSearch] = useState("");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [selected, setSelected] = useState<Track | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!hydrated) return;
    if (!user) router.push("/login");
  }, [hydrated, user, router]);

  useEffect(() => {
    if (!hydrated || !user) return;
    let cancelled = false;
    getCatalog(genre || undefined)
      .then((c) => {
        if (cancelled) return;
        setTracks(c.tracks);
        setGenres((prev) => (prev.length === 0 ? c.genres : prev));
        setError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message ?? "Failed to load catalog");
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [hydrated, user, genre]);

  const handleGenreChange = useCallback(
    (next: string) => {
      if (next === genre) return;
      setLoading(true);
      setGenre(next);
    },
    [genre],
  );

  const updateLocalRating = useCallback(
    (trackId: string, next: number | null) => {
      setTracks((prev) =>
        prev.map((t) => (t.id === trackId ? { ...t, user_rating: next } : t)),
      );
      setSelected((cur) =>
        cur && cur.id === trackId ? { ...cur, user_rating: next } : cur,
      );
    },
    [],
  );

  const handleRate = useCallback(
    async (trackId: string, rating: number) => {
      updateLocalRating(trackId, rating);
      try {
        await setRating(trackId, rating);
      } catch (e) {
        getCatalog(genre || undefined)
          .then((c) => setTracks(c.tracks))
          .catch(() => {});
        setError((e as Error).message ?? "Failed to save rating");
      }
    },
    [genre, updateLocalRating],
  );

  const handleClearRating = useCallback(
    async (trackId: string) => {
      updateLocalRating(trackId, null);
      try {
        await clearRating(trackId);
      } catch (e) {
        getCatalog(genre || undefined)
          .then((c) => setTracks(c.tracks))
          .catch(() => {});
        setError((e as Error).message ?? "Failed to clear rating");
      }
    },
    [genre, updateLocalRating],
  );

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    let out = tracks;
    if (favoritesOnly) {
      out = out.filter(
        (t) => t.user_rating !== null && (t.user_rating ?? 0) >= 4,
      );
    }
    if (q) {
      out = out.filter((t) => {
        const hay = [
          t.display_name,
          t.suno?.title,
          t.suno?.tags,
          t.camelot_key ?? "",
        ]
          .join(" ")
          .toLowerCase();
        return hay.includes(q);
      });
    }
    return out;
  }, [tracks, search, favoritesOnly]);

  if (!user) return null;

  return (
    <Shell username={user.username}>
      <section className="px-[60px] pt-10 pb-6 border-b border-line">
        <div className="flex items-end justify-between gap-6">
          <div>
            <Crumb>library · {loading ? "loading…" : `${filtered.length} tracks`}</Crumb>
            <h1 className="font-display italic font-normal text-[64px] leading-[0.95] tracking-display-tight m-0 mt-2">
              The catalog<span className="text-ember">.</span>
            </h1>
          </div>
          {/* Making songs belongs where the library is, not only inside the
              session wizard. Renders nothing when ACE is unreachable. The
              genre filter you are looking at preselects the form. */}
          <GenerateSongs defaultGenre={genre || null} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="search name · tag · key…"
            className="w-72 bg-transparent border-0 border-b border-line2 px-0 py-2
              font-display italic text-xl text-cream
              outline-none focus:border-ember transition-colors
              placeholder:text-faint"
          />
        </div>

        {/* Genre + favorites pills */}
        <div className="flex flex-wrap items-center gap-2 mt-6">
          <FilterPill
            active={genre === ""}
            onClick={() => handleGenreChange("")}
          >
            all
          </FilterPill>
          {genres.map((g) => (
            <FilterPill
              key={g}
              active={genre === g}
              onClick={() => handleGenreChange(g)}
            >
              {g}
            </FilterPill>
          ))}
          <button
            onClick={() => setFavoritesOnly((v) => !v)}
            aria-pressed={favoritesOnly}
            data-testid="favorites-filter"
            className={
              "font-mono text-[11px] uppercase tracking-mono px-3 py-1 border transition-colors " +
              (favoritesOnly
                ? "border-ember text-ember bg-ember/10"
                : "border-line2 text-mute hover:border-line2 hover:text-ember-text")
            }
          >
            ★ favoritos
          </button>
        </div>
      </section>

      {error && (
        <div className="mx-[60px] mt-4 border border-ember p-4 font-mono text-xs text-ember">
          {error}
        </div>
      )}

      {/* Grid */}
      <section className="px-[60px] py-8 flex-1">
        {loading ? (
          <p className="font-mono text-xs text-faint uppercase tracking-mono">
            loading catalog…
          </p>
        ) : filtered.length === 0 ? (
          <p className="font-mono text-xs text-faint uppercase tracking-mono">
            no tracks match.
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {filtered.map((t) => (
              <TrackCard
                key={t.id}
                track={t}
                list={filtered}
                onClick={() => setSelected(t)}
                onRate={handleRate}
                onClearRating={handleClearRating}
              />
            ))}
          </div>
        )}
      </section>

      {selected && (
        <TrackDetail
          track={selected}
          list={filtered}
          onClose={() => setSelected(null)}
          onRate={handleRate}
          onClearRating={handleClearRating}
        />
      )}
    </Shell>
  );
}

// ── Filter pill ──────────────────────────────────────────────────────────
function FilterPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "font-mono text-[11px] uppercase tracking-mono px-3 py-1 border transition-colors " +
        (active
          ? "border-ember text-ember bg-ember/10"
          : "border-line2 text-mute hover:text-ember-text")
      }
    >
      {children}
    </button>
  );
}

// ── Track card ──────────────────────────────────────────────────────────
function TrackCard({
  track,
  list,
  onClick,
  onRate,
  onClearRating,
}: {
  track: Track;
  list: Track[];
  onClick: () => void;
  onRate: (trackId: string, rating: number) => void;
  onClearRating: (trackId: string) => void;
}) {
  const cover = track.suno?.cover_url;
  const { play } = usePlayer();
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      data-testid="track-card"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className="group bg-surf border border-line text-left hover:border-ember transition-colors cursor-pointer focus:outline-none focus:border-ember relative"
    >
      <div className="aspect-square relative overflow-hidden">
        {cover ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={cover}
            alt={track.display_name}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
          />
        ) : (
          <Stripe alpha={0.18} className="w-full h-full flex items-center justify-center">
            <span className="font-mono text-[10px] text-faint uppercase tracking-mono">
              no art
            </span>
          </Stripe>
        )}
        {track.camelot_key && (
          <span className="absolute top-1.5 left-1.5 bg-ink/80 text-ember text-[10px] px-1.5 py-0.5 font-mono">
            {track.camelot_key}
          </span>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setMenuOpen((v) => !v);
          }}
          aria-label={`Add ${track.display_name} to a playlist`}
          data-testid="track-card-add"
          className="absolute top-1.5 right-1.5 w-7 h-7 rounded-full bg-ink/80 text-ember text-sm flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-ember hover:text-cream transition-all"
        >
          +
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            play(track, list);
          }}
          aria-label={`Play ${track.display_name}`}
          data-testid="track-card-play"
          className="absolute bottom-1.5 right-1.5 w-8 h-8 rounded-full bg-ember text-cream text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 hover:scale-110 transition-all shadow"
        >
          ▶
        </button>
      </div>
      {menuOpen && (
        <AddToPlaylistMenu
          trackId={track.id}
          onClose={() => setMenuOpen(false)}
        />
      )}
      <div className="p-3">
        <p className="font-display italic text-[17px] leading-[1.15] text-ember-text truncate">
          {track.display_name}
        </p>
        {track.suno?.disambiguated && track.suno?.title && (
          <p className="text-[10px] text-mute truncate mt-0.5">
            orig: {track.suno.title}
          </p>
        )}
        <p className="font-mono text-[10px] text-faint uppercase tracking-mono mt-1">
          {track.bpm ? `${track.bpm} BPM` : "—"} ·{" "}
          {formatDuration(track.duration_sec)}
        </p>
        <div
          className="mt-1.5"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          <StarRating
            value={track.user_rating ?? null}
            onChange={(n) => onRate(track.id, n)}
            onClear={() => onClearRating(track.id)}
            size="sm"
            label={track.display_name}
          />
        </div>
      </div>
    </div>
  );
}

// ── Detail drawer ───────────────────────────────────────────────────────
function TrackDetail({
  track,
  list,
  onClose,
  onRate,
  onClearRating,
}: {
  track: Track;
  list: Track[];
  onClose: () => void;
  onRate: (trackId: string, rating: number) => void;
  onClearRating: (trackId: string) => void;
}) {
  const suno = track.suno ?? {};
  const { play, currentTrack } = usePlayer();
  const isThisPlaying = currentTrack?.id === track.id;
  const [addOpen, setAddOpen] = useState(false);
  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 flex items-stretch justify-end animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-surf border-l border-line overflow-y-auto p-9 animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-6">
          <div>
            <Crumb>track detail</Crumb>
            <h2 className="font-display italic font-normal text-3xl tracking-display-snug mt-1">
              {track.display_name}
            </h2>
            {suno.disambiguated && suno.title && (
              <p className="text-xs text-mute mt-2">
                Original Suno title:{" "}
                <span className="text-ember-text">{suno.title}</span>
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-faint hover:text-ember text-base"
          >
            ✕
          </button>
        </div>

        {suno.cover_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={suno.cover_url}
            alt={track.display_name}
            className="w-full border border-line2 mb-6"
          />
        ) : (
          <Stripe alpha={0.18} className="w-full aspect-square mb-6 border-line2" />
        )}

          {/* The catalog's tracks DO carry `waveform_peaks` — computed at
              build time — and until now nothing drew them outside /live.
              Only while this track is the one playing: a waveform whose
              playhead cannot move is a picture, not a control. */}
          {isThisPlaying && (
            <div className="mb-4">
              <Waveform
                peaks={track.waveform_peaks}
                fallbackDuration={track.duration_sec ?? null}
                label={`Seek ${track.display_name}`}
              />
            </div>
          )}

        <div className="flex gap-2 mb-6 relative">
          <button
            onClick={() => play(track, list)}
            data-testid="track-detail-play"
            className="flex-1 bg-ember text-cream font-sans text-sm font-medium py-3 hover:brightness-110 transition-all"
          >
            ▶ Play
          </button>
          <button
            onClick={() => setAddOpen((v) => !v)}
            data-testid="track-detail-add"
            className="px-5 bg-transparent border border-line2 text-ember-text font-sans text-sm py-3 hover:border-ember hover:text-ember transition-colors"
          >
            + Playlist
          </button>
          {addOpen && (
            <AddToPlaylistMenu
              trackId={track.id}
              onClose={() => setAddOpen(false)}
            />
          )}
        </div>

        <div className="flex items-center gap-3 mb-6 border border-line p-3">
          <Crumb>your rating</Crumb>
          <StarRating
            value={track.user_rating ?? null}
            onChange={(n) => onRate(track.id, n)}
            onClear={() => onClearRating(track.id)}
            size="md"
            label={track.display_name}
          />
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm mb-6">
          <Field label="Genre" value={track.genre_folder ?? track.genre} />
          <Field label="BPM" value={track.bpm?.toString()} />
          <Field label="Key" value={track.camelot_key} />
          <Field label="Duration" value={formatDuration(track.duration_sec)} />
          <Field label="Artist" value={suno.artist} />
          <Field label="Year" value={suno.year} />
        </dl>

        {suno.prompt && (
          <Section label="Prompt">
            <p className="text-sm text-ember-text leading-[1.55] whitespace-pre-wrap">
              {suno.prompt}
            </p>
          </Section>
        )}

        {suno.tags && suno.tags !== suno.prompt && (
          <Section label="Tags">
            <p className="text-xs text-mute leading-[1.55]">{suno.tags}</p>
          </Section>
        )}

        {suno.lyrics && (
          <Section label="Lyrics">
            <pre className="text-xs text-ember-text whitespace-pre-wrap font-mono">
              {suno.lyrics}
            </pre>
          </Section>
        )}

        {track.file && (
          <p className="font-mono text-[10px] text-faint mt-8 break-all">
            <span className="text-faint">file: </span>
            {track.file}
          </p>
        )}
      </div>
    </div>
  );
}

function Field({
  label,
  value,
}: {
  label: string;
  value?: string | null;
}) {
  return (
    <div className="border-b border-line pb-1.5">
      <dt className="font-mono text-[10px] text-faint uppercase tracking-mono">
        {label}
      </dt>
      <dd className="text-ember-text mt-0.5">{value ?? "—"}</dd>
    </div>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-6">
      <Crumb className="mb-1.5 block">{label}</Crumb>
      {children}
    </section>
  );
}
