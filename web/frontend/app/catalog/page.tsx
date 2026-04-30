"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getCatalog } from "@/lib/api";
import { getUser, clearAuth } from "@/lib/auth";
import type { Track } from "@/lib/types";

function formatDuration(sec: number | null | undefined) {
  if (!sec) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function CatalogPage() {
  const router = useRouter();
  const [user, setUser] = useState<ReturnType<typeof getUser>>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [genres, setGenres] = useState<string[]>([]);
  const [genre, setGenre] = useState<string>("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Track | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const u = getUser();
    if (!u) {
      router.push("/login");
      return;
    }
    setUser(u);
  }, [router]);

  useEffect(() => {
    if (!user) return;
    setLoading(true);
    setError(null);
    getCatalog(genre || undefined)
      .then((c) => {
        setTracks(c.tracks);
        if (genres.length === 0) setGenres(c.genres);
      })
      .catch((e) => setError(e.message ?? "Failed to load catalog"))
      .finally(() => setLoading(false));
  }, [user, genre, genres.length]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return tracks;
    return tracks.filter((t) => {
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
  }, [tracks, search]);

  if (!user) return null;

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-pixel text-neon text-base glow tracking-widest">
            APOLLO / CATALOG
          </h1>
          <p className="text-muted text-xs mt-1">
            {loading ? "Loading…" : `${filtered.length} tracks`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/dashboard")}
            className="text-muted text-xs hover:text-[#e2e2ff] transition-colors"
          >
            ← Dashboard
          </button>
          <button
            onClick={() => {
              clearAuth();
              router.push("/login");
            }}
            className="text-muted text-xs hover:text-[#e2e2ff] transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <button
          onClick={() => setGenre("")}
          className={`text-xs px-3 py-1 rounded border transition-colors ${
            genre === ""
              ? "border-neon text-neon bg-neon/10"
              : "border-border text-muted hover:border-neon hover:text-neon"
          }`}
        >
          All
        </button>
        {genres.map((g) => (
          <button
            key={g}
            onClick={() => setGenre(g)}
            className={`text-xs px-3 py-1 rounded border transition-colors ${
              genre === g
                ? "border-neon text-neon bg-neon/10"
                : "border-border text-muted hover:border-neon hover:text-neon"
            }`}
          >
            {g}
          </button>
        ))}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search name / tag / key…"
          className="ml-auto bg-surface border border-border rounded px-3 py-1 text-xs text-[#e2e2ff] placeholder-muted focus:border-neon focus:outline-none w-64"
        />
      </div>

      {error && (
        <div className="border border-danger rounded p-4 text-xs text-danger mb-4">
          {error}
        </div>
      )}

      {/* Grid */}
      {loading ? (
        <p className="text-muted text-xs animate-pulse">Loading catalog…</p>
      ) : filtered.length === 0 ? (
        <p className="text-muted text-xs">No tracks match.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {filtered.map((t) => (
            <TrackCard key={t.id} track={t} onClick={() => setSelected(t)} />
          ))}
        </div>
      )}

      {/* Detail drawer */}
      {selected && (
        <TrackDetail track={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function TrackCard({ track, onClick }: { track: Track; onClick: () => void }) {
  const cover = track.suno?.cover_url;
  return (
    <button
      onClick={onClick}
      className="group bg-surface border border-border rounded overflow-hidden text-left hover:border-neon transition-colors"
    >
      <div className="aspect-square bg-[#0a0a0f] relative overflow-hidden">
        {cover ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={cover}
            alt={track.display_name}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-muted text-[10px] font-pixel">
            NO ART
          </div>
        )}
        {track.camelot_key && (
          <span className="absolute top-1 right-1 bg-[#0a0a0f]/80 text-neon text-[10px] px-1.5 py-0.5 rounded font-mono">
            {track.camelot_key}
          </span>
        )}
      </div>
      <div className="p-2">
        <p className="text-xs text-[#e2e2ff] truncate font-bold">
          {track.display_name}
        </p>
        {track.suno?.disambiguated && track.suno?.title && (
          <p className="text-[10px] text-muted truncate">
            orig: {track.suno.title}
          </p>
        )}
        <p className="text-[10px] text-muted mt-0.5">
          {track.bpm ? `${track.bpm} BPM` : "—"} ·{" "}
          {formatDuration(track.duration_sec)}
        </p>
      </div>
    </button>
  );
}

function TrackDetail({
  track,
  onClose,
}: {
  track: Track;
  onClose: () => void;
}) {
  const suno = track.suno ?? {};
  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 flex items-stretch justify-end animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-surface border-l border-border overflow-y-auto p-6 animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-lg text-[#e2e2ff] font-bold">
              {track.display_name}
            </h2>
            {suno.disambiguated && suno.title && (
              <p className="text-xs text-muted mt-1">
                Original Suno title: <span className="text-purple">{suno.title}</span>
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-muted hover:text-danger text-sm"
          >
            ✕
          </button>
        </div>

        {suno.cover_url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={suno.cover_url}
            alt={track.display_name}
            className="w-full rounded border border-border mb-4"
          />
        )}

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs mb-4">
          <Field label="Genre" value={track.genre_folder ?? track.genre} />
          <Field label="BPM" value={track.bpm?.toString()} />
          <Field label="Key" value={track.camelot_key} />
          <Field label="Duration" value={formatDuration(track.duration_sec)} />
          <Field label="Artist" value={suno.artist} />
          <Field label="Year" value={suno.year} />
        </dl>

        {suno.prompt && (
          <Section label="Prompt">
            <p className="text-xs text-[#e2e2ff] leading-relaxed whitespace-pre-wrap">
              {suno.prompt}
            </p>
          </Section>
        )}

        {suno.tags && suno.tags !== suno.prompt && (
          <Section label="Tags">
            <p className="text-xs text-muted leading-relaxed">{suno.tags}</p>
          </Section>
        )}

        {suno.lyrics && (
          <Section label="Lyrics">
            <pre className="text-xs text-[#e2e2ff] whitespace-pre-wrap font-mono">
              {suno.lyrics}
            </pre>
          </Section>
        )}

        {track.file && (
          <p className="text-[10px] text-muted mt-6 break-all">
            <span className="text-muted">file: </span>
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
    <div className="border-b border-border pb-1">
      <dt className="text-[10px] text-muted uppercase tracking-wider">
        {label}
      </dt>
      <dd className="text-[#e2e2ff]">{value ?? "—"}</dd>
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
    <section className="mb-4">
      <h3 className="text-[10px] text-muted uppercase tracking-wider mb-1">
        {label}
      </h3>
      {children}
    </section>
  );
}
