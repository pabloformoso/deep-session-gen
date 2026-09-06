"""Quality bench runner + reference extraction (S-1 / issue #71).

- extract_references(): deterministic sweep over catalog WAVs (fixed sort,
  first N per genre) -> reference ranges JSON. Genre->folder map is explicit:
  lofi AND ambient share `tracks/lofi - ambient`.
- run_bench(): render a session, compute all metrics, compare against the
  references, write report.{json,md} + WAV. Two tiers: reference_informed
  failures gate --strict; advisory failures only print.
- bench_wav(): same verdict for audio this bench did NOT render — the
  algorave/Strudel lane hands us a finished WAV and no spec sequence, so
  the symbolic half is simply absent. Same references, same margins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .genres import GENRE_PACKS
from .quality import (
    NORM_TARGET_LUFS,
    analyze_wav,
    load_references,
    session_report,
)
from .render_audio import SR, render_audio
from .spec import PatternSpec

GENRE_FOLDERS = {
    # Bench key -> catalog folder. The key must match the folder's LEADING
    # text, because `_reference_genre` resolves a folder by that rule (see
    # web/backend/generator.py) — "cocktail" claims "cocktail house".
    #
    # Extended 2026-09-05 from 2 folders to 7, once the catalog had 512 tracks
    # to measure. Before that a genre with no references made `Score` answer
    # "no verdict", which reads as a broken button rather than as a missing
    # measurement.
    "lofi": "lofi - ambient",
    "ambient": "lofi - ambient",
    "deep": "deep house",
    "healing": "Healing",
    "aural": "aural",
    "cocktail": "cocktail house",
    "soul": "soul jazz",
    "synthware": "synthware",
    # "chillout" is deliberately NOT here yet. `test_catalog_references_
    # committed_and_complete` requires every key in this map to have measured
    # references, and a genre with no WAVs cannot have any. Add it in the same
    # change that gives it music, and re-run scripts/extract_quality_references.
}

REFERENCES_PATH = Path(__file__).parent / "quality_references.json"

# reference_informed margins: the render is sparse even after loudness
# normalization, so ranges are generous by design — the tier catches
# gross wrongness (screeching brightness, white-noise tilt), not taste.
CENTROID_RATIO_MAX = 2.5   # render centroid within [ref_min/R, ref_max*R]
TILT_DELTA_MAX = 8.0       # dB/oct beyond the reference range
NOVELTY_MAX = 0.95         # consecutive phrases sharing ~nothing = mode chaos


class BenchInputError(ValueError):
    """Something the operator handed the bench is wrong (file, genre, refs).

    Raised instead of letting soundfile/json blow up so the CLI can turn it
    into a one-line message rather than a traceback.
    """


def extract_references(genre_dirs: dict[str, Path], n: int = 5, sr_limit: int | None = None) -> dict:
    """Analyze the first `n` WAVs (sorted by name) per genre. Deterministic."""
    refs: dict = {}
    for genre, folder in sorted(genre_dirs.items()):
        files = sorted(Path(folder).glob("*.wav"))[:n]
        if not files:
            continue
        centroids, tilts, lufs_vals = [], [], []
        for path in files:
            audio, sr = sf.read(str(path), always_2d=True)
            mono = audio.mean(axis=1)
            if sr_limit and len(mono) > sr * sr_limit:
                mono = mono[: sr * sr_limit]
            metrics = analyze_wav(mono.astype(np.float32), sr)
            ri, adv = metrics["reference_informed"], metrics["advisory"]
            if ri["centroid_hz"] is not None:
                centroids.append(ri["centroid_hz"])
            if ri["tilt_db_per_oct"] is not None:
                tilts.append(ri["tilt_db_per_oct"])
            if adv["lufs"] is not None:
                lufs_vals.append(adv["lufs"])
        refs[genre] = {
            "files": [p.name for p in files],
            "norm_target_lufs": NORM_TARGET_LUFS,
            "centroid_hz": {"min": round(min(centroids), 1), "max": round(max(centroids), 1)},
            "tilt_db_per_oct": {"min": round(min(tilts), 2), "max": round(max(tilts), 2)},
            "advisory_lufs": {"min": round(min(lufs_vals), 1), "max": round(max(lufs_vals), 1)},
        }
    return refs


def _check_reference_informed(genre_ref: dict, audio_metrics: dict, phrase_novelties: list[float]) -> list[str]:
    failures = []
    ri = audio_metrics["reference_informed"]
    c = ri["centroid_hz"]
    if c is not None and genre_ref:
        lo = genre_ref["centroid_hz"]["min"] / CENTROID_RATIO_MAX
        hi = genre_ref["centroid_hz"]["max"] * CENTROID_RATIO_MAX
        if not lo <= c <= hi:
            failures.append(f"centroid {c:.0f}Hz outside [{lo:.0f}, {hi:.0f}]")
    t = ri["tilt_db_per_oct"]
    if t is not None and genre_ref:
        lo = genre_ref["tilt_db_per_oct"]["min"] - TILT_DELTA_MAX
        hi = genre_ref["tilt_db_per_oct"]["max"] + TILT_DELTA_MAX
        if not lo <= t <= hi:
            failures.append(f"tilt {t:.1f}dB/oct outside [{lo:.1f}, {hi:.1f}]")
    for i, nov in enumerate(phrase_novelties):
        if nov > NOVELTY_MAX:
            failures.append(f"novelty {nov} > {NOVELTY_MAX} at phrase {i + 2} (mode chaos)")
    return failures


def run_bench(genre: str, phrases: int = 2, seed: int = 0, out_dir=None,
              specs: list[PatternSpec] | None = None,
              references_path=REFERENCES_PATH) -> tuple[dict, bool]:
    """Render + measure + compare. Returns (report, reference_informed_passed)."""
    if specs is None:
        starter = PatternSpec.from_dict(GENRE_PACKS[genre]["starter"])
        specs = [starter] * phrases

    chunks = [render_audio(s, seed + i) for i, s in enumerate(specs)]
    audio = np.concatenate(chunks)
    audio_metrics = analyze_wav(audio)
    symbolic = session_report(specs, seed)
    novelties = [p["novelty_vs_prev"] for p in symbolic["phrases"] if "novelty_vs_prev" in p]

    references = load_references(references_path) if Path(references_path).exists() else {}
    genre_ref = references.get(genre, {})
    failures = _check_reference_informed(genre_ref, audio_metrics, novelties)

    report = {
        "genre": genre,
        "seed": seed,
        "phrases": symbolic["phrases"],
        "audio": audio_metrics,
        "reference": genre_ref,
        "reference_informed_failures": failures,
        "passed": not failures,
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sf.write(str(out / "session.wav"), audio, SR)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return report, not failures


def load_wav_mono(path, sr: int = SR) -> tuple[np.ndarray, int, int]:
    """Read any WAV as a mono buffer at the bench SR.

    Stereo collapses by channel mean and a foreign rate is resampled, so
    a 48 kHz stereo render scores on the same footing as the mono numpy
    one. Returns (mono, source_rate, source_channels) — the CLI reports
    what it had to convert, since that explains a surprising centroid.
    """
    file = Path(path)
    if not file.is_file():
        raise BenchInputError(f"WAV not found: {file}")
    try:
        audio, file_sr = sf.read(str(file), always_2d=True)
    except RuntimeError as exc:                    # soundfile.LibsndfileError
        raise BenchInputError(f"cannot read {file} as audio: {exc}") from exc
    if audio.shape[0] == 0:
        raise BenchInputError(f"{file} has no audio frames")
    channels = audio.shape[1]
    mono = audio.mean(axis=1).astype(np.float32)
    if file_sr != sr:
        import librosa                             # heavy; only the WAV path needs it

        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr).astype(np.float32)
    return mono, int(file_sr), int(channels)


def bench_wav(wav_path, genre: str, out_dir=None,
              references_path=REFERENCES_PATH) -> tuple[dict, bool]:
    """Score a WAV this bench did not render. Returns (report, passed).

    The audio half is identical to run_bench's — same analyze_wav, same
    reference bands, same margins — so an external render is judged by the
    rule the generative engine is already judged by. The symbolic half is
    empty: no specs, no novelty, nothing to be honest about there.
    """
    refs_file = Path(references_path)
    if not refs_file.is_file():
        raise BenchInputError(f"references file not found: {refs_file}")
    try:
        references = load_references(refs_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BenchInputError(f"references file {refs_file} is not valid JSON: {exc}") from exc
    genre_ref = references.get(genre)
    if not genre_ref:
        known = ", ".join(sorted(references)) or "none"
        raise BenchInputError(f"no references for genre '{genre}' in {refs_file} (has: {known})")

    mono, file_sr, channels = load_wav_mono(wav_path)
    audio_metrics = analyze_wav(mono, SR)
    failures = _check_reference_informed(genre_ref, audio_metrics, [])

    report = {
        "genre": genre,
        "wav": str(Path(wav_path)),
        "source_sample_rate": file_sr,
        "source_channels": channels,
        "duration_sec": round(len(mono) / SR, 2),
        "phrases": [],
        "audio": audio_metrics,
        "reference": genre_ref,
        "reference_informed_failures": failures,
        "passed": not failures,
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    return report, not failures


def to_markdown(report: dict) -> str:
    source = f"wav {Path(report['wav']).name}" if "wav" in report else f"seed {report['seed']}"
    lines = [f"# Quality bench — {report['genre']} ({source})", ""]
    adv, ri = report["audio"]["advisory"], report["audio"]["reference_informed"]
    lines += ["## Audio",
              f"- LUFS {adv['lufs']} · LRA {adv['lra']} · crest {adv['crest_db']} dB *(advisory)*",
              f"- centroid {ri['centroid_hz']} Hz · tilt {ri['tilt_db_per_oct']} dB/oct "
              f"*(reference_informed, normalized to {NORM_TARGET_LUFS} LUFS)*"]
    if "wav" in report:
        lines.append(f"- source {report['source_channels']}ch @ {report['source_sample_rate']} Hz "
                     f"· {report['duration_sec']}s *(downmixed/resampled to {SR} Hz mono)*")
    lines.append("")
    if report["phrases"]:
        lines.append("## Phrases")
        for i, p in enumerate(report["phrases"]):
            nov = f" · novelty {p['novelty_vs_prev']}" if "novelty_vs_prev" in p else ""
            lines.append(f"{i + 1}. energy {p['energy']}{nov} · density {p['note_density']}")
            lines.append(f"   {p['reason']}")
        lines.append("")
    lines += ["## Verdict",
              "PASS" if report["passed"] else "FAIL: " + "; ".join(report["reference_informed_failures"])]
    return "\n".join(lines) + "\n"
