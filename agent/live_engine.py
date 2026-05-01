"""
LiveEngine — real-time two-deck audio engine for Apollo LiveDJ.

Architecture:
  - sounddevice.OutputStream callback: fills audio buffer from pre-loaded numpy arrays.
  - Watchdog thread: monitors playback position, fires events, triggers crossfades.
  - Pre-stretch thread: time-stretches the next track in the background while
    the current track plays, so crossfades are instant.

All public methods are thread-safe (protected by self._lock).
Events are pushed to a threading.Queue shared with the LiveDJ agent.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import soundfile as sf

# sounddevice requires PortAudio — guarded so the module can be imported in
# headless / CI environments without audio hardware.
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except OSError:  # PortAudio library not found
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False

try:
    import librosa as _librosa
    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False

try:
    import pyrubberband as _pyrubberband
    _HAS_PYRUBBERBAND = True
except ImportError:  # pragma: no cover
    _HAS_PYRUBBERBAND = False

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------
TRACK_STARTED       = "track_started"
APPROACHING_CF      = "approaching_crossfade"
CROSSFADE_TRIGGERED = "crossfade_triggered"
CROSSFADE_FINISHED  = "crossfade_finished"
TRACK_ENDED         = "track_ended"
SESSION_ENDED       = "session_ended"

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE        = 44100
_CHANNELS           = 2
_BLOCK_SIZE         = 2048
_BPM_THRESHOLD      = 5      # min BPM diff to trigger time-stretch
_STRETCH_MAX        = 1.5    # safety ceiling (v1.3 bound)
_STRETCH_MIN        = 1.0 / _STRETCH_MAX

_PROJECT_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# LiveEngine
# ---------------------------------------------------------------------------

class LiveEngine:
    """Two-deck real-time DJ engine.

    Parameters
    ----------
    playlist:
        List of track dicts from context_variables["playlist"].
    event_queue:
        threading.Queue shared with the LiveDJ agent loop.
    crossfade_sec:
        Crossfade blend duration in seconds (default 12).
    approach_warn_sec:
        How many seconds before the crossfade point to fire APPROACHING_CF (default 30).
    """

    def __init__(
        self,
        playlist: list[dict],
        event_queue: Queue,
        crossfade_sec: int = 12,
        approach_warn_sec: int = 30,
    ) -> None:
        self.playlist = list(playlist)
        self.event_queue = event_queue
        self.crossfade_sec = crossfade_sec
        self.approach_warn_sec = approach_warn_sec

        # State
        self._state = "idle"  # idle | playing | crossfading | ended
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Audio buffers (float32 stereo numpy arrays)
        self._audio: np.ndarray | None = None        # current track
        self._next_audio: np.ndarray | None = None   # pre-stretched next track
        self._pos: int = 0       # sample index into _audio
        self._cf_start: int = 0  # sample where crossfade started
        self._next_pos: int = 0  # sample index into _next_audio during crossfade
        self._in_point: int = 0  # start offset in _next_audio (from hot cue IN)

        # Playlist tracking
        self._idx: int = 0
        self._extend_samples: int = 0  # extra samples before auto-crossfade

        # Watchdog signals
        self._cf_just_finished: bool = False  # set by callback, cleared by watchdog
        self._prev_idx: int = 0

        # Threads
        self._stream: sd.OutputStream | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._prestretch_thread: threading.Thread | None = None
        self._prestretch_ready = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start live playback from the first track."""
        if not self.playlist:
            self._emit(SESSION_ENDED)
            return

        self._audio = self._load_audio(self.playlist[0])
        self._pos = 0
        self._idx = 0
        self._prev_idx = 0
        self._state = "playing"

        if not _SD_AVAILABLE or sd is None:
            raise RuntimeError(
                "sounddevice / PortAudio not available. "
                "Install PortAudio (e.g. 'apt install libportaudio2') to use live mode."
            )

        self._stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
            blocksize=_BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._emit(TRACK_STARTED, track=self.playlist[0])

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="live-watchdog"
        )
        self._watchdog_thread.start()

        if len(self.playlist) > 1:
            self._start_prestretch(0, 1)

    def crossfade_now(self) -> str:
        """Trigger crossfade immediately, skipping the auto-timer."""
        with self._lock:
            if self._state != "playing":
                return f"Cannot crossfade: engine is '{self._state}'."
            if self._idx + 1 >= len(self.playlist):
                return "No next track to crossfade into."
            # Advance position to the crossfade point so watchdog fires on next tick
            cf = self._cf_point_samples(self.playlist[self._idx])
            self._pos = max(self._pos, cf)
        return "Crossfade triggered."

    def extend_track(self, seconds: int) -> str:
        """Delay the upcoming auto-crossfade by `seconds` seconds."""
        with self._lock:
            self._extend_samples += int(seconds * _SAMPLE_RATE)
        return f"Crossfade delayed by {seconds}s."

    def skip_track(self) -> str:
        """Hard-cut to the next track without crossfade."""
        with self._lock:
            next_idx = self._idx + 1
            if next_idx >= len(self.playlist):
                return "No next track."
            next_audio = (
                self._next_audio
                if self._next_audio is not None
                else self._load_audio(self.playlist[next_idx])
            )
            self._audio = next_audio
            self._pos = self._in_point
            self._next_audio = None
            self._idx = next_idx
            self._extend_samples = 0
            self._state = "playing"
        self._emit(TRACK_STARTED, track=self.playlist[next_idx])
        if next_idx + 1 < len(self.playlist):
            self._start_prestretch(next_idx, next_idx + 1)
        return f"Skipped to '{self.playlist[next_idx]['display_name']}'."

    def queue_swap(self, position: int, track_id: str) -> str:
        """Replace a future playlist position with a catalog track."""
        idx = position - 1
        with self._lock:
            if idx <= self._idx or idx >= len(self.playlist):
                return f"Position {position} is not a future slot."
        catalog = _load_catalog()
        track = next((t for t in catalog if t["id"] == track_id), None)
        if not track:
            return f"Track ID '{track_id}' not found in catalog."
        with self._lock:
            self.playlist[idx] = track
        return f"Queued '{track['display_name']}' at position {position}."

    def set_crossfade_point(self, position_sec: float) -> str:
        """Manually set where the crossfade begins in the current track."""
        with self._lock:
            if self._audio is None:
                return "No track playing."
            target = int(position_sec * _SAMPLE_RATE)
            current_cf = self._cf_point_samples(self.playlist[self._idx])
            self._extend_samples += target - current_cf
        return f"Crossfade point set to {position_sec:.1f}s."

    def get_state(self) -> dict:
        """Return a snapshot of engine state for the agent."""
        with self._lock:
            idx = self._idx
            pos = self._pos
            state = self._state
            audio_len = len(self._audio) if self._audio is not None else 0

        pos_sec = pos / _SAMPLE_RATE
        track = self.playlist[idx] if idx < len(self.playlist) else None
        next_track = self.playlist[idx + 1] if idx + 1 < len(self.playlist) else None

        if track and audio_len:
            with self._lock:
                cf_sec = self._cf_point_samples(track) / _SAMPLE_RATE
            secs_to_cf = max(0.0, cf_sec - pos_sec)
        else:
            secs_to_cf = 0.0

        return {
            "state": state,
            "position_sec": round(pos_sec, 1),
            "current_track": _track_summary(track),
            "next_track": _track_summary(next_track),
            "seconds_to_crossfade": round(secs_to_cf, 1),
            "playlist_remaining": len(self.playlist) - idx - 1,
        }

    def stop(self) -> None:
        """Stop playback and release audio resources."""
        self._stop_event.set()
        with self._lock:
            self._state = "idle"
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Audio callback (runs in sounddevice's low-latency thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        with self._lock:
            if self._audio is None or self._state == "idle":
                outdata[:] = 0
                return

            if self._state == "playing":
                end = self._pos + frames
                chunk = self._audio[self._pos : end]
                n = len(chunk)
                outdata[:n] = chunk
                if n < frames:
                    outdata[n:] = 0
                self._pos += n

            elif self._state == "crossfading":
                cf_elapsed = self._pos - self._cf_start
                cf_len = int(self.crossfade_sec * _SAMPLE_RATE)
                remaining = cf_len - cf_elapsed
                n = min(frames, max(0, remaining))

                if n > 0 and self._next_audio is not None:
                    o_end = min(self._pos + n, len(self._audio))
                    i_end = min(self._next_pos + n, len(self._next_audio))
                    out_chunk = self._audio[self._pos : o_end]
                    in_chunk = self._next_audio[self._next_pos : i_end]
                    actual = min(len(out_chunk), len(in_chunk))
                    if actual > 0:
                        t = np.linspace(
                            cf_elapsed / cf_len,
                            (cf_elapsed + actual) / cf_len,
                            actual, endpoint=False,
                        ).reshape(-1, 1).astype(np.float32)
                        outdata[:actual] = out_chunk[:actual] * (1.0 - t) + in_chunk[:actual] * t
                        if actual < frames:
                            outdata[actual:] = 0
                        self._pos += actual
                        self._next_pos += actual
                    else:
                        outdata[:] = 0
                else:
                    outdata[:] = 0

                # Crossfade complete: swap tracks
                if remaining <= frames:
                    self._state = "playing"
                    self._audio = self._next_audio
                    self._pos = self._next_pos
                    self._next_audio = None
                    self._extend_samples = 0
                    self._cf_just_finished = True  # watchdog will emit events

    # ------------------------------------------------------------------
    # Watchdog thread
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        approached = False
        cf_triggered = False

        while not self._stop_event.is_set():
            time.sleep(0.05)  # 50 ms granularity

            with self._lock:
                state = self._state
                pos = self._pos
                idx = self._idx
                audio_len = len(self._audio) if self._audio is not None else 0
                cf_just_finished = self._cf_just_finished
                if cf_just_finished:
                    self._cf_just_finished = False

            if state == "idle":
                continue

            # ── Crossfade finished: emit events, advance bookkeeping ─────────
            if cf_just_finished:
                prev_track = self.playlist[self._prev_idx]
                cur_track = self.playlist[idx] if idx < len(self.playlist) else None
                self._emit(CROSSFADE_FINISHED, from_track=prev_track, to_track=cur_track)
                self._emit(TRACK_ENDED, track=prev_track)

                if cur_track:
                    self._emit(TRACK_STARTED, track=cur_track)
                    approached = False
                    cf_triggered = False
                    self._prev_idx = idx
                    if idx + 1 < len(self.playlist):
                        self._start_prestretch(idx, idx + 1)
                else:
                    self._emit(SESSION_ENDED)
                    return
                continue

            if idx >= len(self.playlist):
                self._emit(SESSION_ENDED)
                return

            with self._lock:
                cf_samples = self._cf_point_samples(self.playlist[idx])
            cf_sec = cf_samples / _SAMPLE_RATE
            pos_sec = pos / _SAMPLE_RATE

            # ── APPROACHING_CF warning ───────────────────────────────────────
            if not approached and pos_sec >= cf_sec - self.approach_warn_sec:
                next_idx = idx + 1
                self._emit(
                    APPROACHING_CF,
                    track=self.playlist[idx],
                    next_track=self.playlist[next_idx] if next_idx < len(self.playlist) else None,
                    seconds_remaining=round(max(0.0, cf_sec - pos_sec), 1),
                )
                approached = True

            # ── Trigger crossfade ────────────────────────────────────────────
            if not cf_triggered and pos >= cf_samples:
                next_idx = idx + 1
                if next_idx < len(self.playlist):
                    self._prestretch_ready.wait(timeout=3.0)
                    with self._lock:
                        if self._next_audio is not None and self._state == "playing":
                            self._cf_start = pos
                            self._next_pos = self._in_point
                            self._idx = next_idx
                            self._state = "crossfading"
                            cf_triggered = True
                    if cf_triggered:
                        self._emit(
                            CROSSFADE_TRIGGERED,
                            from_track=self.playlist[idx],
                            to_track=self.playlist[next_idx],
                        )
                else:
                    # Last track — let it play to the end
                    if pos >= audio_len:
                        self._emit(TRACK_ENDED, track=self.playlist[idx])
                        self._emit(SESSION_ENDED)
                        return

    # ------------------------------------------------------------------
    # Pre-stretch thread
    # ------------------------------------------------------------------

    def _start_prestretch(self, current_idx: int, next_idx: int) -> None:
        if self._prestretch_thread and self._prestretch_thread.is_alive():
            return
        self._prestretch_ready.clear()
        self._prestretch_thread = threading.Thread(
            target=self._prestretch_worker,
            args=(current_idx, next_idx),
            daemon=True,
            name="live-prestretch",
        )
        self._prestretch_thread.start()

    def _prestretch_worker(self, current_idx: int, next_idx: int) -> None:
        if next_idx >= len(self.playlist):
            return
        current_track = self.playlist[current_idx]
        next_track = self.playlist[next_idx]

        audio = self._load_audio(next_track)
        audio = self._time_stretch(audio, next_track, current_track)

        in_pt = self._in_point_of(next_track)
        # Trim to in-point
        audio_trimmed = audio[in_pt:] if in_pt < len(audio) else audio

        with self._lock:
            self._next_audio = audio_trimmed
            self._in_point = 0  # already trimmed to in-point
        self._prestretch_ready.set()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _load_audio(self, track: dict) -> np.ndarray:
        """Load a track WAV as float32 stereo at _SAMPLE_RATE."""
        rel = track.get("file", "")
        path = (_PROJECT_DIR / rel) if rel and not Path(rel).is_absolute() else Path(rel)
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # Ensure stereo
        if audio.shape[1] == 1:
            audio = np.hstack([audio, audio])
        audio = audio[:, :2]
        # Resample if needed
        if sr != _SAMPLE_RATE and _HAS_LIBROSA:
            audio = _librosa.resample(audio.T, orig_sr=sr, target_sr=_SAMPLE_RATE).T
        return audio.astype(np.float32)

    def _time_stretch(
        self, audio: np.ndarray, track: dict, target_track: dict
    ) -> np.ndarray:
        """Stretch `audio` so its BPM matches the current track's BPM."""
        if not _HAS_PYRUBBERBAND:
            return audio
        from_bpm = float(track.get("bpm") or 0)
        to_bpm = float(target_track.get("bpm") or 0)
        if from_bpm <= 0 or to_bpm <= 0:
            return audio
        if abs(from_bpm - to_bpm) <= _BPM_THRESHOLD:
            return audio
        ratio = to_bpm / from_bpm
        ratio = max(_STRETCH_MIN, min(_STRETCH_MAX, ratio))
        stretched = _pyrubberband.time_stretch(audio, _SAMPLE_RATE, ratio)
        return stretched.astype(np.float32)

    def _cf_point_samples(self, track: dict) -> int:
        """Return sample index where crossfade should begin, respecting hot cues + extensions."""
        cues = track.get("hot_cues", [])
        out_cues = [c for c in cues if c.get("type") == "out"]
        if out_cues:
            sec = float(out_cues[0]["position_sec"])
        else:
            duration = float(track.get("duration_sec") or (len(self._audio) / _SAMPLE_RATE))
            sec = max(0.0, duration - self.crossfade_sec - 5)
        return int(sec * _SAMPLE_RATE) + self._extend_samples

    @staticmethod
    def _in_point_of(track: dict) -> int:
        """Return sample offset for the IN hot cue, or 0."""
        cues = track.get("hot_cues", [])
        in_cues = [c for c in cues if c.get("type") == "in"]
        return int(in_cues[0]["position_sec"] * _SAMPLE_RATE) if in_cues else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, type_: str, **kwargs) -> None:
        self.event_queue.put({"type": type_, **kwargs})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_catalog() -> list[dict]:
    catalog_path = _PROJECT_DIR / "tracks" / "tracks.json"
    if not catalog_path.exists():
        return []
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f).get("tracks", [])


def _track_summary(track: dict | None) -> dict | None:
    if not track:
        return None
    return {
        "display_name": track.get("display_name", "?"),
        "bpm": track.get("bpm", 0),
        "camelot_key": track.get("camelot_key", "?"),
        "hot_cues": track.get("hot_cues", []),
    }
