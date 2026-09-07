"""Cover art for catalog tracks.

The catalog page has always rendered a cover — 510 of the 513 tracks
carry one, because they were imported from Suno and came with
``suno.cover_url`` pointing at Suno's CDN. The three that show a blank
tile are precisely the ones produced at home: the two `Velvet Corridor`
takes and `AChillE`. So the UI slot exists and is proven; what is
missing is a cover for anything Apollo generates itself.

This module fills that slot from the SAME artwork pipeline the video
renderer uses, so a generated track's cover matches the visual language
of its genre instead of being a second, unrelated style system.

Design notes:

* **Keyed by track id, not display name.** Session artwork dedups on
  display name on purpose (two tracks with one name share an image).
  A catalog cover cannot: ids are unique and filesystem-safe, display
  names are neither — `Velvet Corridor` exists twice, and a name may
  contain a slash. ``_generate_artwork(cache_name=...)`` separates the
  filename from the prompt so the image is still ABOUT the song.
* **Never fatal.** A publish that produced a catalog track must not be
  reported as failed because an image call did not come back, exactly
  as ``_record_publish`` is never allowed to turn a successful publish
  into a failure. Everything here returns ``None`` instead of raising.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

#: Covers live beside session artwork but in their own directory, so a
#: `--build-catalog` or a session render never collides with them.
COVER_DIR_NAME = "catalog"

#: Track ids are slugs, but this is the value that becomes a filesystem
#: path, so it is validated rather than trusted.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cover_dir() -> Path:
    return _repo_root() / "artwork" / COVER_DIR_NAME


def is_safe_track_id(track_id: str) -> bool:
    """True when ``track_id`` can be used as a bare filename.

    Rejects anything with a separator or a traversal component — the id
    reaches this module from a URL path, so it is untrusted input.
    """
    return bool(track_id) and bool(_SAFE_ID.match(track_id)) and track_id not in {".", ".."}


def cover_path(track_id: str) -> Path | None:
    """Absolute path of ``track_id``'s cover, or ``None`` for an unsafe id.

    The file need not exist — callers check that themselves so they can
    tell "no cover yet" from "bad id".
    """
    if not is_safe_track_id(track_id):
        return None
    return cover_dir() / f"{track_id}.png"


def has_cover(track_id: str) -> bool:
    p = cover_path(track_id)
    return bool(p and p.is_file())


def generate_cover(track_id: str, display_name: str, genre_folder: str) -> str | None:
    """Render and cache a cover for one catalog track.

    Returns the file path on success, ``None`` when it was skipped (no
    Azure image deployment configured, unsafe id) or when the image call
    failed. Never raises: the caller is a publish that has already
    succeeded.

    Imports ``main`` lazily — it is a ~4700-line module and importing it
    at module scope would drag the whole render pipeline into every
    process that merely serves the API.
    """
    path = cover_path(track_id)
    if path is None:
        print(f"[covers] refusing unsafe track id {track_id!r}", flush=True)
        return None
    if path.is_file():
        return str(path)
    try:
        main = importlib.import_module("main")
        theme = main.GENRE_THEMES.get((genre_folder or "").strip().lower())
        return main._generate_artwork(
            display_name or track_id,
            str(cover_dir()),
            theme,
            cache_name=track_id,
        )
    except Exception as exc:  # noqa: BLE001 — a cover is never worth a 500
        print(f"[covers] cover generation failed for {track_id}: {exc}", flush=True)
        return None


def cover_url_for(track_id: str) -> str | None:
    """Relative API URL for an EXISTING cover, or ``None``.

    Relative on purpose: the frontend reaches the backend through a
    proxy in dev and through nginx in prod, so an absolute URL built
    server-side would be wrong in one of them.
    """
    if not has_cover(track_id):
        return None
    return f"/api/tracks/{track_id}/cover"


__all__ = [
    "cover_dir",
    "cover_path",
    "cover_url_for",
    "generate_cover",
    "has_cover",
    "is_safe_track_id",
]
