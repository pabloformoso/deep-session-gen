"""Cover art for tracks Apollo generates itself.

The catalog page has always rendered a cover: 510 of the 513 tracks
carry ``suno.cover_url`` from their Suno import. The three blanks were
exactly the tracks made at home — the two `Velvet Corridor` takes and
`AChillE` — which is every track the publish endpoint creates. The slot
existed and was proven; only the local half was missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "web") not in sys.path:
    sys.path.insert(0, str(ROOT / "web"))

from backend import covers  # noqa: E402


# --- id safety: the id becomes a filename, so it is untrusted ------------

@pytest.mark.parametrize(
    "bad",
    [
        "",
        ".",
        "..",
        "../../etc/passwd",
        "healing/../../secret",
        "with space",
        "semi;colon",
        "sub/dir",
        "back\\slash",
        "null\x00byte",
    ],
)
def test_unsafe_track_ids_are_refused(bad):
    assert covers.is_safe_track_id(bad) is False
    assert covers.cover_path(bad) is None


@pytest.mark.parametrize(
    "good", ["healing--achille", "deep-house--velvet-corridor-v2", "a.b_c-1"]
)
def test_real_catalog_shaped_ids_are_accepted(good):
    assert covers.is_safe_track_id(good) is True
    assert covers.cover_path(good) is not None


def test_cover_path_stays_inside_the_cover_directory():
    p = covers.cover_path("healing--achille")
    assert p is not None
    assert p.parent == covers.cover_dir()
    assert p.name == "healing--achille.png"


def test_generate_cover_refuses_an_unsafe_id_without_calling_azure(monkeypatch):
    """An unsafe id must be rejected BEFORE any import or network work."""
    def explode(*_a, **_kw):  # pragma: no cover — must never run
        raise AssertionError("generation attempted for an unsafe id")

    monkeypatch.setattr(covers.importlib, "import_module", explode)
    assert covers.generate_cover("../escape", "Name", "healing") is None


# --- presence / URL -------------------------------------------------------

def test_has_cover_is_false_when_the_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)
    assert covers.has_cover("nothing-here") is False
    assert covers.cover_url_for("nothing-here") is None


def test_cover_url_is_relative_so_it_works_behind_dev_proxy_and_nginx(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)
    (tmp_path / "healing--achille.png").write_bytes(b"\x89PNG")
    url = covers.cover_url_for("healing--achille")
    assert url == "/api/tracks/healing--achille/cover"
    assert not url.startswith("http")


def test_cover_url_for_is_none_for_an_unsafe_id(tmp_path, monkeypatch):
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)
    assert covers.cover_url_for("../../etc/passwd") is None


# --- never fatal ----------------------------------------------------------

def test_generation_failure_returns_none_instead_of_raising(
    tmp_path, monkeypatch
):
    """A publish that already succeeded must not be reported as failed."""
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)

    class Boom:
        GENRE_THEMES: dict = {}

        @staticmethod
        def _generate_artwork(*_a, **_kw):
            raise RuntimeError("azure said no")

    monkeypatch.setattr(covers.importlib, "import_module", lambda _n: Boom)
    assert covers.generate_cover("healing--x", "X", "healing") is None


def test_an_existing_cover_is_not_regenerated(tmp_path, monkeypatch):
    """Publishing the same id twice must not buy a second image."""
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)
    (tmp_path / "healing--x.png").write_bytes(b"\x89PNG")

    def explode(*_a, **_kw):  # pragma: no cover — must never run
        raise AssertionError("regenerated an existing cover")

    monkeypatch.setattr(covers.importlib, "import_module", explode)
    assert covers.generate_cover("healing--x", "X", "healing") == str(
        tmp_path / "healing--x.png"
    )


def test_the_genre_theme_reaches_the_artwork_call(tmp_path, monkeypatch):
    """A cover must use its genre's visual language, not the silent default."""
    monkeypatch.setattr(covers, "cover_dir", lambda: tmp_path)
    seen = {}

    class Fake:
        GENRE_THEMES = {"synthware": {"artwork_style": "dark-techno"}}

        @staticmethod
        def _generate_artwork(name, _dir, theme, cache_name=None):
            seen.update(name=name, theme=theme, cache_name=cache_name)
            return "/tmp/x.png"

    monkeypatch.setattr(covers.importlib, "import_module", lambda _n: Fake)
    covers.generate_cover("synthware--neon", "Neon Rain", "Synthware")

    assert seen["theme"] == {"artwork_style": "dark-techno"}
    # The PROMPT gets the pretty name, the FILENAME gets the id.
    assert seen["name"] == "Neon Rain"
    assert seen["cache_name"] == "synthware--neon"
