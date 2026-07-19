import tempfile
from pathlib import Path

import pytest

from kiui.agent.storage import (
    allocated_size,
    clean_storage,
    cleanable_entries,
    format_size,
    storage_entries,
)


def _symlinks_supported() -> bool:
    """Windows requires elevated privileges (or Developer Mode) for symlinks."""
    with tempfile.TemporaryDirectory() as d:
        try:
            Path(d, "link").symlink_to(Path(d, "target"))
            return True
        except OSError:
            return False


def _write(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_storage_entries_measure_each_top_level_entry(tmp_path):
    _write(tmp_path / ".kia" / "sessions" / "one.json")
    _write(tmp_path / ".kia" / "history")

    entries = storage_entries(tmp_path)

    assert [entry.name for entry in entries] == ["history", "sessions"]
    assert {entry.name: entry.size for entry in entries} == {
        "history": allocated_size(tmp_path / ".kia" / "history"),
        "sessions": allocated_size(tmp_path / ".kia" / "sessions"),
    }


def test_storage_does_not_create_missing_kia_directory(tmp_path):
    assert storage_entries(tmp_path) == []
    assert not (tmp_path / ".kia").exists()


def test_clean_storage_preserves_skills_and_unknown_entries(tmp_path):
    _write(tmp_path / ".kia" / "sessions" / "one.json")
    _write(tmp_path / ".kia" / "tool-results" / "one" / "result.txt")
    _write(tmp_path / ".kia" / "history")
    _write(tmp_path / ".kia" / "skills" / "custom" / "SKILL.md")
    _write(tmp_path / ".kia" / "notes")
    expected = sum(entry.size for entry in cleanable_entries(tmp_path))

    removed = clean_storage(tmp_path)

    assert removed == expected
    assert not (tmp_path / ".kia" / "sessions").exists()
    assert not (tmp_path / ".kia" / "tool-results").exists()
    assert not (tmp_path / ".kia" / "history").exists()
    assert (tmp_path / ".kia" / "skills" / "custom" / "SKILL.md").exists()
    assert (tmp_path / ".kia" / "notes").exists()


@pytest.mark.skipif(
    not _symlinks_supported(), reason="symlinks not permitted on this system"
)
def test_clean_storage_unlinks_top_level_symlinks_without_following_them(tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "keep.txt")
    kia = tmp_path / ".kia"
    kia.mkdir()
    (kia / "sessions").symlink_to(outside, target_is_directory=True)

    clean_storage(tmp_path)

    assert not (kia / "sessions").exists()
    assert (outside / "keep.txt").exists()


def test_format_size():
    assert format_size(0) == "0 B"
    assert format_size(1024) == "1.0 KiB"
    assert format_size(1024**2) == "1.0 MiB"
