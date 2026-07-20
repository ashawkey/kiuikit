from pathlib import Path

KIA_DIR_NAME = ".kia"


def get_kia_dir(cwd: str | Path | None = None) -> Path:
    """Return the .kia directory for the given working directory, creating it if needed."""
    base = Path(cwd) if cwd else Path.cwd()
    kia_dir = base / KIA_DIR_NAME
    kia_dir.mkdir(parents=True, exist_ok=True)
    return kia_dir
