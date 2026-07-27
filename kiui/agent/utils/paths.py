from pathlib import Path

KIA_DIR_NAME = ".kia"
KIA_GITIGNORE_NAME = ".gitignore"
KIA_GITIGNORE_CONTENT = "*\n"


def get_kia_dir(cwd: str | Path | None = None) -> Path:
    """Return the self-ignored .kia directory, creating it if needed."""
    base = Path(cwd) if cwd else Path.cwd()
    kia_dir = base / KIA_DIR_NAME
    kia_dir.mkdir(parents=True, exist_ok=True)
    gitignore = kia_dir / KIA_GITIGNORE_NAME
    if (
        not gitignore.exists()
        or gitignore.read_text(encoding="utf-8") != KIA_GITIGNORE_CONTENT
    ):
        gitignore.write_text(KIA_GITIGNORE_CONTENT, encoding="utf-8")
    return kia_dir
