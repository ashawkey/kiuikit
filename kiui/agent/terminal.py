from __future__ import annotations

import os
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from pygments.lexers.markup import MarkdownLexer


class AtFileCompleter(Completer):
    """Auto-complete file paths when the user types ``@<partial_path>``.

    Only triggers when ``@`` appears at the start of a word (preceded by
    whitespace, beginning-of-line, etc.).  Directories are shown with a
    trailing ``/``.

    When no directory separator is present in *partial* the search is
    **recursive** and matches filenames fuzzily (substring →
    character-sequence), so ``@d.txt`` can expand to ``@a/b/c/d.txt``.
    When a directory separator *is* present the completer falls back to
    exact-prefix matching inside the specified directory (classic Tab
    completion).
    """

    # Hard limit on completions shown & entries scanned
    MAX_COMPLETIONS = 30
    MAX_SCAN = 2000

    # Directories skipped during recursive scans
    _SKIP_DIRS: frozenset[str] = frozenset({
        ".git", ".hg", ".svn",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
        "node_modules", ".venv", "venv", ".env", ".eggs",
        ".kia",  # our own cache dir
    })

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self._work_dir = Path(work_dir or os.getcwd()).resolve()

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        at_pos = text.rfind("@")
        if at_pos == -1:
            return

        # @ must start a word (preceded by whitespace / BOL / newline)
        if at_pos > 0 and text[at_pos - 1] not in (" ", "\n", "\t", "\r"):
            return

        partial = text[at_pos + 1:]

        # Bail out if the very next character is whitespace — the user
        # hasn't started typing a path yet (e.g. "@ ").  A bare "@"
        # (empty *partial*) is allowed and shows the top-level listing.
        if partial and partial[0] in (" ", "\t"):
            return

        base_dir, prefix = self._split(partial)
        search_dir = self._work_dir / base_dir if base_dir else self._work_dir

        try:
            search_dir = search_dir.resolve()
        except (OSError, RuntimeError):
            return

        if not search_dir.is_dir():
            return

        # Decide strategy: recursive-fuzzy vs direct-listing
        has_sep = "/" in partial or "\\" in partial

        if has_sep:
            yield from self._direct_completions(search_dir, base_dir, prefix, partial)
        elif prefix:
            yield from self._fuzzy_completions(search_dir, base_dir, prefix, partial)
        else:
            yield from self._direct_completions(search_dir, base_dir, "", partial)

    # ------------------------------------------------------------------
    # Completion strategies
    # ------------------------------------------------------------------

    def _direct_completions(self, search_dir, base_dir, prefix, partial):
        """Classic prefix match inside a single directory."""
        try:
            entries = sorted(
                search_dir.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            return

        for entry in entries:
            name = entry.name
            if not name.lower().startswith(prefix.lower()):
                continue
            yield self._make_completion(name, base_dir, partial, entry.is_dir())

    def _fuzzy_completions(self, search_dir, base_dir, prefix, partial):
        """Recursive scan with fuzzy filename matching."""
        import glob as glob_module

        scored: list[tuple[int, str, bool]] = []  # (score, rel_path, is_dir)
        seen: set[str] = set()
        scanned = 0

        # ---- pass 1: fast substring glob ---------------------------------
        escaped = glob_module.escape(prefix)
        pattern = f"**/*{escaped}*"
        try:
            for entry in search_dir.glob(pattern):
                if self._skip_entry(entry):
                    continue
                scanned += 1
                rel = self._relative(entry)
                if rel in seen:
                    continue
                seen.add(rel)
                score = self._score(entry.name, prefix)
                scored.append((score, rel, entry.is_dir()))
        except PermissionError:
            pass

        # ---- pass 2: character-sequence fallback (only when needed) ------
        if len(scored) < 5 and scanned < self.MAX_SCAN:
            try:
                for entry in search_dir.rglob("*"):
                    if self._skip_entry(entry):
                        continue
                    scanned += 1
                    if scanned > self.MAX_SCAN:
                        break
                    rel = self._relative(entry)
                    if rel in seen:
                        continue
                    seen.add(rel)
                    score = self._score(entry.name, prefix)
                    if score > 0:
                        scored.append((score, rel, entry.is_dir()))
            except PermissionError:
                pass

        # ---- sort & yield -------------------------------------------------
        # Sort: higher score first, then dirs first, then shorter path, then alpha
        scored.sort(key=lambda x: (-x[0], not x[2], len(x[1]), x[1].lower()))

        for _score, rel, is_dir in scored[: self.MAX_COMPLETIONS]:
            yield self._make_completion(rel, "", partial, is_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_completion(
        self, name: str, base_dir: str, partial: str, is_dir: bool
    ) -> Completion:
        """Build a ``Completion`` object."""
        completed = f"{base_dir}/{name}" if base_dir else name
        completed = completed.replace("\\", "/")
        if is_dir:
            completed += "/"
        display = f"{name}/" if is_dir else name
        return Completion(
            text=f"@{completed}",
            start_position=-len(partial) - 1,  # back to the @
            display=display,
        )

    def _relative(self, entry: Path) -> str:
        """Return *entry* as a forward-slash path relative to work_dir."""
        try:
            return entry.relative_to(self._work_dir).as_posix()
        except ValueError:
            return entry.as_posix()

    def _skip_entry(self, entry: Path) -> bool:
        """True if *entry* should be excluded from completions."""
        # Skip hidden files/dirs and well-known bulky directories
        if entry.name.startswith("."):
            return True
        if entry.is_dir() and entry.name in self._SKIP_DIRS:
            return True
        return False

    @staticmethod
    def _score(name: str, pattern: str) -> int:
        """Score a filename against the user's pattern (higher = better)."""
        nl = name.lower()
        pl = pattern.lower()

        if nl == pl:
            return 100  # exact match
        if nl.startswith(pl):
            return 80  # prefix match
        if pl in nl:
            return 60  # substring match
        if AtFileCompleter._char_seq_match(nl, pl):
            return 40  # character-sequence match
        return 0  # no match

    @staticmethod
    def _char_seq_match(name: str, pattern: str) -> bool:
        """True when every char of *pattern* appears in *name* in order."""
        idx = 0
        for ch in name:
            if idx < len(pattern) and ch == pattern[idx]:
                idx += 1
        return idx == len(pattern)

    @staticmethod
    def _split(path_str: str) -> tuple[str, str]:
        """Split *path_str* into ``(directory, prefix)``."""
        if not path_str:
            return "", ""
        path_str = path_str.replace("\\", "/")
        if path_str.endswith("/"):
            return path_str.rstrip("/"), ""
        if "/" not in path_str:
            return "", path_str
        *dirs, prefix = path_str.rsplit("/", 1)
        return "/".join(dirs), prefix


class NonEmptyInputValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(message="Please enter a message.")


class TerminalInput:
    def __init__(
        self,
        history_path: str | Path | None = None,
        prompt_label: str = "> ",
        work_dir: str | Path | None = None,
    ):
        self._prompt_label = prompt_label
        self._last_ctrl_c = 0.0  # timestamp of last Ctrl+C on an empty buffer
        self.style = Style.from_dict({
            "prompt": "bold ansiyellow",
            "": "ansiyellow",
        })

        self.history = FileHistory(str(history_path)) if history_path else None
        self._session = PromptSession(
            multiline=True,
            style=self.style,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._create_keybindings(),
            lexer=PygmentsLexer(MarkdownLexer),
            validator=NonEmptyInputValidator(),
            validate_while_typing=False,
            completer=AtFileCompleter(work_dir),
        )

    def _create_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            buf = event.current_buffer
            if buf.text:
                # Non-empty prompt: Ctrl+C just clears what's typed.
                buf.reset()
                self._last_ctrl_c = 0.0
                return
            # Empty prompt: arm on first press, quit on a second within 1s.
            now = time.monotonic()
            if self._last_ctrl_c and now - self._last_ctrl_c < 1.0:
                event.app.exit(exception=EOFError)  # quit the CLI
            else:
                self._last_ctrl_c = now
                run_in_terminal(lambda: print("(press Ctrl+C again to exit)"))

        return kb

    def prompt(self) -> str:
        """Read a line of input from the terminal.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
            EOFError: When the user presses Ctrl+D (end of input).
        """
        return self._session.prompt([("class:prompt", self._prompt_label)])