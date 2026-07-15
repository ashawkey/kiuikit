from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Iterable

from filelock import FileLock
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

    # Hard limit on completions shown
    MAX_COMPLETIONS = 30

    # Upper bound on filesystem entries visited during a recursive scan.
    # Bounds worst-case latency on huge repos so the UI never blocks for
    # long on a single keystroke.
    _MAX_SCAN_ENTRIES = 20000

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

        # Bail out if there is any whitespace in *partial* — means the
        # cursor has moved past the @ token and the user is typing
        # additional text.  This avoids expensive re-scans once the file
        # path is confirmed.  (The backend's _AT_PATH_RE also rejects
        # whitespace inside @ references, so this is consistent.)
        if " " in partial or "\t" in partial:
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
        """Recursive scan with fuzzy filename matching.

        Walks the tree manually with :func:`os.scandir`, pruning hidden
        and well-known bulky directories *during* traversal so we never
        descend into ``.git`` / ``node_modules`` / etc.  A visited-entry
        budget (:data:`_MAX_SCAN_ENTRIES`) caps worst-case latency on very
        large repositories, keeping the completer responsive on every
        keystroke.

        ``pathlib``'s ``**`` glob is deliberately avoided here: it eagerly
        walks the entire subtree (including ignored dirs) before any
        filtering, which stalls the UI on large repos.
        """
        pl = prefix.lower()
        scored: list[tuple[int, str, bool]] = []  # (score, rel_path, is_dir)
        visited = 0
        work_dir = str(self._work_dir)

        # DFS stack of directories to scan (start at the search root).
        stack: list[str] = [str(search_dir)]
        while stack and visited < self._MAX_SCAN_ENTRIES:
            current = stack.pop()
            try:
                it = os.scandir(current)
            except (PermissionError, OSError):
                continue
            with it:
                for entry in it:
                    visited += 1
                    if visited >= self._MAX_SCAN_ENTRIES:
                        break

                    name = entry.name
                    if name.startswith("."):
                        continue
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        continue

                    if is_dir:
                        if name in self._SKIP_DIRS:
                            continue
                        stack.append(entry.path)

                    score = self._score(name, pl)
                    if score <= 0:
                        continue
                    rel = os.path.relpath(entry.path, work_dir).replace(os.sep, "/")
                    scored.append((score, rel, is_dir))

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


class SharedFileHistory(FileHistory):
    """``FileHistory`` safe for multiple concurrent ``kia`` agents.
    """

    def __init__(self, filename):
        super().__init__(filename)
        self._lock = FileLock(f"{filename}.lock")

    def store_string(self, string: str) -> None:
        payload = [f"\n# {datetime.datetime.now()}\n"]
        payload.extend(f"+{line}\n" for line in string.split("\n"))
        data = "".join(payload).encode("utf-8")
        with self._lock:
            with open(self.filename, "ab") as f:
                f.write(data)

    def load_history_strings(self) -> Iterable[str]:
        with self._lock:
            return super().load_history_strings()

    async def load(self) -> AsyncGenerator[str, None]:
        # prompt_toolkit calls load() before every prompt. Drop the cache so we
        # re-read entries appended by other agents since the last prompt.
        self._loaded = False
        self._loaded_strings = []
        async for item in super().load():
            yield item


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

        self.history = SharedFileHistory(str(history_path)) if history_path else None
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
            erase_when_done=True,
        )

    def _create_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            buf = event.current_buffer
            # If the completion menu is open, dismiss it without
            # submitting.  Tab already applies the highlighted
            # completion to the buffer as the user cycles, so by the
            # time Enter is pressed the correct text is already there.
            # This lets the user press Enter to confirm a file path
            # and then press Enter again to send the message.
            if buf.complete_state is not None:
                buf.complete_state = None  # dismiss the menu
            else:
                buf.validate_and_handle()

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

    async def prompt_async(self) -> str:
        """Read input without moving prompt_toolkit off the main thread."""
        return await self._session.prompt_async(
            [("class:prompt", self._prompt_label)]
        )
