from __future__ import annotations

import datetime
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Callable, Iterable

from filelock import FileLock
from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import is_searching
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.search import accept_search
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError
from pygments.lexers.markup import MarkdownLexer

from kiui.agent.utils.io import sanitize_unicode


class AtFileCompleter(Completer):
    """Auto-complete file paths when the user types ``@<partial_path>``.

    Only triggers when ``@`` appears at the start of a word (preceded by
    whitespace, beginning-of-line, etc.).  Directories are shown with a
    trailing ``/``.

    Matching strategy mirrors production coding agents (Claude Code /
    Codex CLI):

    * **Candidate index built once, cached in memory.** Every keystroke
      matches against the cached list instead of re-walking the disk, so
      latency is independent of repo size after the first build.  The
      candidate set is sourced from ``git`` (tracked + untracked, honoring
      ``.gitignore``) which is both fast and skips the bulky/ignored dirs
      that would otherwise dominate a raw filesystem walk.  Outside a git
      repo we fall back to a pruned filesystem walk.
    * **fzf-style fuzzy scorer.** A subsequence matcher rewards matches at
      path/word boundaries and consecutive runs, and matches the basename
      before the full path so ``@terminal`` surfaces
      ``kiui/agent/terminal.py`` at the top even in a deep tree.

    When a directory separator is present in *partial*, the completer uses
    exact-prefix listing for known directories and repo-wide fuzzy matching
    otherwise. This allows queries such as ``@agent/term`` to match
    ``kiui/agent/terminal.py``.
    """

    # Hard limit on completions shown
    MAX_COMPLETIONS = 30

    # How long a built candidate index stays valid before a refresh.
    # Short enough to pick up new files during a session, long enough to
    # avoid rebuilding on every burst of keystrokes.
    _INDEX_TTL = 5.0

    # Cap on candidates held in memory. Guards pathological monorepos;
    # git enumeration is cheap so this is rarely hit in practice.
    _MAX_CANDIDATES = 200000

    # Directories skipped during the filesystem-fallback walk (used only
    # when not inside a git repo).
    _SKIP_DIRS: frozenset[str] = frozenset({
        ".git", ".hg", ".svn",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
        "node_modules", ".venv", "venv", ".env", ".eggs",
        ".kia",  # our own cache dir
    })

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self._work_dir = Path(work_dir or os.getcwd()).resolve()
        # Cached candidate index: list of (rel_path, is_dir).
        self._index: list[tuple[str, bool]] = []
        self._index_built_at = 0.0

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

        # Use classic prefix listing when the directory portion is exact.
        # Otherwise keep fuzzy-matching the whole path, including separators.
        has_sep = "/" in partial or "\\" in partial
        if has_sep:
            base_dir, prefix = self._split(partial)
            search_dir = self._work_dir / base_dir if base_dir else self._work_dir
            try:
                search_dir = search_dir.resolve()
            except (OSError, RuntimeError):
                return
            if search_dir.is_dir():
                yield from self._direct_completions(search_dir, base_dir, prefix, partial)
                return

        # Fuzzy-match against the cached, repo-wide index.
        # An empty query lists the top-level entries from the same index
        # so ignored/hidden noise (.git, .venv, ...) stays out.
        if not partial:
            yield from self._toplevel_completions(partial)
            return

        yield from self._fuzzy_completions(partial)

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

    def _toplevel_completions(self, partial):
        """List top-level entries (dirs first, then files) from the cached
        index, so an empty ``@`` shows a clean listing without ignored or
        hidden noise."""
        seen: set[str] = set()
        entries: list[tuple[str, bool]] = []
        for rel, is_dir in self._get_index():
            top = rel.split("/", 1)[0]
            if top in seen:
                continue
            seen.add(top)
            entries.append((top, is_dir or "/" in rel))
        entries.sort(key=lambda e: (not e[1], e[0].lower()))
        for name, is_dir in entries[: self.MAX_COMPLETIONS]:
            yield self._make_completion(name, "", partial, is_dir)

    def _fuzzy_completions(self, partial):
        """Fuzzy-match *partial* against the cached repo-wide index.

        The index is built once and reused across keystrokes (see
        :meth:`_get_index`), so matching cost is independent of repo size.
        Candidates are ranked with an fzf-style scorer that matches the
        basename first (falling back to the full path), so a short query
        finds a deeply-nested file.
        """
        pl = partial.lower()
        scored: list[tuple[int, str, bool]] = []  # (score, rel_path, is_dir)

        for rel, is_dir in self._get_index():
            candidate = f"{rel}/" if is_dir else rel
            base = rel.rsplit("/", 1)[-1]
            score = self._fuzzy_score(base.lower(), pl)
            if score > 0:
                score += 20  # prefer basename hits
            else:
                score = self._fuzzy_score(candidate.lower(), pl)
            if score <= 0:
                continue
            # Shallower paths and shorter names rank higher on ties.
            score -= rel.count("/")
            scored.append((score, rel, is_dir))

        # Sort: higher score first, then dirs first, then shorter path, then alpha
        scored.sort(key=lambda x: (-x[0], not x[2], len(x[1]), x[1].lower()))

        for _score, rel, is_dir in scored[: self.MAX_COMPLETIONS]:
            yield self._make_completion(rel, "", partial, is_dir)

    # ------------------------------------------------------------------
    # Candidate index (built once, cached)
    # ------------------------------------------------------------------

    def _get_index(self) -> list[tuple[str, bool]]:
        """Return the cached (rel_path, is_dir) candidate list, rebuilding
        it if the TTL has expired."""
        now = time.monotonic()
        if self._index and now - self._index_built_at < self._INDEX_TTL:
            return self._index
        self._index = self._build_index()
        self._index_built_at = now
        return self._index

    def _build_index(self) -> list[tuple[str, bool]]:
        index = self._git_index()
        if index is None:
            index = self._walk_index()
        return index[: self._MAX_CANDIDATES]

    def _git_index(self) -> list[tuple[str, bool]] | None:
        """Enumerate candidates via git (tracked + untracked, honoring
        ``.gitignore``).  Returns None when not inside a git repo or git
        is unavailable.

        Git lists files only, so parent directories are synthesized to
        keep directory completions working.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                capture_output=True, text=True, cwd=str(self._work_dir), timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None

        files: list[str] = []
        dirs: set[str] = set()
        for line in result.stdout.splitlines():
            rel = line.strip()
            if not rel:
                continue
            rel = rel.replace("\\", "/")
            files.append(rel)
            # Synthesize ancestor directories.
            parts = rel.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))

        index: list[tuple[str, bool]] = [(d, True) for d in dirs]
        index.extend((f, False) for f in files)
        return index

    def _walk_index(self) -> list[tuple[str, bool]]:
        """Filesystem fallback used outside a git repo: DFS walk pruning
        hidden and well-known bulky directories."""
        index: list[tuple[str, bool]] = []
        work_dir = str(self._work_dir)
        stack: list[str] = [work_dir]
        while stack and len(index) < self._MAX_CANDIDATES:
            current = stack.pop()
            try:
                it = os.scandir(current)
            except (PermissionError, OSError):
                continue
            with it:
                for entry in it:
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
                    rel = os.path.relpath(entry.path, work_dir).replace(os.sep, "/")
                    index.append((rel, is_dir))
        return index


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
    def _fuzzy_score(text: str, pattern: str) -> int:
        """fzf-style subsequence score of *pattern* against *text* (both
        lowercase). Returns 0 when *pattern* is not a subsequence.

        Rewards: exact/prefix match, matches at word/path boundaries
        (after ``/ _ - .`` or a start), and consecutive runs. Penalizes
        gaps. Higher is better.
        """
        if not pattern:
            return 1
        if text == pattern:
            return 1000
        if text.startswith(pattern):
            return 500

        score = 0
        pi = 0
        prev_matched = False
        boundary_chars = "/_-. "
        for i, ch in enumerate(text):
            if pi < len(pattern) and ch == pattern[pi]:
                bonus = 1
                if i == 0 or text[i - 1] in boundary_chars:
                    bonus += 8  # boundary match (word/path segment start)
                if prev_matched:
                    bonus += 5  # consecutive run
                score += bonus
                pi += 1
                prev_matched = True
            else:
                prev_matched = False
        if pi != len(pattern):
            return 0  # not a subsequence
        return score

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


class MessageValidator(Validator):
    def __init__(self, has_pending: Callable[[], bool]):
        self._has_pending = has_pending

    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(message="Please enter a message.")
        if self._has_pending():
            raise ValidationError(message="Another message is already pending.")


class SharedFileHistory(FileHistory):
    """``FileHistory`` safe for multiple concurrent ``kia`` agents.
    """

    def __init__(self, filename):
        super().__init__(filename)
        self._lock = FileLock(f"{filename}.lock")

    def store_string(self, string: str) -> None:
        string = sanitize_unicode(string)
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
        self._busy = False
        self._status: list[tuple[str, str]] = []
        self._status_lock = threading.Lock()
        self._cancel: Callable[[], None] | None = None
        self._pending_text: Callable[[], str | None] | None = None
        self._edit_pending: Callable[[], str | None] | None = None
        self.style = Style.from_dict({
            "prompt": "bold ansiyellow",
            "prompt.busy": "bold ansicyan",
            "status.spinner": "bold ansigreen",
            "status.text": "ansiwhite",
            "status.detail": "ansibrightblack",
            "status.pending": "ansimagenta",
            "status.progress": "ansicyan",
            "status.context.low": "ansicyan",
            "status.context.medium": "ansiyellow",
            "status.context.high": "ansired",
            "status.track": "ansibrightblack",
            "separator": "ansibrightblack",
            "separator.busy": "ansicyan",
            "": "ansiyellow",
        })

        self.history = SharedFileHistory(str(history_path)) if history_path else None
        output = create_output()
        if hasattr(output, "enable_cpr"):
            output.enable_cpr = False
        self._session = PromptSession(
            multiline=True,
            style=self.style,
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._create_keybindings(),
            # Synchronize near the visible lines instead of reparsing the whole
            # multiline buffer on every keystroke.
            lexer=PygmentsLexer(MarkdownLexer, sync_from_start=False),
            validator=MessageValidator(self._has_pending),
            validate_while_typing=False,
            completer=AtFileCompleter(work_dir),
            erase_when_done=True,
            output=output,
        )

    def _create_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter", filter=~is_searching)
        def _(event):
            buf = event.current_buffer
            if buf.complete_state is not None:
                state = buf.complete_state
                completion = state.current_completion
                if completion is None and state.completions:
                    completion = state.completions[0]
                if completion is not None:
                    buf.apply_completion(completion)
                else:
                    buf.complete_state = None
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter", filter=~is_searching)
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-c", filter=~is_searching)
        def _(event):
            buf = event.current_buffer
            if self._busy:
                if self._cancel is not None:
                    self._cancel()
                return
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

        @kb.add("escape", filter=~is_searching)
        def _(event):
            if self._busy and self._cancel is not None:
                self._cancel()

        @kb.add("up", filter=~is_searching)
        def _(event):
            pending = (
                self._pending_text()
                if self._pending_text is not None
                else None
            )
            if pending is not None and not event.current_buffer.text:
                text = self._edit_pending() if self._edit_pending is not None else None
                if text is not None:
                    event.current_buffer.text = text
                    event.current_buffer.cursor_position = len(text)
                return
            event.current_buffer.auto_up(count=event.arg)

        @kb.add("tab", filter=is_searching)
        def _(event):
            accept_search.call(event)

        return kb

    def _has_pending(self) -> bool:
        return bool(
            self._pending_text is not None and self._pending_text() is not None
        )

    def _message(self):
        pending = self._pending_text() if self._pending_text is not None else None
        with self._status_lock:
            status = list(self._status)
        if self._busy and not status:
            status = [
                ("class:status.spinner", "⠋ "),
                ("class:status.text", "Working..."),
            ]
        if self._busy and pending is None:
            status.append(("class:status.detail", " · messages sent now will be queued"))
        if pending is not None:
            preview = pending.replace("\n", " ")
            if len(preview) > 20:
                preview = preview[:20] + "..."
            label = f"pending: {preview} · runs next"
            if status:
                status.append(("class:status.pending", f" · {label}"))
            else:
                status = [("class:status.pending", label)]
        width = max(1, self._session.app.output.get_size().columns - 1)
        prompt = [
            (
                "class:separator.busy" if self._busy else "class:separator",
                "─" * width,
            ),
            ("", "\n"),
            (
                "class:prompt.busy" if self._busy else "class:prompt",
                self._prompt_label,
            ),
        ]
        if status:
            return [*status, ("", "\n"), *prompt]
        return prompt

    def set_runtime_state(
        self,
        *,
        cancel: Callable[[], None],
        pending_text: Callable[[], str | None],
        edit_pending: Callable[[], str | None],
    ) -> None:
        self._cancel = cancel
        self._pending_text = pending_text
        self._edit_pending = edit_pending

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        app = self._session.app
        if app.is_running:
            app.invalidate()

    def set_status(self, status: list[tuple[str, str]] | None) -> None:
        with self._status_lock:
            self._status = status or []
        app = self._session.app
        if app.is_running:
            app.invalidate()

    def prompt(self) -> str:
        """Read a line of input from the terminal.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C.
            EOFError: When the user presses Ctrl+D (end of input).
        """
        return self._session.prompt(self._message)

    async def prompt_async(self, default: str = "") -> str:
        """Read input without moving prompt_toolkit off the main thread."""
        return await self._session.prompt_async(self._message, default=default)

    @property
    def app(self):
        return self._session.app

    @property
    def text(self) -> str:
        return self._session.default_buffer.text
