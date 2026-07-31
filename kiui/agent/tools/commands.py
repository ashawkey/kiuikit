"""Foreground shell command tool."""

import codecs
import locale
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from kiui.agent.utils.interrupt import CancelWatcher

from .constants import (
    EXEC_DISPLAY_FLUSH_LINES,
    EXEC_DISPLAY_FLUSH_SECONDS,
    EXEC_READER_JOIN_TIMEOUT,
    MAX_EXEC_ARTIFACT_BYTES,
    MAX_EXEC_OUTPUT_CHARS,
    MAX_STREAMING_BUFFER_CHARS,
)
from .process_util import _terminate_process


class CommandToolsMixin:
    def _exec_command(
        self, command: str, cwd: str | None = None, timeout: float | None = 300
    ) -> dict[str, Any]:
        """Execute a non-interactive shell command, streaming merged output."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or null")
        cwd = str(self._resolve_path(cwd or "."))

        artifact_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="kia-exec-", suffix=".txt", delete=False
        )
        artifact_path = artifact_file.name

        if sys.platform == "win32":
            # Use PowerShell (with user profile) as the modern default on Windows.
            # -NoLogo suppresses the copyright banner; profile is loaded by default.
            shell_cmd = ["powershell", "-NoLogo", "-Command", command]
            try:
                proc = subprocess.Popen(
                    shell_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, cwd=cwd or None,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            except Exception:
                artifact_file.close()
                Path(artifact_path).unlink(missing_ok=True)
                raise
        else:
            shell_cmd = ["/bin/bash", "-lc", command]
            try:
                proc = subprocess.Popen(
                    shell_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, cwd=cwd or None, start_new_session=True,
                )
            except Exception:
                artifact_file.close()
                Path(artifact_path).unlink(missing_ok=True)
                raise

        output_lines: deque[str] = deque()
        output_size = [0]
        artifact_lock = threading.Lock()
        artifact_size_bytes = [0]
        total_output_chars = [0]
        artifact_truncated = [False]
        artifact_write_error: list[str] = []
        capture_stopped = threading.Event()

        def _drain(stream, lines_buf, size_ref):
            decoder = codecs.getincrementaldecoder(locale.getpreferredencoding())(errors="replace")
            # Rendering one line at a time costs ~100us through rich, which
            # dominates the runtime of any command with lots of output (~20s for
            # 200k lines). Lines are accumulated and rendered in one call at most
            # every EXEC_DISPLAY_FLUSH_SECONDS, and a single flush echoes at most
            # EXEC_DISPLAY_FLUSH_LINES of them: past that the terminal is only
            # scrolling text nobody can read, while the complete output is
            # already being captured to the artifact the model is pointed at.
            display_buf: deque[str] = deque(maxlen=EXEC_DISPLAY_FLUSH_LINES)
            dropped = 0
            last_flush = time.monotonic()

            def flush_display(force: bool = False) -> None:
                nonlocal last_flush, dropped
                if not display_buf and not dropped:
                    return
                now = time.monotonic()
                if not force and now - last_flush < EXEC_DISPLAY_FLUSH_SECONDS:
                    return
                lines = list(display_buf)
                display_buf.clear()
                if dropped:
                    lines.insert(0, f"  [{dropped} line(s) not shown; full output captured]")
                    dropped = 0
                # markup=False: command output is data, so a stray "[/x]" must
                # not be parsed as rich markup (which drops it or raises).
                self.console.print("\n".join(lines), style="dim", markup=False)
                last_flush = now

            def consume(text: str) -> None:
                nonlocal dropped
                if not text or capture_stopped.is_set():
                    return
                lines_buf.append(text)
                size_ref[0] += len(text)
                while size_ref[0] > MAX_STREAMING_BUFFER_CHARS and len(lines_buf) > 1:
                    size_ref[0] -= len(lines_buf.popleft())
                encoded = text.encode("utf-8")
                with artifact_lock:
                    if capture_stopped.is_set():
                        return
                    total_output_chars[0] += len(text)
                    remaining = MAX_EXEC_ARTIFACT_BYTES - artifact_size_bytes[0]
                    if len(encoded) > remaining:
                        artifact_truncated[0] = True
                    if remaining > 0 and not artifact_write_error:
                        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
                        try:
                            artifact_file.write(chunk)
                        except OSError as e:
                            artifact_write_error.append(str(e))
                            artifact_truncated[0] = True
                        else:
                            artifact_size_bytes[0] += len(chunk.encode("utf-8"))
                for display in re.split(r"[\r\n]+", text):
                    if display:
                        if len(display_buf) == display_buf.maxlen:
                            dropped += 1
                        display_buf.append(f"  {display}")

            pending = ""
            try:
                while raw := stream.read1(4096):
                    pending += decoder.decode(raw)
                    start = 0
                    for match in re.finditer(r"\r\n|\r|\n", pending):
                        consume(pending[start:match.end()])
                        start = match.end()
                    pending = pending[start:]
                    flush_display()
                pending += decoder.decode(b"", final=True)
                consume(pending)
            finally:
                flush_display(force=True)
                stream.close()

        reader = threading.Thread(
            target=_drain, args=(proc.stdout, output_lines, output_size), daemon=True,
        )
        reader.start()

        # Wait for the process, watching the keyboard so ESC / Ctrl+C aborts it.
        interrupted = False
        timed_out = False
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            with CancelWatcher(self.cancellation) as watcher:
                while proc.poll() is None:
                    if watcher.is_cancelled:
                        interrupted = True
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = proc.poll() is None
                        if timed_out:
                            break
                    time.sleep(0.1)
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            self.console.warn("Interrupting command...")
            _terminate_process(proc)
        elif timed_out:
            self.console.warn(f"Command timed out after {timeout:g} seconds; terminating it.")
            _terminate_process(proc)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # A background descendant can retain the pipes after the shell exits.
        # Stop capture before closing the shared artifact, then let any lingering
        # daemon drainers discard bytes rather than touching closed state.
        reader.join(timeout=EXEC_READER_JOIN_TIMEOUT)
        readers_incomplete = reader.is_alive()
        if readers_incomplete:
            with artifact_lock:
                capture_stopped.set()
                artifact_truncated[0] = True
            self.console.warn("Output readers did not finish; terminating remaining process tree.")
            _terminate_process(proc)
            reader.join(timeout=EXEC_READER_JOIN_TIMEOUT)

        with artifact_lock:
            try:
                artifact_file.flush()
            except OSError as e:
                if not artifact_write_error:
                    artifact_write_error.append(str(e))
                artifact_truncated[0] = True
            finally:
                artifact_file.close()

        output = "".join(output_lines)

        total_len = len(output)
        truncated = total_len > MAX_EXEC_OUTPUT_CHARS
        truncation_notice = ""
        if truncated:
            guidance = "Search the saved output or rerun with quiet flags or a targeted filter."
            while True:
                output_budget = MAX_EXEC_OUTPUT_CHARS - len(truncation_notice) - 1
                kept_output = output[-output_budget:]
                updated = (
                    f"[output truncated: showing {len(kept_output):,} of "
                    f"{total_len:,} characters. {guidance}]"
                )
                if updated == truncation_notice:
                    output = kept_output
                    break
                truncation_notice = updated

        res: dict[str, Any] = {
            "stdout": output,
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "success": not interrupted and not timed_out and proc.returncode == 0,
            "streamed": True,
            "_artifact_path": artifact_path,
            "original_output_chars": total_output_chars[0],
            "artifact_size_bytes": artifact_size_bytes[0],
            "artifact_truncated": artifact_truncated[0],
        }
        if truncated:
            res["truncated"] = True
            res["truncation_reason"] = "character cap"
            res["guidance"] = "Search the saved output or rerun with quiet flags or a targeted filter."
            res["truncation_notice"] = truncation_notice
        if artifact_write_error:
            res["artifact_capture_error"] = artifact_write_error[0]
        if readers_incomplete:
            res["artifact_capture_incomplete"] = True
        if interrupted:
            res["interrupted"] = True
            res["error"] = "Command was interrupted by user."
        if timed_out:
            res["timed_out"] = True
            res["error"] = f"Command timed out after {timeout:g} seconds."

        return res
