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
    EXEC_READER_JOIN_TIMEOUT,
    MAX_EXEC_ARTIFACT_BYTES,
    MAX_EXEC_OUTPUT_CHARS,
    MAX_STREAMING_BUFFER_CHARS,
)
from .processes import _terminate_process


class CommandToolsMixin:
    def _exec_command(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        """Execute a shell command, streaming output in real-time.

        Uses subprocess.Popen with reader threads so output is displayed as it
        arrives. Rolling character buffers bound memory use for long-running
        processes; the returned result keeps trailing output from both streams,
        reserving up to half its character budget for stderr.
        """
        cwd = str(self._resolve_path(cwd or "."))
        self.console.tool(f"exec_command: {command} (cwd={cwd})")

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
                    shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=cwd or None,
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
                    shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=cwd or None, start_new_session=True,
                )
            except Exception:
                artifact_file.close()
                Path(artifact_path).unlink(missing_ok=True)
                raise

        stdout_lines: deque[str] = deque()
        stderr_lines: deque[str] = deque()
        stdout_size = [0]
        stderr_size = [0]
        artifact_lock = threading.Lock()
        artifact_size_bytes = [0]
        total_output_chars = [0]
        artifact_truncated = [False]
        artifact_write_error: list[str] = []
        capture_stopped = threading.Event()

        def _drain(stream, lines_buf, size_ref, prefix=""):
            decoder = codecs.getincrementaldecoder(locale.getpreferredencoding())(errors="replace")

            def consume(text: str) -> None:
                if not text or capture_stopped.is_set():
                    return
                lines_buf.append(text)
                size_ref[0] += len(text)
                while size_ref[0] > MAX_STREAMING_BUFFER_CHARS and len(lines_buf) > 1:
                    size_ref[0] -= len(lines_buf.popleft())
                captured = f"{prefix}{text}"
                encoded = captured.encode("utf-8")
                with artifact_lock:
                    if capture_stopped.is_set():
                        return
                    total_output_chars[0] += len(captured)
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
                        self.console.print(f"  {prefix}{display}", style="dim")

            pending = ""
            try:
                while raw := stream.read1(4096):
                    pending += decoder.decode(raw)
                    start = 0
                    for match in re.finditer(r"\r\n|\r|\n", pending):
                        consume(pending[start:match.end()])
                        start = match.end()
                    pending = pending[start:]
                pending += decoder.decode(b"", final=True)
                consume(pending)
            finally:
                stream.close()

        t_out = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_lines, stdout_size), daemon=True,
        )
        t_err = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_lines, stderr_size, "[stderr] "), daemon=True,
        )
        t_out.start()
        t_err.start()

        # Wait for the process, watching the keyboard so ESC / Ctrl+C aborts it.
        interrupted = False
        try:
            with CancelWatcher(self.cancellation) as watcher:
                while proc.poll() is None:
                    if watcher.is_cancelled:
                        interrupted = True
                        break
                    time.sleep(0.1)
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            self.console.warn("Interrupting command...")
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
        t_out.join(timeout=EXEC_READER_JOIN_TIMEOUT)
        t_err.join(timeout=EXEC_READER_JOIN_TIMEOUT)
        readers_incomplete = t_out.is_alive() or t_err.is_alive()
        if readers_incomplete:
            with artifact_lock:
                capture_stopped.set()
                artifact_truncated[0] = True
            self.console.warn("Output readers did not finish; terminating remaining process tree.")
            _terminate_process(proc)
            t_out.join(timeout=EXEC_READER_JOIN_TIMEOUT)
            t_err.join(timeout=EXEC_READER_JOIN_TIMEOUT)

        with artifact_lock:
            try:
                artifact_file.flush()
            except OSError as e:
                if not artifact_write_error:
                    artifact_write_error.append(str(e))
                artifact_truncated[0] = True
            finally:
                artifact_file.close()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        total_len = len(stdout) + len(stderr)
        truncated = total_len > MAX_EXEC_OUTPUT_CHARS
        truncation_notice = ""
        if truncated:
            guidance = "Search the saved output or rerun with quiet flags or a targeted filter."
            while True:
                output_budget = MAX_EXEC_OUTPUT_CHARS - len(truncation_notice) - 1
                stderr_budget = min(len(stderr), output_budget // 2)
                kept_stdout = stdout[-(output_budget - stderr_budget):]
                kept_stderr = stderr[-(output_budget - len(kept_stdout)):]
                updated = (
                    f"[output truncated: showing {len(kept_stdout) + len(kept_stderr):,} of "
                    f"{total_len:,} characters. {guidance}]"
                )
                if updated == truncation_notice:
                    stdout, stderr = kept_stdout, kept_stderr
                    break
                truncation_notice = updated

        res: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "success": not interrupted and proc.returncode == 0,
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

        return res
