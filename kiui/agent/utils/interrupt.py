"""Keyboard interruption for the kia agent.

Covers the "agent is busy" phase — while we wait for the LLM (or sleep
during retry backoff).  The user can press **ESC** or **Ctrl+C** to cancel
the in-flight request and return to the prompt.

How it works: the blocking call runs in a worker thread while the main thread
watches the keyboard. Cancellation runs an optional callback (for example,
closing an HTTP client), briefly waits for cleanup, then raises
``RequestInterrupted``.

The user-input phase (Ctrl+C clears prompt / double-tap quits) is handled
separately by prompt_toolkit key bindings in ``terminal.py``.
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Callable, TypeVar

from .io import CancellationToken

T = TypeVar("T")


class RequestInterrupted(Exception):
    """Raised when the user cancels an in-flight request (ESC / Ctrl+C)."""


# Cancel keys read from the raw console: ESC and Ctrl+C.
_CANCEL_KEYS = ("\x1b", "\x03")


def _can_watch() -> bool:
    """True if stdin is an interactive console we can poll for keys."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


if sys.platform == "win32":
    import msvcrt

    def _watch_for_cancel(stop: threading.Event) -> bool:
        """Poll the console; return True if a cancel key is pressed.

        Returns False if *stop* is set first.  Swallows other keys so they
        don't leak into the next prompt; consumes the 2-char sequences that
        arrow / function keys emit.
        """
        while not stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in _CANCEL_KEYS:
                    return True
                if ch in ("\x00", "\xe0"):  # special-key prefix
                    if msvcrt.kbhit():
                        msvcrt.getwch()  # discard the scan code
            else:
                time.sleep(0.03)
        return False

else:
    import select
    import termios
    import tty

    def _watch_for_cancel(stop: threading.Event) -> bool:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch in _CANCEL_KEYS:
                        return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return False


class CancelWatcher:
    """Background watcher that flips ``.cancelled`` on ESC / Ctrl+C.

    Wrap a polling loop you drive yourself (e.g. waiting on a subprocess)::

        with CancelWatcher() as w:
            while proc.poll() is None:
                if w.cancelled:
                    kill(proc); break
                time.sleep(0.1)

    Inactive (never cancels) when stdin isn't an interactive console.
    """

    def __init__(self, cancellation: CancellationToken | None = None) -> None:
        self.cancelled = False
        self._cancellation = cancellation
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = _can_watch()

    def __enter__(self) -> "CancelWatcher":
        if self._active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        if _watch_for_cancel(self._stop):
            self.cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled or bool(
            self._cancellation is not None and self._cancellation.cancelled
        )

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)


def run_interruptible(
    fn: Callable[[], T],
    cancellation: CancellationToken | None = None,
    on_cancel: Callable[[], None] | None = None,
) -> T:
    """Run blocking *fn* while watching for ESC / Ctrl+C.

    Returns ``fn()``'s result.  Re-raises any exception ``fn`` throws.
    Raises ``RequestInterrupted`` if the user cancels first.

    When stdin isn't an interactive console, *fn* simply runs inline.
    """
    if not _can_watch() and cancellation is None:
        return fn()

    result: dict[str, object] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            result["value"] = fn()
        except BaseException as e:  # re-raised on the calling thread
            result["error"] = e
        finally:
            done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    stop = threading.Event()
    cancelled = {"flag": False}

    def watcher() -> None:
        if _watch_for_cancel(stop):
            cancelled["flag"] = True
            done.set()  # wake the main thread

    w = None
    if _can_watch():
        w = threading.Thread(target=watcher, daemon=True)
        w.start()

    try:
        # done.wait() so a real SIGINT (Windows Ctrl+C) interrupts us too.
        while not done.wait(0.1):
            if cancellation is not None and cancellation.cancelled:
                cancelled["flag"] = True
                break
    except KeyboardInterrupt:
        cancelled["flag"] = True
    finally:
        stop.set()
        if w is not None:
            w.join(timeout=0.3)  # let posix restore terminal mode before we return

    if cancelled["flag"]:
        if on_cancel is not None:
            try:
                on_cancel()
            except Exception:
                pass
        worker_thread.join(timeout=0.5)
        raise RequestInterrupted()

    worker_thread.join()
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["value"]  # type: ignore[return-value]
