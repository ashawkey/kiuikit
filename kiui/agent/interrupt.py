"""Graceful Ctrl+C handling for the kiui agent.

- Single Ctrl+C during a task: sets _interrupted flag, aborts the active API request
- Double Ctrl+C (within 1.5s): force-exits via os._exit(1) with best-effort cleanup
- At the prompt (no task running): raises KeyboardInterrupt for prompt_toolkit
"""

import os
import signal
import sys
import time

class InterruptHandler:
    DOUBLE_PRESS_WINDOW = 1.5  # seconds

    def __init__(self):
        self._interrupted = False
        self._task_running = False
        self._last_sigint: float = 0.0
        self._original_handler = signal.getsignal(signal.SIGINT)
        self._agent = None
        self._installed = False

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def reset(self):
        """Clear the interrupted flag (call before each agent iteration)."""
        self._interrupted = False

    def install(self, agent):
        """Install the SIGINT handler. Idempotent."""
        self._agent = agent
        if not self._installed:
            self._original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
            self._installed = True

    def uninstall(self):
        """Restore the original SIGINT handler."""
        if self._installed:
            signal.signal(signal.SIGINT, self._original_handler)
            self._installed = False

    def set_task_running(self, running: bool):
        """Mark whether an agent task (API call / tool execution) is active."""
        self._task_running = running

    def _on_sigint(self, signum, frame):
        now = time.monotonic()

        if now - self._last_sigint < self.DOUBLE_PRESS_WINDOW:
            self._force_exit()
            return

        self._last_sigint = now

        if not self._task_running:
            raise KeyboardInterrupt

        self._interrupted = True
        sys.stderr.write("\nInterrupting... (press Ctrl+C again to force quit)\n")
        sys.stderr.flush()

        # Abort the in-flight HTTP request by closing the httpx client
        if self._agent:
            try:
                self._agent.client.close()
            except Exception:
                pass

    def _force_exit(self):
        """Immediate exit with best-effort cleanup."""
        sys.stderr.write("\nForce quitting...\n")
        sys.stderr.flush()
        if self._agent:
            try:
                self._agent._print_token_summary()
            except Exception:
                pass
        os._exit(1)
