"""Agent-loop control tools."""

import math
import time
from typing import Any

from kiui.agent.utils.interrupt import CancelWatcher


class ControlToolsMixin:
    def _wait(self, seconds: float) -> dict[str, Any]:
        """Wait for a bounded duration while remaining user-interruptible."""
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise ValueError("seconds must be a number")
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        deadline = time.monotonic() + seconds
        interrupted = False
        try:
            with self.console.thinking(label="Waiting", countdown=seconds):
                with CancelWatcher(self.cancellation) as watcher:
                    while (remaining := deadline - time.monotonic()) > 0:
                        if watcher.is_cancelled:
                            interrupted = True
                            break
                        time.sleep(min(0.1, remaining))
        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            return {
                "error": "Wait was interrupted by the user.",
                "success": False,
                "interrupted": True,
            }
        return {"waited_seconds": seconds, "success": True}
