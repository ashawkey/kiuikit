"""Standing-goal command and iteration state machine."""


class GoalMixin:
    def _cmd_goal(self, raw: str):
        """Handle /goal — set, show, or clear the standing goal.

        Usage:
          /goal <text>   set a new goal and start auto-iterating
          /goal          show current goal and status
          /goal clear    clear the goal and stop auto-iteration
        """
        parts = raw.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        low = arg.lower()

        if not arg:  # status
            if self.goal:
                self.console.print(
                    f"[bold blue]Goal[/bold blue] ([green]active[/green] if running, "
                    f"{self.goal_iterations} iteration(s)):\n  {self.goal}\n"
                    "  [dim]Ctrl+C during the loop to stop, or /goal clear[/dim]"
                )
            else:
                self.console.print("No goal set. Use [cyan]/goal <description>[/cyan] to set one.")
            return

        if low in ("clear", "off", "stop", "none"):
            self.goal = None
            self.goal_active = False
            self.goal_iterations = 0
            self._pending_auto = None
            self.console.system("Goal cleared.")
            return

        if self.persona.tools is not None and "report_goal" not in self.persona.tools:
            self.console.warn(
                f"Persona '{self.persona.name}' does not support /goal (report_goal is unavailable)."
            )
            return

        # set a brand-new goal
        self.goal = arg
        self.goal_active = True
        self.goal_iterations = 0
        self._pending_auto = self._build_goal_prompt()
        self.console.system(f"Goal set — agent will iterate until met (Ctrl+C to stop):\n  {arg}")

    def _build_goal_prompt(self) -> str:
        """The auto-injected prompt sent after each round while a goal is active."""
        return (
            f"[GOAL CHECK] Your standing goal is:\n{self.goal}\n\n"
            "Assess whether this goal is now fully met.\n"
            "- If it is fully met, call report_goal(met=true) with a brief reason.\n"
            "- If it is not met, keep working toward it (use tools as needed), then call "
            "report_goal(met=false, reason=...) describing what still remains.\n"
            "Always finish your turn by calling report_goal exactly once."
        )

    def _maybe_continue_goal(self):
        """After a round, decide whether to queue another goal-check iteration.

        Reads the report_goal() result stashed on the tool executor. Stops when
        the goal is reported met or the round was interrupted; otherwise arms the
        next auto-iteration.
        """
        if not (self.goal and self.goal_active):
            return

        if self._last_interrupted:
            # Cancelling a goal round clears the goal entirely.
            self.goal = None
            self.goal_active = False
            self.goal_iterations = 0
            self._pending_auto = None
            self.console.system("[goal] cleared (interrupted).")
            return

        report = self.tool_executor.goal_report
        if report and report.get("met"):
            reason = report.get("reason", "")
            self.console.system(f"[goal] ✓ met after {self.goal_iterations} iteration(s)." + (f" {reason}" if reason else ""))
            self.goal_active = False
            self._pending_auto = None
            return

        # not met (or the model failed to report) → iterate again
        self.goal_iterations += 1
        self._pending_auto = self._build_goal_prompt()
        reason = report.get("reason", "") if report else ""
        self.console.system(
            f"[goal] not met (iteration {self.goal_iterations}) — continuing"
            + (f": {reason}" if reason else "")
        )

