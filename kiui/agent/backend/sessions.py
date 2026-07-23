"""Session selection, persistence, loading, and replay."""

import json
import os
from pathlib import Path

from kiui.agent.context import get_role, get_text, get_tool_calls
from kiui.agent.personas import get_persona
from kiui.agent.tools import format_tool_summary


class SessionMixin:
    SESSIONS_DIR_NAME = "sessions"

    @staticmethod
    def _session_preview(messages: list) -> str:
        """Short preview from the last user message of a saved session."""
        for m in reversed(messages):
            if get_role(m) != "user":
                continue
            text = get_text(m).replace("\n", " ").strip()
            return text[:60] + ("..." if len(text) > 60 else "")
        return ""

    def _pick_session(self) -> str | None:
        """List saved sessions and let the user pick one interactively."""
        sessions_dir = self._sessions_dir()
        files = sorted(sessions_dir.glob("*.json"), reverse=True)
        # Don't offer the current (unsaved-in-progress) session as a target.
        files = [f for f in files if f.stem != self._session_id]
        if not files:
            self.console.system(f"No other saved sessions in {sessions_dir}")
            return None

        stems: list[str] = []
        labels: list[str] = []
        for f in files:
            stem = f.stem
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                messages = meta.get("messages", [])
                n_msgs = len(messages)
                rnd = meta.get("round_id", "?")
                model = meta.get("model", "?")
                preview = self._session_preview(messages)
            except Exception:
                n_msgs, rnd, model, preview = "?", "?", "?", ""
            label = f"{stem}  │  msgs:{n_msgs}  rounds:{rnd}  model:{model}"
            if preview:
                label += f"  │  {preview}"
            stems.append(stem)
            labels.append(label)

        picked = self.console.select(message="Pick a session to resume", choices=labels)
        if picked is None:
            return None
        for stem, label in zip(stems, labels):
            if label == picked:
                return stem
        return None

    def _cmd_resume(self, raw: str):
        """Handle /resume — save the current session, then load a previous one."""
        parts = raw.split(maxsplit=1)
        target: str | None = parts[1].strip() if len(parts) > 1 else None

        if target:
            if not (self._sessions_dir() / f"{target}.json").exists():
                self.console.error(f"Session not found: {target}")
                return
        else:
            target = self._pick_session()
            if target is None:
                self.console.system("Resume cancelled.")
                return

        if target == self._session_id:
            self.console.system(f"Already in session '{target}'.")
            return

        # Save the current session before switching away.
        if self._session_id and self.context.messages:
            try:
                self.save_session(self._session_id)
                self.console.system(f"Session '{self._session_id}' saved.")
            except Exception as e:
                self.console.warn(f"Could not save current session: {e}")

        # Switch to the target session and reset per-session state.
        old_id = self._session_id
        old_save_time = self._last_save_time
        self._session_id = target
        self._last_save_time = 0.0
        if not self.load_session(target):
            self._session_id = old_id
            self._last_save_time = old_save_time
            return
        self.tool_executor.shutdown_processes(clear=True)
        _wd = self.tool_executor._work_dir or os.getcwd()
        if self.changes is not None:
            self.changes.close()
        self.changes = self._create_change_tracker(self._session_id, _wd)
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id

    def _cmd_rewind(self, raw: str):
        """Handle /rewind — roll back to a previous round."""
        if not self.changes:
            self.console.warn("Rewind is only available in interactive chat mode with a session.")
            return

        rounds = self.changes.build_round_list(self.context.messages)
        if not rounds:
            self.console.system("No rounds to rewind.")
            return

        parts = raw.split()
        target_round: int | None = None

        if len(parts) >= 2:
            try:
                target_round = int(parts[1])
                if target_round < 1 or target_round > len(rounds):
                    self.console.error(f"Round must be between 1 and {len(rounds)}.")
                    return
            except ValueError:
                self.console.error(f"Invalid round number: {parts[1]}")
                return
        else:
            # Build round choices for questionary picker
            round_choices: list[str] = []
            round_map: dict[str, int] = {}  # choice label → round number
            for r in reversed(rounds):
                n = r["round"]
                preview = r["preview"]
                files = r.get("files", 0)
                added = r.get("added", 0)
                removed = r.get("removed", 0)
                # Build change summary
                parts_line = []
                if files > 0:
                    parts_line.append(f"{files} file{'s' if files > 1 else ''}")
                if added > 0:
                    parts_line.append(f"+{added} line{'s' if added > 1 else ''}")
                if removed > 0:
                    parts_line.append(f"-{removed} line{'s' if removed > 1 else ''}")
                change_str = ""
                if parts_line:
                    change_str = f"  ({', '.join(parts_line)})"
                elif files == 0 and added == 0 and removed == 0:
                    change_str = "  (no file changes)"
                label = f"Round {n:>3} — {preview}{change_str}"
                round_choices.append(label)
                round_map[label] = n

            picked = self.console.select(
                message="Pick a round to revert",
                choices=round_choices,
            )
            if picked is None:
                return
            target_round = round_map.get(picked)
            if target_round is None or target_round < 1 or target_round > len(rounds):
                self.console.error(f"Invalid choice: {picked}")
                return

        # Revert to the state BEFORE target_round
        actual_target = target_round - 1

        # Ask revert mode
        self.console.print(f"\n[bold blue]? Revert round {target_round}?[/bold blue]")
        if actual_target > 0:
            preview = rounds[actual_target - 1]["preview"]
            self.console.print(f"  Will roll back to after round [cyan]{actual_target}[/cyan] — {preview}")
        else:
            self.console.print("  Will roll back to [cyan]initial state[/cyan] (before any messages)")

        # Show what will be reverted
        reverted_rounds = list(range(target_round, len(rounds) + 1))
        total_files = 0
        total_added = 0
        total_removed = 0
        for r in rounds:
            if r["round"] >= target_round:
                total_files += r.get("files", 0)
                total_added += r.get("added", 0)
                total_removed += r.get("removed", 0)
        if total_files > 0 or total_added > 0 or total_removed > 0:
            impact = []
            if total_files > 0:
                impact.append(f"{total_files} file{'s' if total_files > 1 else ''}")
            if total_added > 0:
                impact.append(f"+{total_added} lines")
            if total_removed > 0:
                impact.append(f"-{total_removed} lines")
            self.console.print(f"  Code impact: [yellow]{', '.join(impact)}[/yellow] across {len(reverted_rounds)} round{'s' if len(reverted_rounds) > 1 else ''}")

        mode_choice = self.console.select(
            message="Choose revert mode",
            choices=[
                "1. Conversation only (keep code changes)",
                "2. Code + conversation (restore both)",
                "3. Code only (keep conversation)",
            ],
        )
        if mode_choice is None:
            return
        mode = mode_choice[0]  # "1", "2", or "3"

        revert_code = mode in ("2", "3")
        revert_conv = mode in ("1", "2")

        # Execute rollback
        self.console.system(f"Reverting round {target_round} and later...")

        if revert_code:
            restored = self.changes.rollback_code(actual_target)
            self.console.system(f"Code reverted: {restored} files restored.")
        else:
            self.console.system("Code changes preserved.")

        if revert_conv:
            reverted, self.round_id = self.changes.rollback_conversation(
                self.context.messages, actual_target
            )
            self.context.replace_messages(reverted)
            replay_messages = len(self.context.messages)
            self.console.reset_timeline()
            self._replay_context()
            self.console.system(
                f"Conversation rolled back to round {self.round_id} ({replay_messages} messages)."
            )
        else:
            self.console.system("Conversation preserved.")

        # Persist the rewinded state to disk
        try:
            self.save_session(self.changes.session_id)
        except Exception:
            pass


    def _sessions_dir(self) -> Path:
        d = self._kia_dir() / self.SESSIONS_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_session(self, name: str | None = None) -> Path:
        """Save the current session state to a JSON file. Returns the file path."""
        if not name:
            name = self._session_timestamp()

        data = {
            "model": self.model,
            "round_id": self.round_id,
            "token_totals": self.token_totals,
            "tool_compaction_totals": self.tool_compaction_totals,
            "system_prompt": self.context.system_prompt,
            "persona": self.persona.name,
            "goal": self.goal,
            "goal_active": self.goal_active,
            "goal_iterations": self.goal_iterations,
            "loaded_skills": sorted(self.tool_executor._loaded_skills),
            "skill_loads": self.tool_executor._skill_loads,
            "messages": self.context.messages,
        }

        path = self._sessions_dir() / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.changes is not None and self.changes.session_id == name:
            self.changes.flush()
        return path

    def load_session(self, name: str) -> bool:
        """Load a previously saved session by name. Returns True on success."""
        path = self._sessions_dir() / f"{name}.json"
        if not path.exists():
            self.console.error(f"Session file not found: {path}")
            return False
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self.console.error(f"Failed to read session file: {e}")
            return False

        if not isinstance(data, dict) or "messages" not in data:
            self.console.error(f"Corrupted session file (missing 'messages' key): {path}")
            return False

        saved_model = data.get("model", "")
        if saved_model != self.model:
            self.console.system(f"Note: session was saved with model '{saved_model}', current model is '{self.model}'")

        # Re-apply the persona the session was saved under, so the prompt and
        # tool surface match the replayed conversation.
        saved_persona = data.get("persona")
        if saved_persona and saved_persona != self.persona.name:
            try:
                self.persona = get_persona(saved_persona)
            except ValueError:
                self.console.warn(f"Session persona '{saved_persona}' not found; keeping '{self.persona.name}'.")
            else:
                # self.tools follows the restored persona via the live property.
                self.system_prompt = self._build_system_prompt()
                self.context.system_prompt["content"] = self.system_prompt

        self.context.replace_messages(data["messages"])
        self.round_id = data.get("round_id", 0)
        saved_totals = data.get("token_totals")
        if saved_totals and isinstance(saved_totals, dict):
            self.token_totals.update(saved_totals)
        saved_compaction = data.get("tool_compaction_totals")
        if saved_compaction and isinstance(saved_compaction, dict):
            self.tool_compaction_totals.update(saved_compaction)

        # Restore skill usage so the model does not redundantly re-load skills
        # whose bodies are already in the replayed conversation, and so telemetry
        # (load counts) carries across --resume. Intersect with currently
        # discovered skills in case a skill was removed since the save.
        available = set(self.skills)
        self.tool_executor._loaded_skills = {
            n for n in data.get("loaded_skills", []) if n in available
        }
        # Re-register tools contributed by restored skills so a resumed session
        # advertises the same tool surface (self.tools reads the registry live).
        # If a skill's tools.py broke since the session was saved, drop it from
        # the loaded set instead of leaving it marked-loaded with missing tools.
        self.tool_executor.reset_skill_tools()
        for name in list(self.tool_executor._loaded_skills):
            skill_dir = self.skills[name].get("dir")
            error = self.tool_executor._register_skill_tools(name, skill_dir)
            if error is not None:
                self.console.warn(error)
                self.tool_executor._loaded_skills.discard(name)
        saved_loads = data.get("skill_loads")
        if isinstance(saved_loads, dict):
            self.tool_executor._skill_loads = {
                n: int(c) for n, c in saved_loads.items() if isinstance(c, int)
            }

        # Restore the goal and auto-resume iteration; Ctrl+C stops it.
        self.goal = data.get("goal")
        self.goal_iterations = data.get("goal_iterations", 0)
        goal_supported = self.persona.tools is None or "report_goal" in self.persona.tools
        if self.goal and goal_supported:
            self.goal_active = True
            self._pending_auto = self._build_goal_prompt()
            self.console.system(f"Goal resumed — iterating until met (Ctrl+C to stop):\n  {self.goal}")
        else:
            if self.goal:
                self.console.warn(
                    f"Saved goal disabled because persona '{self.persona.name}' cannot use report_goal."
                )
            self.goal = None
            self.goal_active = False
            self._pending_auto = None

        self.console.system(
            f"Loaded session '{name}' ({len(self.context.messages)} messages, round {self.round_id})"
        )

        # Replay the conversation so user can recall the context
        self._replay_context()
        return True

    def _replay_context(self):
        """Replay all messages using the same display methods as live chat."""
        msgs = self.context.messages
        if not msgs:
            return

        self.console.system("── Session context (replay) ──")

        for msg in msgs:
            role = get_role(msg)
            if role == "user":
                text = get_text(msg)
                self.console.user_input(text)

            elif role == "assistant":
                text = get_text(msg)
                if text:
                    self.console.response(text)
                for tc in get_tool_calls(msg):
                    fn = tc.get("function", {})
                    fname = fn.get("name", "?")
                    try:
                        fargs = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        fargs = {}
                    self.console.tool(f"{fname}({json.dumps(fargs, ensure_ascii=False)})")

            elif role == "tool":
                result_text = get_text(msg)
                success = "error" not in result_text.lower()
                summary = format_tool_summary(result_text)
                self.console.tool_result(summary, success=success)

        self.console.system("── End of replay ──")

