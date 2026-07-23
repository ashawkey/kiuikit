"""Session selection, branchable persistence, loading, and replay."""

import json
import os
import time
from pathlib import Path

from kiui.agent.context import get_role, get_text, get_tool_calls
from kiui.agent.personas import get_persona
from kiui.agent.session_store import SessionStore
from kiui.agent.tools import format_tool_summary


class SessionMixin:
    SESSIONS_DIR_NAME = "sessions"

    @staticmethod
    def _session_preview(messages: list) -> str:
        for message in reversed(messages):
            if get_role(message) != "user":
                continue
            text = get_text(message).replace("\n", " ").strip()
            return text[:60] + ("..." if len(text) > 60 else "")
        return ""

    def _session_store_for(self, name: str) -> SessionStore:
        return SessionStore(self._sessions_dir(), name)

    def _pick_session(self) -> str | None:
        sessions_dir = self._sessions_dir()
        paths = sorted(
            (path for path in sessions_dir.iterdir() if path.is_dir() and (path / "history.jsonl").exists()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        paths = [path for path in paths if path.name != self._session_id]
        if not paths:
            self.console.system(f"No other saved sessions in {sessions_dir}")
            return None

        names: list[str] = []
        labels: list[str] = []
        for path in paths:
            name = path.name
            try:
                meta = self._session_store_for(name).summary()
                messages = meta["messages"]
                label = (
                    f"{name}  │  msgs:{meta['message_count']}  rounds:{meta['round_id']} "
                    f" model:{meta['model']}"
                )
                preview = self._session_preview(messages)
                if preview:
                    label += f"  │  {preview}"
            except Exception:
                label = f"{name}  │  unreadable"
            names.append(name)
            labels.append(label)

        picked = self.console.select(message="Pick a session to resume", choices=labels)
        if picked is None:
            return None
        return names[labels.index(picked)]

    def _cmd_resume(self, raw: str):
        parts = raw.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else self._pick_session()
        if target is None:
            self.console.system("Resume cancelled.")
            return
        try:
            target_store = self._session_store_for(target)
        except ValueError as e:
            self.console.error(str(e))
            return
        if not target_store.exists:
            self.console.error(f"Session not found: {target}")
            return
        if target == self._session_id:
            self.console.system(f"Already in session '{target}'.")
            return

        if self._session_id and self.context.messages:
            try:
                self.save_session(self._session_id, reason="resume")
            except Exception as e:
                self.console.warn(f"Could not save current session: {e}")

        old_id = self._session_id
        old_save_time = self._last_save_time
        if not self.load_session(target):
            self._session_id = old_id
            self._last_save_time = old_save_time
            return
        self._session_id = target
        self._last_save_time = 0.0
        self.tool_executor.shutdown_processes(clear=True)
        self._install_change_tracker()

    def _cmd_rewind(self, raw: str):
        """Move to any saved revision; subsequent work creates a branch."""
        if not self._session_id or not self.changes:
            self.console.warn("Rewind is only available in interactive chat mode with a session.")
            return

        try:
            self.save_session(self._session_id, reason="pre-rewind")
        except Exception as e:
            self.console.error(f"Could not checkpoint current state: {e}")
            return

        store = self._session_store
        candidates = store.candidates()
        if len(candidates) <= 1:
            self.console.system("No earlier revisions to rewind to.")
            return

        parts = raw.split(maxsplit=1)
        if len(parts) > 1:
            try:
                target_id = store.resolve_revision(parts[1].strip())
            except ValueError as e:
                self.console.error(str(e))
                return
        else:
            labels: list[str] = []
            ids: list[str] = []
            for revision in candidates:
                marker = " [current]" if revision["current"] else ""
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(revision["created_at"]))
                label = (
                    f"{revision['id'][:10]}  │  round {revision['round_id']}  │  "
                    f"{revision['reason']}  │  {timestamp}{marker}"
                )
                labels.append(label)
                ids.append(revision["id"])
            picked = self.console.select(message="Pick a session revision", choices=labels)
            if picked is None:
                return
            target_id = ids[labels.index(picked)]

        if target_id == self._session_revision_id:
            self.console.system("That revision is already checked out.")
            return

        target = store.materialize(target_id)
        mode_choice = self.console.select(
            message="Choose rewind mode",
            choices=[
                "1. Conversation + code",
                "2. Conversation only (keep current code)",
                "3. Code only (keep current conversation)",
            ],
        )
        if mode_choice is None:
            return
        mode = mode_choice[0]

        current_code = self.changes.code_revision_id
        target_code = target.get("code_revision_id")
        if mode in ("1", "3"):
            changed = self.changes.checkout_code(target_code)
            self.console.system(f"Code moved to revision {target_id[:10]} ({changed} operations applied).")

        if mode == "1":
            data = store.checkout(target_id)
            self._session_revision_id = target_id
            self._restore_session_data(data)
            self.changes.code_revision_id = target_code
        elif mode == "2":
            self._restore_session_data(target)
            self.changes.code_revision_id = current_code
            self._session_revision_id = target_id
            self.save_session(self._session_id, reason="conversation-rewind")
        else:
            self.changes.code_revision_id = target_code
            self.save_session(self._session_id, reason="code-rewind")

        self.console.reset_timeline()
        self._replay_context()
        self.console.system(
            f"Checked out revision {self._session_revision_id[:10]} at round {self.round_id}. "
            "New work will branch from here."
        )

    def _sessions_dir(self) -> Path:
        directory = self._kia_dir() / self.SESSIONS_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _session_data(self) -> dict:
        return {
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

    def save_session(self, name: str | None = None, *, reason: str = "autosave") -> Path:
        name = name or self._session_timestamp()
        if self._session_store is None or self._session_store.session_id != name:
            self._session_store = self._session_store_for(name)
            self._session_revision_id = self._session_store.head_id

        changes = self.changes.pending_changes if self.changes and self.changes.session_id == name else []
        code_parent = self.changes.code_revision_id if self.changes and self.changes.session_id == name else None
        revision_id, code_revision_id, _ = self._session_store.commit(
            self._session_data(),
            parent_id=self._session_revision_id,
            code_parent_id=code_parent,
            changes=changes,
            reason=reason,
        )
        self._session_revision_id = revision_id
        if self.changes and self.changes.session_id == name:
            self.changes.mark_committed(code_revision_id)
        return self._session_store.history_path

    def load_session(self, name: str) -> bool:
        try:
            store = self._session_store_for(name)
            if not store.exists:
                self.console.error(f"Session not found: {name}")
                return False
            data = store.materialize()
        except (OSError, ValueError) as e:
            self.console.error(f"Failed to read session: {e}")
            return False

        self._session_store = store
        self._session_revision_id = store.head_id
        self._session_id = name
        self._restore_session_data(data)
        self.console.system(
            f"Loaded session '{name}' ({len(self.context.messages)} messages, round {self.round_id}, "
            f"revision {self._session_revision_id[:10]})"
        )
        self._replay_context()
        return True

    def _restore_session_data(self, data: dict) -> None:
        saved_model = data.get("model", "")
        if saved_model != self.model:
            self.console.system(
                f"Note: session was saved with model '{saved_model}', current model is '{self.model}'"
            )

        saved_persona = data.get("persona")
        if saved_persona and saved_persona != self.persona.name:
            self.persona = get_persona(saved_persona)
            self.system_prompt = self._build_system_prompt()
            self.context.system_prompt["content"] = self.system_prompt

        self.context.replace_messages(data["messages"])
        self.round_id = data.get("round_id", 0)
        self.token_totals.update(data.get("token_totals") or {})
        self.tool_compaction_totals.update(data.get("tool_compaction_totals") or {})

        available = set(self.skills)
        self.tool_executor._loaded_skills = {
            name for name in data.get("loaded_skills", []) if name in available
        }
        self.tool_executor.reset_skill_tools()
        for name in list(self.tool_executor._loaded_skills):
            error = self.tool_executor._register_skill_tools(name, self.skills[name].get("dir"))
            if error is not None:
                self.console.warn(error)
                self.tool_executor._loaded_skills.discard(name)
        self.tool_executor._skill_loads = {
            name: count for name, count in (data.get("skill_loads") or {}).items()
            if isinstance(count, int)
        }

        self.goal = data.get("goal")
        self.goal_iterations = data.get("goal_iterations", 0)
        goal_supported = self.persona.tools is None or "report_goal" in self.persona.tools
        if self.goal and goal_supported:
            self.goal_active = True
            self._pending_auto = self._build_goal_prompt()
        else:
            self.goal = None
            self.goal_active = False
            self._pending_auto = None

    def _install_change_tracker(self) -> None:
        work_dir = self.tool_executor._work_dir or os.getcwd()
        if self.changes is not None:
            self.changes.close()
        code_revision_id = None
        if self._session_store and self._session_store.exists:
            code_revision_id = self._session_store.materialize().get("code_revision_id")
        self.changes = self._create_change_tracker(
            self._session_id, work_dir, self._session_store, code_revision_id
        )
        self.tool_executor._change_tracker = self.changes
        self.tool_executor._get_round_id = lambda: self.round_id

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

