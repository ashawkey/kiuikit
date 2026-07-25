"""Session selection, branchable persistence, loading, and replay."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from rich.table import Table
from rich.text import Text

from kiui.agent.context import CompactionState, get_role, get_text, get_tool_calls
from kiui.agent.personas import get_persona
from kiui.agent.session_store import SessionStore
from kiui.agent.tools import describe_tool_call, format_tool_summary, result_text_failed
from kiui.agent.utils.rewind import CheckoutPlan

_OP_STYLES = {"create": "green", "modify": "yellow", "delete": "red"}

# Human labels for revisions saved by the agent rather than by a user prompt.
_REASON_LABELS = {
    "initial": "(session start)",
    "pre-compaction": "(before compaction)",
    "pre-rewind": "(before rewind)",
    "conversation-rewind": "(after conversation rewind)",
    "code-rewind": "(after code rewind)",
    "resume": "(before resuming another session)",
}


def _reason_label(reason: str) -> str:
    return _REASON_LABELS.get(reason, f"({reason})")


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _file_label(count: int) -> str:
    return f"{count} file{'s' if count != 1 else ''}" if count else "no files"


def _line_label(added: int, removed: int) -> str:
    return " ".join(part for part in (f"+{added}" if added else "", f"-{removed}" if removed else "") if part)


def _relative_time(timestamp: float) -> str:
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 7 * 86400:
        return f"{seconds // 86400}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


class SessionMixin:
    SESSIONS_DIR_NAME = "sessions"
    REWIND_PROMPT_WIDTH = 56
    REWIND_MAX_FILES = 12
    REWIND_MAX_HUNKS = 10
    REWIND_MAX_DROPPED = 6

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
        if not self._session_id or not self.changes or self._session_store is None:
            self.console.warn("Rewind is only available in interactive chat mode with a session.")
            return

        parts = raw.split(maxsplit=1)
        argument = parts[1].strip() if len(parts) > 1 else ""
        store = self._session_store

        if argument.lower() == "list":
            candidates = store.candidates()
            if not candidates:
                self.console.system("No saved revisions yet.")
                return
            self.console.table(self._revision_table(candidates))
            return

        try:
            self.save_session(self._session_id, reason="pre-rewind")
        except Exception as e:
            self.console.error(f"Could not checkpoint current state: {e}")
            return

        candidates = store.candidates()
        if len(candidates) <= 1:
            self.console.system("No earlier revisions to rewind to.")
            return

        if argument:
            try:
                target_id = store.resolve_revision(argument)
            except ValueError as e:
                self.console.error(str(e))
                return
        else:
            target_id = self._pick_revision(candidates)
            if target_id is None:
                self.console.system("Rewind cancelled.")
                return

        if target_id == self._session_revision_id:
            self.console.system("That revision is already checked out.")
            return

        target = store.materialize(target_id)
        plan = self.changes.plan_checkout(target.get("code_revision_id"))
        self._print_rewind_preview(target_id, target, plan)

        mode = self._pick_rewind_mode(target, plan)
        if mode is None:
            self.console.system("Rewind cancelled.")
            return

        if mode == "code":
            self._apply_code_plan(plan, target_id)
            self.save_session(self._session_id, reason="code-rewind")
            # The conversation is untouched, so the transcript on screen still
            # matches it — redrawing would only erase the preview above.
            self.console.system(
                f"Code checked out from revision {target_id[:10]}; conversation unchanged. "
                f"New work will branch from revision {self._session_revision_id[:10]}."
            )
            return

        if mode == "both":
            self._apply_code_plan(plan, target_id)
            data = store.checkout(target_id)
            self._session_revision_id = target_id
            self._restore_session_data(data)
        else:
            self._restore_session_data(target)
            self._session_revision_id = target_id
            self.save_session(self._session_id, reason="conversation-rewind")

        self.console.reset_timeline()
        self._replay_context()
        self.console.system(
            f"Checked out revision {self._session_revision_id[:10]} at round {self.round_id}. "
            "New work will branch from here."
        )

    def _apply_code_plan(self, plan: CheckoutPlan, target_id: str) -> None:
        """Move files to the planned state; this is what advances the code revision."""
        operations = self.changes.apply_plan(plan)
        if plan:
            self.console.system(
                f"Code moved to revision {target_id[:10]}: "
                f"{plan.files} file(s) changed, {operations} operation(s) applied."
            )
        else:
            self.console.system("Code already matched that revision; no files changed.")

    def _pick_revision(self, candidates: list[dict]) -> str | None:
        """Show the revision table, then select one of its rows."""
        self.console.table(self._revision_table(candidates))
        labels = [
            f"{index:>2}. {revision['id'][:10]}  │  round {revision['round_id']}  │  "
            f"{_relative_time(revision['created_at'])}  │  {_file_label(revision['files'])}  │  "
            f"{_shorten(revision['prompt'] or _reason_label(revision['reason']), self.REWIND_PROMPT_WIDTH)}"
            f"{'  [current]' if revision['current'] else ''}"
            for index, revision in enumerate(candidates, start=1)
        ]
        picked = self.console.select(message="Pick a session revision", choices=labels)
        if picked is None:
            return None
        return candidates[labels.index(picked)]["id"]

    def _revision_table(self, candidates: list[dict]) -> Table:
        """Tabulate revisions with the conversation and file context to judge them."""
        table = Table(title=f"Session revisions: {self._session_id}", title_style="bold blue")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Revision", style="cyan", no_wrap=True)
        table.add_column("Round", justify="right", style="blue")
        table.add_column("When", style="dim", no_wrap=True)
        table.add_column("Msgs", justify="right", style="dim")
        table.add_column("Files", justify="right", style="green")
        table.add_column("Saved as", style="magenta")
        table.add_column(
            "Prompt", style="white", no_wrap=True, overflow="ellipsis",
            max_width=self.REWIND_PROMPT_WIDTH,
        )
        for index, revision in enumerate(candidates, start=1):
            delta = revision["new_messages"]
            messages = f"{revision['messages']}" + (f" ({delta:+d})" if delta else "")
            table.add_row(
                str(index),
                revision["id"][:10] + (" [current]" if revision["current"] else ""),
                str(revision["round_id"]),
                _relative_time(revision["created_at"]),
                messages,
                str(revision["files"]) if revision["files"] else "-",
                revision["reason"],
                Text(_shorten(revision["prompt"] or _reason_label(revision["reason"]), self.REWIND_PROMPT_WIDTH)),
            )
        return table

    def _print_rewind_preview(self, target_id: str, target: dict, plan: CheckoutPlan) -> None:
        """Show what checking out *target_id* would do to history and to files."""
        store = self._session_store
        revision = store.revisions[target_id]
        detail = Table.grid(padding=(0, 1))
        detail.add_column(width=12, style="dim", no_wrap=True)
        detail.add_column(overflow="fold")

        detail.add_row(
            "Revision",
            Text(
                f"{target_id[:10]}  ·  round {target.get('round_id', 0)}  ·  "
                f"{_relative_time(revision['createdAt'])}  ·  saved as {revision['reason']}"
            ),
        )
        detail.add_row("Prompt", Text(store.revision_prompt(target_id) or "(no user message yet)"))
        detail.add_row(
            "Conversation",
            Text(
                f"{len(self.context.messages)} → {len(target['messages'])} messages, "
                f"round {self.round_id} → {target.get('round_id', 0)}"
            ),
        )
        for line, style in self._dropped_rounds(target_id):
            detail.add_row("", Text(line, style=style))

        if plan:
            detail.add_row(
                "Files",
                Text(f"{plan.files} will change  (+{plan.added} / -{plan.removed})", style="yellow"),
            )
            for delta in plan.deltas[: self.REWIND_MAX_FILES]:
                detail.add_row(
                    "",
                    Text(f"{delta.op:<7} {delta.path}", style=_OP_STYLES[delta.op]).append(
                        f"   {_line_label(*store.text_stats(delta.before, delta.after))}", style="dim"
                    ),
                )
            hidden = plan.files - self.REWIND_MAX_FILES
            if hidden > 0:
                detail.add_row("", Text(f"… and {hidden} more file(s)", style="dim"))
            if plan.dirty:
                detail.add_row(
                    "",
                    Text(
                        f"! {len(plan.dirty)} file(s) changed on disk since they were recorded "
                        f"and would be overwritten: {', '.join(plan.dirty[:5])}",
                        style="bold red",
                    ),
                )
        else:
            detail.add_row("Files", Text("no files will change", style="green"))

        self.console.print(detail)

    def _dropped_rounds(self, target_id: str) -> list[tuple[str, str]]:
        """Describe the rounds between the current revision and *target_id*."""
        store = self._session_store
        chain = store.revision_ancestors(self._session_revision_id)
        if target_id not in chain:
            return [("target is on a different branch — messages are replaced, not truncated", "yellow")]

        # Several revisions can share a round (autosave, pre-compaction, pre-rewind);
        # the target's own round is not "dropped", only the rounds built on top of it.
        target_round = store.revisions[target_id]["state"].get("round_id", 0)
        seen: set[int] = {target_round}
        dropped: list[tuple[int, str]] = []
        for revision_id in chain[: chain.index(target_id)]:
            round_id = store.revisions[revision_id]["state"].get("round_id", 0)
            if round_id in seen:
                continue
            seen.add(round_id)
            dropped.append((round_id, store.revision_prompt(revision_id)))
        if not dropped:
            return []

        lines = [(f"{len(dropped)} round(s) will be dropped:", "dim")]
        for round_id, prompt in dropped[: self.REWIND_MAX_DROPPED]:
            lines.append((f"  round {round_id}  {_shorten(prompt, self.REWIND_PROMPT_WIDTH)}", "dim"))
        if len(dropped) > self.REWIND_MAX_DROPPED:
            lines.append((f"  … and {len(dropped) - self.REWIND_MAX_DROPPED} more", "dim"))
        return lines

    def _pick_rewind_mode(self, target: dict, plan: CheckoutPlan) -> str | None:
        """Select a rewind mode from options labelled with their actual effect.

        Returns ``"both"``, ``"conversation"``, ``"code"``, or ``None`` to cancel.
        """
        conversation = f"{len(self.context.messages)} → {len(target['messages'])} messages"
        code = (
            f"{plan.files} file(s) reverted (+{plan.added} / -{plan.removed})" if plan
            else "no file changes"
        )
        modes = ["both", "conversation", "code"]
        while True:
            choices = [
                f"1. Conversation + code — {conversation}, {code}",
                f"2. Conversation only — {conversation}, files untouched",
                f"3. Code only — conversation kept, {code}",
            ]
            if plan:
                choices.append(f"{len(choices) + 1}. Show the file diffs")
            choices.append(f"{len(choices) + 1}. Cancel")

            picked = self.console.select(message="Choose rewind mode", choices=choices)
            if picked is None:
                return None
            index = choices.index(picked)
            if index < len(modes):
                return modes[index]
            if plan and index == len(modes):
                self._show_rewind_diffs(plan)
                continue
            return None

    def _show_rewind_diffs(self, plan: CheckoutPlan) -> None:
        """Render each file the checkout would touch, hunk by hunk."""
        store = self._session_store
        for delta in plan.deltas[: self.REWIND_MAX_FILES]:
            hunks = store.text_hunks(delta.before, delta.after)
            if not hunks:
                self.console.system(f"{delta.path}: {delta.op} (binary or directory — no text diff)")
                continue
            for line_num, old_chunk, new_chunk in hunks[: self.REWIND_MAX_HUNKS]:
                self.console.diff_edit(
                    path=delta.path,
                    old_text=old_chunk,
                    new_text=new_chunk,
                    line_num=line_num,
                )
            if len(hunks) > self.REWIND_MAX_HUNKS:
                self.console.system(
                    f"{delta.path}: {len(hunks) - self.REWIND_MAX_HUNKS} more hunk(s) not shown"
                )
        hidden = plan.files - self.REWIND_MAX_FILES
        if hidden > 0:
            self.console.system(f"{hidden} more file(s) not shown")

    def _sessions_dir(self) -> Path:
        directory = self._kia_dir() / self.SESSIONS_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _session_data(self) -> dict:
        return {
            "model": self.model,
            "model_alias": self.model_alias,
            "provider": self.provider_name,
            "round_id": self.round_id,
            "token_totals": self.token_totals,
            "tool_compaction_totals": self.tool_compaction_totals,
            "compaction_totals": self.compaction_totals,
            "system_prompt": self.context.system_prompt,
            "persona": self.persona.name,
            "persona_digest": self.persona.digest,
            "goal": self.goal,
            "goal_active": self.goal_active,
            "goal_iterations": self.goal_iterations,
            "loaded_skills": sorted(self.tool_executor._loaded_skills),
            "skill_loads": self.tool_executor._skill_loads,
            "messages": self.context.messages,
            "compaction_state": asdict(self.context.compaction_state),
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

        try:
            self._restore_session_data(data)
        except ValueError as e:
            self.console.error(f"Failed to restore session: {e}")
            return False

        self._session_store = store
        self._session_revision_id = store.head_id
        self._session_id = name
        self.console.system(
            f"Loaded session '{name}' ({len(self.context.messages)} messages, round {self.round_id}, "
            f"revision {self._session_revision_id[:10]})"
        )
        self._replay_context()
        return True

    def _restore_session_data(self, data: dict) -> None:
        saved_model = data.get("model", "")
        saved_provider = data.get("provider", "openai")
        if saved_model != self.model or saved_provider != self.provider_name:
            self.console.system(
                f"Note: session was saved with '{saved_provider}/{saved_model}', "
                f"current model is '{self.provider_name}/{self.model}'"
            )

        saved_persona = data.get("persona")
        if saved_persona:
            persona = get_persona(saved_persona, personas=self.personas)
            saved_digest = data.get("persona_digest")
            if saved_digest and saved_digest != persona.digest:
                self.console.warn(
                    f"Persona '{saved_persona}' changed since this session was saved."
                )
            if persona != self.persona:
                self.persona = persona
                self.system_prompt = self._build_system_prompt()
                self.context.system_prompt["content"] = self.system_prompt

        self.context.replace_messages(data["messages"])
        # Sessions saved before compaction state was carried structurally simply
        # start empty; the next pass rebuilds it from whatever history remains.
        carried = data.get("compaction_state") or {}
        self.context.compaction_state = CompactionState(
            original_request=carried.get("original_request", ""),
            read_files=tuple(carried.get("read_files", ())),
            modified_files=tuple(carried.get("modified_files", ())),
            skills=tuple(carried.get("skills", ())),
        )
        self.round_id = data.get("round_id", 0)
        self.token_totals.update(data.get("token_totals") or {})
        self.tool_compaction_totals.update(data.get("tool_compaction_totals") or {})
        self.compaction_totals.update(data.get("compaction_totals") or {})

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
                    self.console.tool(describe_tool_call(fname, fargs))

            elif role == "tool":
                result_text = get_text(msg)
                summary = format_tool_summary(result_text)
                self.console.tool_result(summary, success=not result_text_failed(result_text))

        self.console.system("── End of replay ──")

