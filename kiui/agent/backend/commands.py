"""Interactive slash commands for :class:`LLMAgent`."""

from kiui.agent.context import (
    build_tool_name_index,
    compact_context,
    estimate_context_chars,
    get_role,
    get_text,
    get_tool_call_id,
    get_tool_calls,
    msg_chars,
)
from kiui.agent.models import REASONING_EFFORTS, resolve_model_profile
from kiui.agent.permissions import PermissionMode
from kiui.agent.personas import DEFAULT_PERSONA, PersonaInfo, get_persona, list_personas


class AgentCommandsMixin:
    COMMANDS = {"help", "compact", "usage", "exit", "quit", "clear", "resume", "perm", "model", "reasoning", "context", "rewind", "skills", "goal", "system_prompt", "persona"}

    def _handle_command(self, raw: str) -> bool:
        """Handle a /command.  Returns True if the agent loop should stop."""
        cmd = raw.split()[0][1:].lower()

        if cmd in ("exit", "quit"):
            return True

        if cmd == "help":
            self._cmd_help()
        elif cmd == "compact":
            self._cmd_compact()
        elif cmd == "usage":
            self._cmd_usage()
        elif cmd == "clear":
            self._cmd_clear()
        elif cmd == "resume":
            self._cmd_resume(raw)
        elif cmd == "perm":
            self._cmd_perm(raw)
        elif cmd == "model":
            self._cmd_model(raw)
        elif cmd == "reasoning":
            self._cmd_reasoning(raw)
        elif cmd == "context":
            self._cmd_context()
        elif cmd == "system_prompt":
            self._cmd_system_prompt()
        elif cmd == "rewind":
            self._cmd_rewind(raw)
        elif cmd == "skills":
            self._cmd_skills(raw)
        elif cmd == "persona":
            self._cmd_persona(raw)
        elif cmd == "goal":
            self._cmd_goal(raw)

        return False

    def _cmd_help(self):
        self.console.print(
            "[bold blue]Available commands:[/bold blue]\n"
            "\n"
            "  [cyan]!<cmd>[/cyan]       — Run a shell command directly (e.g. !ls, !git diff)\n"
            "  [cyan]/help[/cyan]        — Show this help message\n"
            "  [cyan]/context[/cyan]     — Show a concise one-line-per-message context log\n"
            "  [cyan]/system_prompt[/cyan] — Print the current full system prompt\n"
            "  [cyan]/compact[/cyan]     — Force context compaction via LLM summarization\n"
            "  [cyan]/usage[/cyan]       — Show token usage for this session\n"
            "  [cyan]/model[/cyan]       — Show or switch LLM model (/model <name>)\n"
            "  [cyan]/reasoning[/cyan]   — Show or set reasoning effort (none|minimal|low|medium|high|xhigh)\n"
            "  [cyan]/skills[/cyan]      — List skills; /skills <name> to load one, /skills reload to re-scan\n"
            "  [cyan]/persona[/cyan]     — List personas; /persona <name> to switch (restarts conversation)\n"
            "  [cyan]/goal[/cyan]        — Set a goal the agent auto-iterates toward (/goal <text> | clear)\n"
            "  [cyan]/perm[/cyan]        — Show or change permission mode (/perm auto|default|strict)\n"
            "  [cyan]/rewind[/cyan]      — Roll back conversation and/or code to a previous round\n"
            "  [cyan]/clear[/cyan]       — Clear conversation history (keep system prompt)\n"
            "  [cyan]/resume[/cyan]      — Save current, then resume a previous session (/resume [session_id])\n"
            "  [cyan]/exit[/cyan]        — Exit the agent (also: /quit)\n"
            "\n"
            "  Press [bold]Enter[/bold] to send, [bold]Escape → Enter[/bold] for a newline."
        )

    def _cmd_compact(self):
        if len(self.context.messages) <= 2:
            self.console.system("Not enough messages to compact.")
            return
        before_chars = estimate_context_chars(self.context.messages)
        before_msgs = len(self.context.messages)
        before_tokens = self.token_estimator.chars_to_tokens(before_chars)
        with self.console.thinking(
            label="Compacting",
            progress=True,
            status_suffix=f"{before_msgs} messages, ~{before_tokens:,} tokens",
        ):
            self.context.replace_messages(
                compact_context(
                    self.context.messages, self.client, self.model,
                    console=self.console,
                    context_length=self.context_length,
                    chars_per_token=self.token_estimator.chars_per_token,
                )
            )
        after_chars = estimate_context_chars(self.context.messages)
        after_msgs = len(self.context.messages)
        after_tokens = self.token_estimator.chars_to_tokens(after_chars)
        saved_pct = (1 - after_chars / before_chars) * 100 if before_chars else 0
        self.console.system(
            f"Compaction complete: "
            f"{before_msgs} messages → {after_msgs} messages, "
            f"~{before_tokens:,} tokens → ~{after_tokens:,} tokens "
            f"(saved {saved_pct:.0f}%)"
        )

    def _cmd_usage(self):
        ctx_chars = estimate_context_chars(self.context.messages)
        ctx_tokens = self.token_estimator.chars_to_tokens(ctx_chars)
        ctx_pct = ctx_tokens / self.context_length * 100 if self.context_length else 0

        self.console.print(
            f"[bold blue]Session usage (round {self.round_id}):[/bold blue]\n"
            f"  Total tokens   : {self.token_totals['total']}\n"
            f"  Prompt tokens  : {self.token_totals['prompt']}  (cached: {self.token_totals['cached_prompt']})\n"
            f"  Output tokens  : {self.token_totals['completion']}  (reasoning: {self.token_totals['reasoning']})\n"
            f"  Context window : ~{ctx_tokens} / {self.context_length} tokens [{ctx_pct:.0f}%]\n"
            f"  Messages       : {len(self.context.messages)}\n"
            f"  Tool compaction: {self._tool_compaction_summary()}"
        )

        skill_loads = self.tool_executor._skill_loads
        if skill_loads:
            summary = ", ".join(
                f"{n} ({c}\u00d7)"
                for n, c in sorted(skill_loads.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            self.console.print(f"  Skills loaded  : {summary}")

    def _cmd_clear(self):
        self._restart_session()

    def _cmd_persona(self, raw: str = "/persona"):
        """List personas, or switch to one (switching restarts the conversation).

        Usage: ``/persona`` (list) | ``/persona <name>`` (switch + restart).
        """
        parts = raw.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            try:
                personas = list_personas()
            except ValueError as e:
                self.console.error(str(e))
                return
            self.console.print(f"[bold blue]Installed personas ({len(personas)}):[/bold blue]\n")
            for name, info in personas.items():
                current = " [green](current)[/green]" if name == self.persona.name else ""
                default = " [dim](default)[/dim]" if name == DEFAULT_PERSONA else ""
                tools = "all tools" if info.tools is None else (
                    f"tools: {', '.join(sorted(info.tools))}" if info.tools else "no tools"
                )
                self.console.print(f"  [cyan]{name}[/cyan]{default}{current}")
                if info.description:
                    self.console.print(f"    {info.description}")
                self.console.print(f"    [dim]{tools}[/dim]")
            self.console.print(
                "\n[dim]/persona <name> to switch (restarts the conversation)[/dim]"
            )
            return

        target = parts[1].strip()
        if target == self.persona.name:
            self.console.system(f"Already using persona '{target}'.")
            return
        try:
            persona = get_persona(target)
        except ValueError as e:
            self.console.error(str(e))
            return

        self._switch_persona(persona)

    def _switch_persona(self, persona: PersonaInfo):
        """Apply a new persona (prompt + tool surface) and restart the conversation.

        A persona switch is a full restart: the new prompt and tool whitelist
        would otherwise conflict with the existing history (old tool calls the
        new persona no longer offers, context sections it no longer shows).
        """
        # Save and clear while the old persona still owns the conversation.
        self._restart_session()
        self.persona = persona
        self.tools = self._get_tool_definitions()
        self.system_prompt = self._build_system_prompt()
        self.context.system_prompt["content"] = self.system_prompt
        self.console.system(f"Switched to persona: {persona.name}")


    def _cmd_system_prompt(self):
        self.console.print(
            f"[bold blue]System prompt:[/bold blue]\n\n{self.context.system_prompt['content']}"
        )

    def _cmd_context(self):
        msgs = self.context.messages
        if not msgs:
            self.console.system("Context is empty (no messages).")
            return

        total_chars = 0
        lines = [f"[bold blue]Context log ({len(msgs)} messages):[/bold blue]"]

        tc_id_to_name = build_tool_name_index(msgs)

        for idx, m in enumerate(msgs):
            role = get_role(m)
            text = get_text(m)
            chars = msg_chars(m)
            total_chars += chars

            preview = text.replace("\n", " ").strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."

            role_tag = f"[cyan]{role:>9}[/cyan]"
            size_tag = f"[dim]{chars:>6} ch[/dim]"

            tcs = get_tool_calls(m) if role == "assistant" else []
            if tcs:
                n_tc = len(tcs)
                tc_names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs)
                extra = f"[yellow]{n_tc} call{'s' if n_tc > 1 else ''}[/yellow] ({tc_names})"
                if preview:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}  {preview}")
                else:
                    lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {extra}")
            elif role == "tool":
                tid = get_tool_call_id(m)
                tool_name = tc_id_to_name.get(tid, "?") if tid else "?"
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  [magenta]({tool_name})[/magenta]  {preview}")
            else:
                lines.append(f"  [dim]#{idx:<3}[/dim] {role_tag} {size_tag}  {preview}")

        est_tokens = self.token_estimator.chars_to_tokens(total_chars)
        ctx_pct = est_tokens / self.context_length * 100 if self.context_length else 0
        lines.append(f"\n  [bold]Total:[/bold] ~{est_tokens:,} tokens / {self.context_length:,} [{ctx_pct:.0f}%]")

        self.console.print("\n".join(lines))

    def _cmd_perm(self, raw: str):
        parts = raw.split()
        if len(parts) < 2:
            mode = self.permissions.mode.value
            allowed = ", ".join(sorted(self.permissions.session_allowed_tools)) or "(none)"
            self.console.print(
                f"[bold blue]Permission mode:[/bold blue] [cyan]{mode}[/cyan]\n"
                f"  Session-allowed tools: {allowed}\n"
                f"  Usage: [cyan]/perm auto|default|strict[/cyan]"
            )
            return

        target = parts[1].lower()
        valid = {m.value for m in PermissionMode}
        if target not in valid:
            self.console.error(f"Unknown mode '{target}'. Choose from: {', '.join(sorted(valid))}")
            return

        new_mode = PermissionMode(target)
        self.permissions.mode = new_mode
        self.permissions.reset_session()
        self.console.system(f"Permission mode changed to: {new_mode.value}")

    def _cmd_reasoning(self, raw: str):
        parts = raw.split(maxsplit=1)
        if len(parts) == 1:
            self.console.system(
                f"Reasoning: {self.profile.reasoning or 'unsupported'}, effort: {self.reasoning_effort}"
            )
            return
        effort = parts[1].strip().lower()
        if effort not in REASONING_EFFORTS:
            self.console.error(f"Invalid reasoning effort '{effort}'. Choose: {', '.join(REASONING_EFFORTS)}")
            return
        self.reasoning_effort = effort
        self.console.system(f"Reasoning effort set to {effort}.")

    def _cmd_model(self, raw: str):
        from kiui.config import conf

        parts = raw.split(maxsplit=1)
        openai_conf = conf.get("openai", {})

        if len(parts) < 2 or not parts[1].strip():
            lines = [
                f"[bold blue]Current model:[/bold blue] [cyan]{self.model}[/cyan]"
                + (f" (alias: {self.model_alias})" if self.model_alias else ""),
                "[bold blue]Available models:[/bold blue]",
            ]
            for name, mc in openai_conf.items():
                marker = " [green]◀[/green]" if name == self.model_alias else ""
                lines.append(f"  [cyan]{name}[/cyan] → {mc.get('model', name)}{marker}")
            lines.append("\n  Usage: [cyan]/model <name>[/cyan]")
            self.console.print("\n".join(lines))
            return

        target = parts[1].strip()
        if target not in openai_conf:
            self.console.error(f"Model '{target}' not found in config. Use /model to list available models.")
            return

        if target == self.model_alias:
            self.console.system(f"Already using model '{target}'.")
            return

        model_conf = openai_conf[target]
        self.model = model_conf.get("model", target)
        self.model_alias = target
        self._api_key = model_conf.get("api_key", "")
        self._base_url = model_conf.get("base_url", "")
        self.client = self._request_client()
        self.profile = resolve_model_profile(self.model, self.model_alias)
        self.tools = self._get_tool_definitions()
        self.context_length = model_conf.get("context_length", self.profile.context_length)
        self.max_output_tokens = model_conf.get("max_output_tokens", self.profile.max_output_tokens)
        self.reasoning_effort = model_conf.get("reasoning_effort", self.reasoning_effort)
        self.show_thinking = self.profile.reasoning is not None

        if self.subagent_manager:
            self.subagent_manager.model_alias = target
            self.subagent_manager.reasoning_effort = self.reasoning_effort

        self.console.system(
            f"Switched to model: {self.model} "
            f"(context: {self.context_length:,} tokens, reasoning: "
            f"{self.profile.reasoning or 'none'}/{self.reasoning_effort})"
        )

