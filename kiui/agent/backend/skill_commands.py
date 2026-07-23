"""Skill discovery and interactive skill commands."""

from pathlib import Path


class SkillCommandsMixin:
    def _cmd_skills(self, raw: str = "/skills"):
        """List, reload, or manually load skills.

        Usage: ``/skills`` (list) | ``/skills reload`` | ``/skills <name>`` (load).
        """
        parts = raw.split()
        arg = parts[1] if len(parts) > 1 else ""

        if not arg:
            self._list_skills()
            return
        if arg.lower() == "reload":
            self._reload_skills()
            return
        # Any other argument is treated as a skill name to load.
        self._load_skill_into_context(arg)

    def _report_skill_issues(self, issues: dict):
        """Warn about shadowed and malformed skills surfaced by discover_skills.

        Non-fatal: discovery already dropped these entries. Shown once at init
        and on every ``/skills reload`` so users notice name collisions or
        broken SKILL.md files instead of them vanishing silently.
        """
        for err in issues.get("errors", []):
            self.console.warn(
                f"skill '{err['name']}' ignored ({err['reason']}): {err['path']}"
            )
        shadowed = issues.get("shadowed", [])
        if shadowed:
            names = sorted({sh["name"] for sh in shadowed})
            preview = ", ".join(names[:5])
            if len(names) > 5:
                preview += f", … (+{len(names) - 5} more)"
            self.console.warn(
                f"{len(shadowed)} lower-precedence duplicate skill(s) ignored: {preview}"
            )

    def _skills_summary(self) -> str:
        """Return discovered skill count and its share of the system prompt."""
        from kiui.agent.skills import build_skills_prompt_section

        skills_section = build_skills_prompt_section(self.skills)
        total_tokens = self.token_estimator.chars_to_tokens(len(self.system_prompt))
        skill_tokens = self.token_estimator.chars_to_tokens(len(skills_section))
        percent = 100 * skill_tokens / total_tokens if total_tokens else 0
        return f"{len(self.skills)} available · ~{skill_tokens:,} tokens ({percent:.1f}% of prompt)"

    def _report_skills_summary(self):
        """Report discovered skill count and prompt share."""
        self.console.system(self._skills_summary())

    def _list_skills(self):
        """List bundled, project, and personal kia skills."""
        if not self.skills:
            from kiui.agent.skills import SKILL_DIRS
            base = Path(self.tool_executor._work_dir) if self.tool_executor._work_dir else Path.cwd()
            skills_dir = base / SKILL_DIRS[0] / "skills"
            searched = f"{SKILL_DIRS[0]}/skills/"
            self.console.print(
                f"[bold blue]No skills available.[/bold blue]\n"
                f"\n"
                f"  Skills are folders each containing a [cyan]SKILL.md[/cyan] file, searched in: [cyan]{searched}[/cyan]\n"
                f"  (under both the project dir and your home dir)\n"
                f"\n"
                f"  [bold]Example:[/bold]\n"
                f"    [cyan]{skills_dir / 'git-workflow' / 'SKILL.md'}[/cyan]\n"
                f"\n"
                f"  Each SKILL.md starts with YAML frontmatter (name + description required),\n"
                f"  followed by markdown instructions:\n"
                f"    [dim]---\\nname: git-workflow\\ndescription: ... when to use it ...\\n---\\n<instructions>[/dim]\n"
                f"  When a skill is relevant, the model can invoke [cyan]load_skill[/cyan], or you can load\n"
                f"  one manually with [cyan]/skills <name>[/cyan]."
            )
            return

        self.console.print(f"[bold blue]Available skills ({len(self.skills)}):[/bold blue]\n")
        from kiui.agent.skills import validate_skill
        for name, info in sorted(self.skills.items()):
            desc = info.get("description", "")
            path = info.get("path", "")
            loaded = " [green](loaded)[/green]" if name in self.tool_executor._loaded_skills else ""
            inactive = " [dim](manual only)[/dim]" if not info.get("active", True) else ""
            loads = self.tool_executor._skill_loads.get(name, 0)
            uses = f" [dim]· loaded {loads}×[/dim]" if loads else ""
            self.console.print(f"  [cyan]{name}[/cyan]{loaded}{inactive}{uses}")
            self.console.print(f"    {desc}")
            self.console.print(f"    [dim]{path}[/dim]")
            for warning in validate_skill(name, info.get("frontmatter", {})):
                self.console.print(f"    [yellow]! {warning}[/yellow]")
            self.console.print()
        self.console.print(
            "[dim]/skills <name> to load one manually · /skills reload to re-scan[/dim]"
        )

    def _reload_skills(self):
        """Re-discover skills from disk and refresh the system prompt.

        Picks up skills created or edited during the session (e.g. via
        skill-creator). Already-loaded skill bodies remain in the conversation;
        their loaded state is preserved when the skill still exists.
        """
        from kiui.agent.skills import discover_skills

        before = set(self.skills)
        issues: dict = {}
        self.skills = discover_skills(self.work_dir, issues=issues)
        self.tool_executor._skills = self.skills
        # Drop loaded-state and any contributed tools for skills that no longer exist.
        removed_skills = self.tool_executor._loaded_skills - set(self.skills)
        for gone in removed_skills:
            self.tool_executor.unregister_skill_tools(gone)
        self.tool_executor._loaded_skills &= set(self.skills)
        self._report_skill_issues(issues)

        # Rebuild the system prompt so the advertised skill list stays current.
        # The advertised tool surface follows the registry automatically via the
        # `tools` property, so removed skills' tools stop being advertised.
        self.system_prompt = self._build_system_prompt()
        self.context.system_prompt["content"] = self.system_prompt
        self._report_skills_summary()

        after = set(self.skills)
        added = sorted(after - before)
        removed = sorted(before - after)
        summary = f"Reloaded skills ({len(self.skills)} available)."
        if added:
            summary += f" Added: {', '.join(added)}."
        if removed:
            summary += f" Removed: {', '.join(removed)}."
        self.console.system(summary)

    def _load_skill_into_context(self, name: str):
        """Manually load a skill's instructions into the conversation context.

        Reuses the load_skill tool path so behavior matches model-invoked loads,
        then injects the body as a user message so it takes effect on the next
        turn. Lets the user force a skill the model did not auto-select.
        """
        result = self.tool_executor._load_skill(name)
        if not result.get("success"):
            self.console.warn(result.get("error", f"Could not load skill '{name}'."))
            return

        if "content" not in result:
            # Already loaded earlier in this session; nothing new to inject.
            self.console.system(result.get("message", f"Skill '{name}' is already loaded."))
            return

        self.context.add({
            "role": "user",
            "content": f"[Manually loaded skill '{name}']\n\n{result['content']}",
        })
        # Any tools the skill contributes are advertised automatically on the
        # next turn via the `tools` property.
        self.console.system(
            f"Loaded skill '{name}' into context. It will guide the next response."
        )

