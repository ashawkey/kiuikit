"""Public Python API for one-shot kia agent runs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from kiui.config import LOCAL_CONFIG_PATH, conf
from kiui.agent.backend import LLMAgent
from kiui.agent.models import ReasoningEffort
from kiui.agent.providers import provider_names
from kiui.agent.ui import AgentConsole
from kiui.agent.utils.interrupt import TurnOutcome


@dataclass(frozen=True)
class AgentRunResult:
    """Result of a completed :func:`run_agent` invocation."""

    response: str | None
    outcome: TurnOutcome
    token_usage: dict[str, int]
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.outcome == TurnOutcome.COMPLETED


def run_agent(
    task: str,
    *,
    model_alias: str | None = None,
    persona: str | None = None,
    work_dir: str | Path | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    stream: bool = False,
    verbose: bool = False,
    quiet: bool = True,
    console: AgentConsole | None = None,
) -> AgentRunResult:
    """Run one independent, non-interactive agent task.

    ``model_alias`` selects an entry under ``openai`` in the loaded kiui
    configuration; omitting it selects the first configured model. The run has
    a fresh conversation, does not create an interactive session or rewind
    history, and always releases provider, process, and skill resources before
    returning.

    Progress output is suppressed by default. Pass ``quiet=False`` to observe
    normal output, optionally through a custom ``console``. Configuration and
    construction errors are raised; provider failures and interruptions are
    represented by ``AgentRunResult.outcome``.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    model_configs = conf.get("openai", {})
    if not isinstance(model_configs, dict) or not model_configs:
        raise ValueError(f"No models found in config: {LOCAL_CONFIG_PATH}")

    alias = model_alias or next(iter(model_configs))
    if alias not in model_configs:
        available = ", ".join(model_configs)
        raise ValueError(f"Model '{alias}' not found in config. Available: {available}")

    model_conf = model_configs[alias]
    provider_name = model_conf.get("provider", "openai")
    if provider_name not in provider_names():
        available = ", ".join(provider_names())
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

    run_console = console or AgentConsole()
    agent = LLMAgent(
        model=model_conf.get("model", alias),
        api_key=model_conf.get("api_key", ""),
        base_url=model_conf.get("base_url", ""),
        provider_name=provider_name,
        model_alias=alias,
        verbose=verbose,
        stream=stream,
        reasoning_effort=reasoning_effort
        or model_conf.get("reasoning_effort", "high"),
        context_length=model_conf.get("context_length"),
        max_output_tokens=model_conf.get("max_output_tokens"),
        persona=persona,
        exec_mode=True,
        work_dir=str(work_dir) if work_dir is not None else None,
        console=run_console,
    )

    try:
        output_context = run_console.suppressed() if quiet else nullcontext()
        with output_context:
            response = agent.execute(task)
        return AgentRunResult(
            response=response,
            outcome=agent._last_turn_outcome,
            token_usage=dict(agent.token_totals),
            error=agent._last_error,
        )
    finally:
        agent.close()


__all__ = ["AgentRunResult", "TurnOutcome", "run_agent"]
