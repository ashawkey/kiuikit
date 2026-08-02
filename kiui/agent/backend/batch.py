"""Context-isolated turns: run a prompt, keep the result, discard the context.

A repetitive workload (caption these 1000 images, classify these 5000 rows) is
a sequence of *independent* tasks. Running them as ordinary turns makes every
item pay for every earlier item — quadratic prompt cost — and eventually forces
compaction to summarize the very results the user asked for.

:meth:`IsolatedTurnMixin.run_isolated_turn` is the mechanism that breaks that
coupling: one turn runs against an empty message history, and the enclosing
conversation is restored byte-for-byte afterwards. Repeated calls therefore
cost a constant system/tool prefix rather than a growing one, and nothing an
item did can leak into the next one.

The mechanism lives here because it needs the agent's context and agentic loop.
The *policy* built on it — how items are enumerated, where results are written,
when to give up — belongs to the ``batch`` skill, which drives this through
``ToolExecutor.isolated_turn``.
"""

from kiui.agent.context import CompactionState
from kiui.agent.utils.interrupt import TurnOutcome


class IsolatedTurnMixin:
    """Run agentic turns that leave no trace in the conversation."""

    def run_isolated_turn(self, prompt: str) -> tuple[str | None, TurnOutcome]:
        """Run one full agentic turn for *prompt*, then discard its context.

        Returns ``(response, outcome)``. *response* is the assistant's final
        text, or ``None`` when the turn produced none. A user-interrupt outcome
        tells a caller to stop rather than record an ordinary failure.

        The caller keeps the returned value; conversational state is rolled
        back: messages, compaction state, the compaction floor, enclosing turn
        flags, images queued for the next request, and which skills are loaded.
        That restoration is unconditional, so a failing turn cannot poison the
        conversation the caller resumes with. Usage accounting and external tool
        effects remain.

        Skill state is part of the rollback because it lives on the executor,
        not in the message history: a skill loaded by one item must not remain
        loaded or keep its contributed tools registered in later items or the
        enclosing conversation. Already-loaded skills may be loaded again, so
        an item can still obtain instructions that are absent from its isolated
        message history.

        Isolation covers the two things a discarded turn must not touch on its
        way past: it is never rendered or published (see
        :meth:`AgentConsole.suppressed`), and it never commits a session
        revision — a compaction inside an item would otherwise move the durable
        head onto that item's context.

        Isolated turns must not nest: an inner restore would resurrect the outer
        turn's discarded context as if it were the conversation.
        """
        if self._isolated_turn_active:
            raise RuntimeError("Isolated turns cannot nest.")
        if self.cancellation is not None and self.cancellation.cancelled:
            return None, TurnOutcome.USER_INTERRUPTED

        snapshot = list(self.context.messages)
        compaction_state = self.context.compaction_state
        compaction_floor = self._compaction_floor_tokens
        pending_images = list(self._pending_images)
        skill_state = self.tool_executor.skill_state()
        outer_interrupted = self._last_interrupted
        outer_interrupt_reverts_prompt = self._interrupt_reverts_prompt
        outer_finish_reason = self._last_finish_reason
        outer_turn_outcome = self._last_turn_outcome

        self._isolated_turn_active = True
        try:
            # The enclosing assistant message still has the run_batch tool call
            # open, so it is not a valid prefix for another provider request.
            # Starting from an empty history also gives every item the promised
            # independent context: only the shared system prompt and tools remain.
            self.context.replace_messages([])
            self.context.compaction_state = CompactionState()
            self._compaction_floor_tokens = None
            self._pending_images.clear()
            self.context.add({"role": "user", "content": prompt})
            # The turn is not part of the conversation, so it is not rendered or
            # published either: streaming hundreds of discarded item turns would
            # bury the transcript and evict the real timeline from the bounded
            # event history that reconnecting web clients replay.
            with self.console.suppressed():
                response = self.get_response()
            return response, self._last_turn_outcome
        finally:
            self._isolated_turn_active = False
            # replace_messages copies, so the snapshot stays reusable even
            # though eviction and compaction may have rewritten the live list.
            self.context.replace_messages(snapshot)
            self.context.compaction_state = compaction_state
            self._compaction_floor_tokens = compaction_floor
            self._pending_images.clear()
            self._pending_images.extend(pending_images)
            self.tool_executor.restore_skill_state(skill_state)
            # get_response uses these fields to decide whether the enclosing
            # prompt may be withdrawn after cancellation. An isolated item must
            # report its status through the return value without overwriting the
            # enclosing turn's rollback state.
            self._last_interrupted = outer_interrupted
            self._last_turn_outcome = outer_turn_outcome
            self._interrupt_reverts_prompt = outer_interrupt_reverts_prompt
            self._last_finish_reason = outer_finish_reason
