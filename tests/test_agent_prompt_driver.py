"""The terminal may only ever run one prompt application at a time."""

import asyncio

import pytest

from kiui.agent.terminal import PromptDriver


class _FakeTerminal:
    """Mimics prompt_toolkit: one Application per session, reused per prompt."""

    def __init__(self):
        self.running = False
        self.starts = 0
        self.defaults = []
        self.text = ""

    async def prompt_async(self, default: str = "") -> str:
        assert not self.running, "Application is already running."
        self.running = True
        self.starts += 1
        self.defaults.append(default)
        try:
            await asyncio.sleep(3600)
        finally:
            # prompt_toolkit only clears the flag once its teardown finishes.
            await asyncio.sleep(0)
            self.running = False
        return ""


async def _ask(driver, answers, label, hold=0.0):
    async with driver.paused():
        await asyncio.sleep(hold)
        answers.append(label)
        return label


def test_overlapping_asks_never_start_two_prompts():
    asyncio.run(_test_overlapping_asks_never_start_two_prompts())


async def _test_overlapping_asks_never_start_two_prompts():
    """A second ask arriving before the first finished must queue, not overlap.

    Reproduces the crash after `/rewind`: a browser or Escape answer resolves the
    prompt broker from another thread, so the agent's next ask reaches the UI
    loop while the previous pause is still tearing the editor down.
    """
    terminal = _FakeTerminal()
    driver = PromptDriver(terminal)
    driver.start()
    await asyncio.sleep(0)
    assert terminal.running

    answers: list[str] = []
    await asyncio.gather(
        _ask(driver, answers, "revision", hold=0.02),
        _ask(driver, answers, "mode"),
        _ask(driver, answers, "diffs"),
    )

    assert sorted(answers) == ["diffs", "mode", "revision"]
    assert terminal.running
    assert not driver.suspended
    await driver.stop()


def test_a_cancelled_ask_still_reopens_the_editor_with_its_draft():
    asyncio.run(_test_a_cancelled_ask_still_reopens_the_editor_with_its_draft())


async def _test_a_cancelled_ask_still_reopens_the_editor_with_its_draft():
    terminal = _FakeTerminal()
    driver = PromptDriver(terminal)
    driver.start()
    await asyncio.sleep(0)

    terminal.text = "half typed input"
    ask = asyncio.create_task(_ask(driver, [], "cancelled", hold=3600))
    await asyncio.sleep(0.02)
    ask.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ask

    assert terminal.running
    assert terminal.defaults[-1] == "half typed input"
    assert not driver.suspended
    await driver.stop()


def test_restart_replaces_a_finished_prompt():
    asyncio.run(_test_restart_replaces_a_finished_prompt())


async def _test_restart_replaces_a_finished_prompt():
    terminal = _FakeTerminal()
    driver = PromptDriver(terminal)
    driver.start()
    await asyncio.sleep(0)

    await driver.restart("retry me")
    await asyncio.sleep(0)
    assert terminal.running
    assert terminal.starts == 2
    assert terminal.defaults[-1] == "retry me"
    await driver.stop()
    assert not terminal.running


def test_starting_a_second_prompt_directly_is_rejected():
    asyncio.run(_test_starting_a_second_prompt_directly_is_rejected())


async def _test_starting_a_second_prompt_directly_is_rejected():
    terminal = _FakeTerminal()
    driver = PromptDriver(terminal)
    driver.start()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="already running"):
        driver.start()
    await driver.stop()
