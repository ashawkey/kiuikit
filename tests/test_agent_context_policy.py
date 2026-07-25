"""Tests for per-tool output retention during context compaction."""

import json

import pytest

from kiui.agent.context import (
    COMPACTION_MIN_YIELD_RATIO,
    SUMMARY_MARKER,
    TokenEstimator,
    ToolResultEnvelope,
    compact_context,
    compact_tool_result_envelope,
    compaction_target_tokens,
    compaction_trigger_tokens,
    eviction_trigger_tokens,
    get_text,
    msg_chars,
    needs_compaction,
    prune_context,
)


@pytest.mark.parametrize(
    "tool_name",
    ["read_file", "web_fetch", "ls", "glob_files", "grep_files"],
)
def test_document_and_search_compaction_keeps_prefix(tool_name):
    text = "BEGINNING\n" + "middle\n" * 3000 + "END\n"

    result = compact_tool_result_envelope(
        ToolResultEnvelope(tool_name, {}, {}, text),
        16_000,
    )

    assert result.retained_chars < result.original_chars
    assert "BEGINNING" in result.text
    assert "END" not in result.text


def test_exec_compaction_keeps_latest_output_and_diagnostics():
    text = "BEGINNING\n" + "middle\n" * 3000 + "ERROR: failed\nLATEST\n"

    result = compact_tool_result_envelope(
        ToolResultEnvelope("exec_command", {"command": "custom"}, {}, text),
        16_000,
    )

    assert result.retained_chars < result.original_chars
    assert "BEGINNING" not in result.text
    assert "ERROR: failed" in result.text
    assert "LATEST" in result.text


def test_process_compaction_keeps_status_and_latest_log():
    process_result = {
        "processes": [{
            "process_id": "p-1",
            "status": "exited",
            "exit_code": 1,
            "log_tail": "old log line\n" * 1000 + "LATEST LOG LINE\n",
            "log_tail_truncated": True,
        }],
        "success": True,
    }

    result = compact_tool_result_envelope(
        ToolResultEnvelope(
            "inspect_processes",
            {},
            process_result,
            json.dumps(process_result, indent=2),
        ),
        16_000,
    )

    assert result.retained_chars < result.original_chars
    assert '"status": "exited"' in result.text
    assert '"exit_code": 1' in result.text
    assert "LATEST LOG LINE" in result.text


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


def test_prompt_tokens_anchor_beats_a_lagging_ratio():
    """The anchor, not the smoothed ratio, decides live usage.

    Character counting cannot see tool schemas or wire framing, so once the API
    has reported a real count the estimate must start from it and only estimate
    the delta.
    """
    estimator = TokenEstimator()

    estimator.observe(100_000, 25_000)  # ratio snaps to 4.0 on first observation
    assert estimator.prompt_tokens(100_000) == 25_000
    assert estimator.prompt_tokens(140_000) == 35_000
    assert estimator.prompt_tokens(60_000) == 15_000  # shrinks after compaction

    estimator.observe(100_000, 50_000)  # ratio only eases toward 2.0 via EMA
    assert estimator.chars_to_tokens(100_000) != 50_000
    assert estimator.prompt_tokens(100_000) == 50_000


def test_unanchored_estimator_falls_back_to_the_ratio():
    assert TokenEstimator(initial=4.0).prompt_tokens(40_000) == 10_000


def test_compaction_trigger_reserves_output_headroom():
    # Large window: the ratio governs, not the (relatively tiny) reserve.
    assert compaction_trigger_tokens(1_000_000, max_output_tokens=32_000) == 850_000
    # Typical window: reserve keeps the trigger below the flat ratio.
    assert compaction_trigger_tokens(200_000, max_output_tokens=32_000) == 160_000
    # Small window with a huge output cap: the floor prevents constant compaction.
    assert compaction_trigger_tokens(32_000, max_output_tokens=16_000) == 16_000


def test_unknown_context_length_never_forces_compaction():
    assert not needs_compaction([{"role": "user", "content": "x" * 10_000}], 0)


@pytest.mark.parametrize(
    "context_length,max_output_tokens",
    [
        (128_000, 32_000),   # typical window
        (1_000_000, 64_000),  # large window, ratio governs
        (258_000, 128_000),  # gpt-5: reserve drags the trigger to its floor
        (32_000, 16_000),    # small window, huge output cap
    ],
)
def test_eviction_always_gets_a_pass_before_compaction(context_length, max_output_tokens):
    """Layer 2 is cheap and reversible; it must never be stranded above layer 3."""
    evict = eviction_trigger_tokens(context_length, max_output_tokens)
    compact = compaction_trigger_tokens(context_length, max_output_tokens)
    assert evict < compact


@pytest.mark.parametrize(
    "context_length,max_output_tokens",
    [
        (128_000, 32_000),   # typical window
        (1_000_000, 64_000),  # large window, ratio governs
        (258_000, 128_000),  # gpt-5: reserve drags the trigger to its floor
        (32_000, 16_000),    # small window, huge output cap
    ],
)
def test_compaction_target_never_lands_on_its_own_trigger(context_length, max_output_tokens):
    """A pass that hits its target must not be re-triggered by the next result.

    gpt-5 used to put both on 129k of a 258k window, so every tool call after a
    compaction bought another one.
    """
    trigger = compaction_trigger_tokens(context_length, max_output_tokens)
    target = compaction_target_tokens(context_length, max_output_tokens)
    assert target < trigger
    # The gap has to be worth more than the yield a pass must produce at all,
    # otherwise clearing the trigger is within noise of a single tool result.
    assert trigger - target >= context_length * COMPACTION_MIN_YIELD_RATIO


def test_a_split_too_small_to_pay_for_itself_spends_no_round_trip():
    """Real pressure is not enough; the split still has to free more than it writes."""
    messages = [{"role": "user", "content": "start"}]
    for i in range(3):  # old and cheap — all that sits outside the keep window
        messages += _turn(f"s{i}", "read_file", {"file": f"s{i}.py"}, "x" * 1_000)
    for i in range(12):  # recent and bulky — protected from summarization
        messages += _turn(f"b{i}", "read_file", {"file": f"b{i}.py"}, "x" * 5_000)
    messages.append({"role": "assistant", "content": "done"})
    calls = []

    result, _ = compact_context(
        messages, lambda prompt: calls.append(prompt) or "## Goal\ng",
        context_length=CONTEXT_LENGTH, chars_per_token=CHARS_PER_TOKEN,
        used_tokens=95_000,  # far over the trigger: the pressure is real
    )

    # The summarizable half is ~3k chars; the summary written back in its place
    # is allowed up to 32k. Compacting here would *grow* the context.
    assert not calls
    assert result is messages


# ---------------------------------------------------------------------------
# Layer 2 eviction
# ---------------------------------------------------------------------------

CONTEXT_LENGTH = 100_000
CHARS_PER_TOKEN = 4.0
# window 400k chars · trigger 55k tokens · target 45k · min yield 32k chars
# · protected tail min(25k tokens, 40k) = 100k chars


def _turn(call_id, name, arguments, output):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


def _with_artifact(body, path):
    """Wrap *body* in the footer Layer 1 writes when it offloads to disk."""
    return (
        f"[Large read_file result compacted: 999,999 characters]\n{body}\n"
        f"[compacted: ~9→~1 tokens, -90%."
        f" Captured output: {path}. Use a narrower read.]"
    )


def _tail_turns():
    """Three exec_command turns, large enough to fill the protected tail."""
    messages = []
    for i in range(3):
        messages += _turn(f"t{i}", "exec_command", {"command": f"echo {i}"}, "y" * 40_000)
    return messages


def _prune(messages, used_tokens):
    return prune_context(messages, CONTEXT_LENGTH, CHARS_PER_TOKEN, used_tokens=used_tokens)


def test_eviction_leaves_context_alone_below_the_trigger():
    messages = [{"role": "user", "content": "start"}]
    for i in range(5):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 30_000)
    messages += _tail_turns()

    assert _prune(messages, used_tokens=50_000) is messages


def test_eviction_trims_oldest_results_and_spares_the_protected_tail():
    messages = [{"role": "user", "content": "start"}]
    for i in range(5):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 30_000)
    messages += _tail_turns()

    result = _prune(messages, used_tokens=67_500)

    assert result is not messages
    # Oldest four cover the shortfall; the fifth is left intact.
    for index in (2, 4, 6, 8):
        assert "[Trimmed: kept beginning" in get_text(result[index])
    assert get_text(result[10]) == "x" * 30_000
    # The newest turns are never touched, whatever the pressure.
    for index in (12, 14, 16):
        assert get_text(result[index]) == "y" * 40_000


def test_eviction_skips_a_pass_that_cannot_pay_for_the_cache_break():
    """Small results are left alone rather than nibbled every round."""
    messages = [{"role": "user", "content": "start"}]
    for i in range(5):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 5_000)
    messages += _tail_turns()

    assert _prune(messages, used_tokens=60_000) is messages


def test_superseded_read_is_cleared_and_keeps_its_recovery_pointer():
    path = ".kia/tool-results/s1/r1-c0-read_file.txt"
    messages = [{"role": "user", "content": "start"}]
    messages += _turn("c0", "read_file", {"file": "a.py"},
                      _with_artifact("x" * 40_000, path))
    messages += _turn("c1", "read_file", {"file": "b.py"}, "x" * 5_000)
    messages += _turn("c2", "read_file", {"file": "c.py"}, "x" * 5_000)
    # A later identical read makes the first one stale.
    messages += _turn("c3", "read_file", {"file": "a.py"}, "x" * 40_000)
    messages += _tail_turns()[:4]

    result = _prune(messages, used_tokens=60_000)

    cleared = get_text(result[2])
    assert cleared.startswith("[cleared: read_file(a.py)")
    assert "superseded by a later identical call" in cleared
    assert path in cleared
    # Results that are still current are not collateral damage.
    assert get_text(result[4]) == "x" * 5_000
    assert get_text(result[6]) == "x" * 5_000


def test_supersession_is_scoped_to_the_same_target():
    """A snapshot of one process must not stale-mark another's."""
    messages = [{"role": "user", "content": "start"}]
    messages += _turn("c0", "inspect_processes", {"process_id": "p-1"}, "x" * 40_000)
    messages += _turn("c1", "read_file", {"file": "b.py"}, "x" * 30_000)
    messages += _turn("c2", "read_file", {"file": "c.py"}, "x" * 30_000)
    messages += _turn("c3", "inspect_processes", {"process_id": "p-2"}, "x" * 40_000)
    messages += _tail_turns()[:4]

    result = _prune(messages, used_tokens=60_000)

    assert "[cleared:" not in get_text(result[2])


def test_write_invalidates_an_earlier_read_of_the_same_file():
    messages = [{"role": "user", "content": "start"}]
    messages += _turn("c0", "read_file", {"file": "a.py"}, "x" * 40_000)
    messages += _turn("c1", "read_file", {"file": "b.py"}, "x" * 5_000)
    messages += _turn("c2", "read_file", {"file": "c.py"}, "x" * 5_000)
    messages += _turn("c3", "edit_file", {"file": "a.py"}, "x" * 40_000)
    messages += _tail_turns()[:4]

    result = _prune(messages, used_tokens=60_000)

    assert "file modified after this read" in get_text(result[2])


def test_eviction_clears_only_what_is_recoverable_from_disk():
    """Trimming comes first; clearing escalates only when it falls short."""
    messages = [{"role": "user", "content": "start"}]
    for i in range(3):
        path = f".kia/tool-results/s1/r{i}-c{i}-read_file.txt"
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"},
                          _with_artifact("x" * 30_000, path))
    messages += _tail_turns()

    modest = _prune(messages, used_tokens=57_000)
    assert "[Trimmed: kept beginning" in get_text(modest[2])

    severe = _prune(messages, used_tokens=82_500)
    cleared = get_text(severe[2])
    assert cleared.startswith("[cleared: read_file(f0.py)")
    assert "evicted to free context" in cleared
    assert ".kia/tool-results/s1/r0-c0-read_file.txt" in cleared


def test_trimming_keeps_the_recovery_pointer_so_a_later_pass_can_clear():
    """A prefix policy drops the ingress footer; the pointer must be re-attached."""
    messages = [{"role": "user", "content": "start"}]
    for i in range(5):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"},
                          _with_artifact("x" * 30_000, f".kia/tool-results/s1/r{i}.txt"))
    messages += _tail_turns()

    trimmed = _prune(messages, used_tokens=67_500)
    assert "[Trimmed: kept beginning" in get_text(trimmed[2])
    assert ".kia/tool-results/s1/r0.txt" in get_text(trimmed[2])

    # Without the pointer the message would be stuck at its trimmed size forever.
    cleared = _prune(trimmed, used_tokens=95_000)
    assert get_text(cleared[2]).startswith("[cleared: read_file(f0.py)")
    assert ".kia/tool-results/s1/r0.txt" in get_text(cleared[2])


# ---------------------------------------------------------------------------
# Layer 3 compaction
# ---------------------------------------------------------------------------


def _capture(reply="## Goal\nship it"):
    seen = {}

    def summarize(prompt):
        seen["prompt"] = prompt
        return reply

    return seen, summarize


def test_original_request_is_carried_verbatim_across_compactions():
    request = "Refactor kiui/agent/context.py to fix the eviction thresholds"
    messages = [{"role": "user", "content": request}]
    for i in range(6):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 200)
    messages.append({"role": "assistant", "content": "done"})

    seen, summarize = _capture()
    first, state = compact_context(messages, summarize)
    assert f"## Original request\n{request}" in get_text(first[0])

    # Compacting the result again must not paraphrase or drop the request.
    second_input = first + [
        {"role": "user", "content": "keep going"},
        {"role": "assistant", "content": "ok"},
    ]
    seen, summarize = _capture()
    second, _ = compact_context(second_input, summarize, state=state)
    assert f"## Original request\n{request}" in get_text(second[0])


def test_recompaction_updates_the_previous_summary_instead_of_reducing_it():
    prior = (
        f"{SUMMARY_MARKER}\nhandoff\n\n## Original request\nbuild it\n\n"
        "## Goal\nship the feature"
    )
    messages = [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "worked on it"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "more work"},
    ]

    seen, summarize = _capture()
    compact_context(messages, summarize)

    prompt = seen["prompt"]
    assert "<previous-summary>" in prompt
    assert "ship the feature" in prompt
    assert "PRESERVE every still-relevant fact" in prompt
    # The prior summary is supplied once, as the thing being updated.
    assert prompt.count("ship the feature") == 1


def _filler(count):
    """Trailing messages so the 40% split reaches the turns under test."""
    return [{"role": "assistant", "content": f"step {i}"} for i in range(count)]


def test_summary_input_drops_the_middle_not_the_tail():
    messages = [{"role": "user", "content": "FIRST-REQUEST"}]
    for i in range(400):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, f"MID{i} " + "x" * 900)

    seen, summarize = _capture()
    compact_context(messages, summarize)

    prompt = seen["prompt"]
    # The oldest entries anchor what the session set out to do...
    assert "FIRST-REQUEST" in prompt
    assert "MID0 " in prompt
    # ...and the newest of the compacted range carry the current state.
    assert any(f"MID{i} " in prompt for i in range(145, 159))
    # The middle is what gets sacrificed.
    assert "MID80 " not in prompt
    assert "middle of the conversation omitted" in prompt


def test_summary_carries_file_and_skill_state_without_contents():
    messages = [{"role": "user", "content": "start"}]
    messages += _turn("c0", "read_file", {"file": "a.py"}, "CONTENTS OF A")
    messages += _turn("c1", "edit_file", {"file": "b.py"}, "ok")
    messages += _turn("c2", "load_skill", {"name": "monitor"}, "skill text")
    messages += _filler(12)

    seen, summarize = _capture()
    result, _ = compact_context(messages, summarize)

    content = get_text(result[0])
    assert "- Modified: b.py" in content
    assert "- Read: a.py" in content
    assert "CONTENTS OF A" not in content  # a list, never the contents
    assert "- Active: monitor" in content


def test_file_lists_merge_across_compactions_and_track_writes():
    messages = [{"role": "user", "content": "start"}]
    messages += _turn("c0", "read_file", {"file": "a.py"}, "x")
    messages += _turn("c1", "read_file", {"file": "b.py"}, "x")
    messages += _filler(10)

    seen, summarize = _capture()
    first, state = compact_context(messages, summarize)
    assert "- Read: a.py, b.py" in get_text(first[0])

    second_input = first[:1] + list(_turn("c2", "edit_file", {"file": "a.py"}, "ok"))
    second_input += _filler(10)
    seen, summarize = _capture()
    second, _ = compact_context(second_input, summarize, state=state)

    content = get_text(second[0])
    assert "- Modified: a.py" in content
    assert "- Read: b.py" in content  # a.py moved out of the read list


def test_compaction_never_splits_into_the_protected_recent_window():
    messages = [{"role": "user", "content": "start"}]
    for i in range(20):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 20_000)
    messages.append({"role": "assistant", "content": "done"})

    seen, summarize = _capture()
    result, _ = compact_context(
        messages, summarize,
        context_length=CONTEXT_LENGTH,
        chars_per_token=CHARS_PER_TOKEN,
        used_tokens=90_000,
    )

    # keep-recent is min(15% of 100k, 20k) = 15k tokens = 60k chars, so the
    # newest turns survive even under heavy pressure.
    kept = sum(msg_chars(m) for m in result[1:])
    assert kept >= 60_000
    assert get_text(result[-1]) == "done"


def test_manual_compaction_of_a_short_session_spends_nothing():
    """/compact skips the trigger, but must not rewrite one message for a round-trip."""
    messages = [{"role": "user", "content": "add a retry to the uploader"}]
    for i in range(12):
        messages += _turn(f"c{i}", "read_file", {"file": f"f{i}.py"}, "x" * 1_500)
    messages.append({"role": "assistant", "content": "done"})
    used = sum(msg_chars(m) for m in messages) // 4
    calls = []

    result, _ = compact_context(
        messages, lambda prompt: calls.append(prompt) or "## Goal\ng",
        context_length=CONTEXT_LENGTH, chars_per_token=CHARS_PER_TOKEN,
        used_tokens=used,
    )

    # The whole history sits inside the protected recent window.
    assert not calls
    assert result is messages


def test_compaction_is_a_no_op_when_the_split_walks_off_the_front():
    """An all-tool-call history must not be 'summarized' into a longer one."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c0",
                "function": {"name": "read_file", "arguments": '{"file": "a.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "x" * 500},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    calls = []

    result, _ = compact_context(messages, lambda prompt: calls.append(prompt) or "S")

    assert not calls  # no round-trip spent on an empty conversation
    assert result is messages


def test_prior_summary_is_not_clipped_when_compacting_again():
    """Repeated compaction must not lose the early session a summary stands for."""
    prior = f"{SUMMARY_MARKER}\n" + "PRIOR-" * 1000
    messages = [
        {"role": "user", "content": prior},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "done"},
    ]
    seen = {}

    def summarize(prompt):
        seen["prompt"] = prompt
        return "new summary"

    result, _ = compact_context(messages, summarize)

    assert seen["prompt"].count("PRIOR-") == 1000
    assert get_text(result[0]).startswith(SUMMARY_MARKER)
