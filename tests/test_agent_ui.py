import io

from rich.console import Console

from kiui.agent.ui import ResponseStream, ThinkingIndicator


def test_thinking_indicator_can_render_a_countdown():
    output = io.StringIO()
    console = Console(file=output, width=80, no_color=True)
    indicator = ThinkingIndicator(console, countdown=61, label="Waiting")

    assert indicator._label_plain(0) == "Waiting... (1m 01s)"
    assert indicator._label_plain(1.1) == "Waiting... (1m 00s)"
    assert indicator._label_plain(99) == "Waiting... (0s)"
    status = "".join(text for _, text in indicator._prompt_status("⠋", 1.1))
    assert "Waiting... (1m 00s)" in status


def make_stream():
    output = io.StringIO()
    console = Console(file=output, width=80, no_color=True)
    return output, ResponseStream(console, None)


def test_response_stream_writes_completed_lines_before_close():
    output, stream = make_stream()

    stream.on_content("Hello **world**\nPending")

    assert "Hello world" in output.getvalue()
    assert "Pending" not in output.getvalue()
    stream.close()
    assert "Pending" in output.getvalue()
    assert output.getvalue().count("Hello world") == 1


def test_response_stream_renders_block_markdown_across_chunks():
    output, stream = make_stream()
    chunks = [
        "# Ti",
        "tle\n- first\n1. sec",
        "ond\n| Name | Value |\n| --- | --- |\n| a | 1 |\n",
        "```python\ndef hi():\n    return 1\n```",
    ]

    for chunk in chunks:
        stream.on_content(chunk)
    stream.close()

    rendered = output.getvalue()
    assert "Title" in rendered
    assert "• first" in rendered
    assert "second" in rendered
    assert "Name" in rendered and "Value" in rendered
    assert "def hi():" in rendered
    assert "return 1" in rendered
    assert "```" not in rendered


def test_response_stream_keeps_streamed_list_items_compact():
    output, stream = make_stream()

    stream.on_content("- first\n- second\n- third\n")
    stream.close()

    lines = output.getvalue().splitlines()
    assert [line.strip() for line in lines] == ["•  • first", "• second", "• third"]


def test_response_stream_preserves_literal_asterisks_and_inline_code():
    output, stream = make_stream()

    for chunk in ["2 *", " 3 and `a*", "b*` and *italic", "*"]:
        stream.on_content(chunk)
    stream.close()

    assert "2 * 3 and a*b* and italic" in output.getvalue()


def test_response_stream_renders_table_without_leading_pipes():
    output, stream = make_stream()

    stream.on_content("A | B\n---|---\n1 | 2\n\n")
    stream.close()

    rendered = output.getvalue()
    assert "A" in rendered and "B" in rendered
    assert "1" in rendered and "2" in rendered
    # The table must be rendered by rich (pipes replaced by a rule line), but
    # the exact box character varies across rich versions (─ vs ━).
    assert "|" not in rendered
    assert any(ch in rendered for ch in ("─", "━"))


def test_response_stream_keeps_text_after_unterminated_table():
    output, stream = make_stream()

    stream.on_content("| A | B |\n|---|---|\n| 1 | 2 |\nafter")
    stream.close()

    rendered = output.getvalue()
    assert "1" in rendered and "2" in rendered
    assert "after" in rendered


def test_response_stream_commits_complete_thinking_lines():
    output = io.StringIO()
    console = Console(file=output, width=80, no_color=True)
    stream = ResponseStream(console, None, show_thinking=True)

    stream.on_thinking("first partial")
    assert output.getvalue() == ""

    stream.on_thinking(" completed\nsecond partial")
    assert output.getvalue() == "first partial completed\n"

    stream.on_content("answer")
    assert output.getvalue() == "first partial completed\nsecond partial\n"

    stream.close()
    assert "answer" in output.getvalue()


def test_response_stream_discards_pending_thinking_on_abort():
    output = io.StringIO()
    console = Console(file=output, width=80, no_color=True)
    stream = ResponseStream(console, None, show_thinking=True)

    stream.on_thinking("complete\npending")
    stream.close(render_terminal=False)

    assert output.getvalue() == "complete\n"


def test_agent_console_stream_output_writes_raw_block_and_emits_event():
    from kiui.agent.ui import AgentConsole
    from kiui.agent.utils.io import EventHub

    output = io.StringIO()
    console = AgentConsole()
    console._console = Console(file=output, width=80, force_terminal=True, color_system="truecolor")
    events = EventHub()
    console.events = events

    # Command output is data: markup and ANSI escapes pass through untouched.
    console.stream_output("a\n[/bad] [dim] tag\n\x1b[31mred\x1b[0m")

    rendered = output.getvalue()
    assert "[/bad] [dim] tag" in rendered
    assert "\x1b[31mred\x1b[0m" in rendered
    assert rendered.startswith("\x1b[2m")  # dim-wrapped on ANSI terminals

    evs = events.after(0)
    assert evs and evs[-1].type == "output"
    assert evs[-1].data["text"] == "a\n[/bad] [dim] tag\n\x1b[31mred\x1b[0m"


def test_agent_console_stream_output_is_plain_when_not_ansi():
    from kiui.agent.ui import AgentConsole

    output = io.StringIO()
    console = AgentConsole()
    console._console = Console(file=output, width=80, no_color=True)

    console.stream_output("plain line")
    console.stream_output("")

    assert output.getvalue() == "plain line\n"
