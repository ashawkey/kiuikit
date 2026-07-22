"""Tests for per-tool output retention during context compaction."""

import json

import pytest

from kiui.agent.context import ToolResultEnvelope, compact_tool_result_envelope


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

    assert result.compacted
    assert "BEGINNING" in result.text
    assert "END" not in result.text


def test_exec_compaction_keeps_latest_output_and_diagnostics():
    text = "BEGINNING\n" + "middle\n" * 3000 + "ERROR: failed\nLATEST\n"

    result = compact_tool_result_envelope(
        ToolResultEnvelope("exec_command", {"command": "custom"}, {}, text),
        16_000,
    )

    assert result.compacted
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

    assert result.compacted
    assert '"status": "exited"' in result.text
    assert '"exit_code": 1' in result.text
    assert "LATEST LOG LINE" in result.text
