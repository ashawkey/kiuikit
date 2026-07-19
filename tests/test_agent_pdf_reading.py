"""Tests for the bundled MinerU PDF-reading skill."""

import importlib.util
import json
from pathlib import Path

import pytest

from kiui.agent.skills import BUNDLED_SKILLS_DIR, read_skill


SKILL_DIR = BUNDLED_SKILLS_DIR / "pdf-reading"


def _load_wrapper():
    path = SKILL_DIR / "scripts" / "parse_pdf.py"
    spec = importlib.util.spec_from_file_location("kia_pdf_parser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bundled_pdf_reading_skill_is_valid():
    skill = read_skill(SKILL_DIR, strict=True)
    assert skill["frontmatter"]["name"] == "pdf-reading"
    assert "PDF" in skill["description"]
    assert (SKILL_DIR / "scripts" / "parse_pdf.py").is_file()


def test_pdf_wrapper_builds_argument_list_without_shell_interpolation(tmp_path):
    wrapper = _load_wrapper()
    args = wrapper.parse_args([
        "paper; echo unsafe.pdf",
        "--output", str(tmp_path / "output with spaces"),
        "--backend", "pipeline",
        "--api-url", "http://127.0.0.1:8000",
        "--start", "2",
        "--end", "4",
    ])
    command = wrapper.build_command(
        args,
        tmp_path / "paper; echo unsafe.pdf",
        tmp_path / "output with spaces",
        "/venv/bin/mineru",
    )

    assert command[0] == "/venv/bin/mineru"
    assert command[command.index("-p") + 1].endswith("paper; echo unsafe.pdf")
    assert command[command.index("-o") + 1].endswith("output with spaces")
    assert command[command.index("-s") + 1] == "2"
    assert command[command.index("-e") + 1] == "4"
    assert command[command.index("--api-url") + 1] == "http://127.0.0.1:8000"


def test_pdf_wrapper_rejects_http_client_backends():
    wrapper = _load_wrapper()
    with pytest.raises(SystemExit):
        wrapper.parse_args(["paper.pdf", "--backend", "vlm-http-client"])


def test_pdf_wrapper_finds_project_mineru_environment(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    executable = tmp_path / ".kia" / "mineru-venv" / "bin" / "mineru"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")

    assert wrapper.find_executable("mineru") == str(executable.resolve())


def test_pdf_wrapper_locates_and_validates_pipeline_output(tmp_path):
    wrapper = _load_wrapper()
    parse_dir = tmp_path / "paper" / "auto"
    images_dir = parse_dir / "images"
    images_dir.mkdir(parents=True)
    (parse_dir / "paper.md").write_text("# Paper\n\n$x = 1$\n", encoding="utf-8")
    (parse_dir / "paper_content_list.json").write_text(json.dumps([
        {"type": "text", "text": "Paper", "page_idx": 0},
        {"type": "equation", "text": "$x = 1$", "page_idx": 1},
    ]), encoding="utf-8")
    (images_dir / "figure.jpg").write_bytes(b"image")

    markdown = wrapper.locate_markdown(tmp_path, "paper", "pipeline", "auto")
    manifest = wrapper.validate_output(markdown)

    assert manifest["page_count"] == 2
    assert manifest["content_blocks"] == 2
    assert manifest["block_types"] == {"equation": 1, "text": 1}
    assert manifest["extracted_images"] == 1
