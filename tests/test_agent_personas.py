"""Tests for declarative persona parsing, discovery, and rendering."""

from pathlib import Path

import pytest

from kiui.agent import personas as personas_module
from kiui.agent.personas import PersonaContext, discover_personas, read_persona


def _write_persona(
    root: Path,
    name: str,
    *,
    description: str = "Test persona",
    tools: str = "all",
    body: str = "You are a test persona.",
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "PERSONA.md").write_text(
        f"---\nname: {name}\n"
        f"description: {description}\ntools: {tools}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_bundled_personas_are_declarative():
    personas = discover_personas()

    assert set(personas) >= {"coder", "chatter", "reviewer"}
    assert personas["coder"].path.endswith("PERSONA.md")
    assert personas["coder"].tools is None


def test_read_persona_validates_markers(tmp_path):
    path = _write_persona(tmp_path, "bad", body="Before {{kia:current-context}} after")

    with pytest.raises(ValueError, match="must occupy its own line"):
        read_persona(path)

    (path / "PERSONA.md").write_text(
        "---\nname: bad\ndescription: Bad\ntools: []\n---\n"
        "{{kia:unknown}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown marker"):
        read_persona(path)


def test_render_expands_markers_once(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "AGENTS.md").write_text("Keep this literal: {{kia:skills}}", encoding="utf-8")
    path = _write_persona(
        tmp_path / "personas",
        "custom",
        tools="[]",
        body="{{kia:skills}}\n{{kia:project-instructions}}\n{{kia:current-context}}",
    )

    prompt = read_persona(path).build(PersonaContext(work_dir=str(work), skills={"x": {}}))

    assert "Keep this literal: {{kia:skills}}" in prompt
    assert "## Current Context" in prompt
    assert "## Skills" not in prompt


def test_bundled_name_cannot_be_shadowed(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    project = tmp_path / "project"
    _write_persona(bundled, "coder", description="Bundled")
    _write_persona(project / ".kia" / "personas", "coder", description="Project")
    monkeypatch.setattr(personas_module, "BUNDLED_PERSONAS_DIR", bundled)

    issues = {}
    personas = discover_personas(project, issues=issues)

    assert personas["coder"].description == "Bundled"
    assert issues["shadowed"][0]["name"] == "coder"


def test_project_persona_shadows_personal(tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(personas_module, "BUNDLED_PERSONAS_DIR", tmp_path / "none")
    _write_persona(project / ".kia" / "personas", "custom", description="Project")
    _write_persona(home / ".kia" / "personas", "custom", description="Personal")

    issues = {}
    personas = discover_personas(project, issues=issues)

    assert personas["custom"].description == "Project"
    assert len(issues["shadowed"]) == 1
