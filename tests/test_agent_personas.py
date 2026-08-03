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
    skills: str = "skills:\n  bundled: []\n  local: false\n",
    body: str = "You are a test persona.",
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "PERSONA.md").write_text(
        f"---\nname: {name}\n"
        f"description: {description}\ntools: {tools}\n{skills}---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_bundled_personas_are_declarative():
    personas = discover_personas()

    assert set(personas) >= {"coder", "chatter", "reviewer"}
    assert personas["coder"].path.endswith("PERSONA.md")
    assert personas["coder"].tools is None
    assert personas["coder"].bundled_skills is None
    assert personas["coder"].local_skills is True
    assert personas["chatter"].tools == frozenset({"web_search", "web_fetch"})
    assert personas["chatter"].bundled_skills == frozenset()
    assert personas["chatter"].local_skills is False
    assert personas["reviewer"].bundled_skills == frozenset({"pdf-reading"})
    assert personas["reviewer"].local_skills is False


def test_read_persona_validates_skill_policy(tmp_path):
    path = _write_persona(
        tmp_path,
        "bad",
        skills="skills:\n  bundled: [not-bundled]\n  local: false\n",
    )
    with pytest.raises(ValueError, match="unknown bundled skill"):
        read_persona(path)

    (path / "PERSONA.md").write_text(
        "---\nname: bad\ndescription: Bad\ntools: all\n"
        "skills:\n  bundled: []\n  local: nope\n---\nPrompt.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="skills.local.*boolean"):
        read_persona(path)


def test_render_all_bundled_skills(tmp_path):
    path = _write_persona(
        tmp_path,
        "custom",
        tools="[load_skill]",
        skills="skills:\n  bundled: all\n  local: false\n",
        body="{{kia:skills}}",
    )
    skills = {
        "pdf-reading": {"description": "Read PDFs", "source": "bundled"},
        "monitor": {"description": "Monitor jobs", "source": "bundled"},
        "project-helper": {"description": "Project helper", "source": "project"},
    }

    persona = read_persona(path)
    prompt = persona.build(PersonaContext(skills=skills))

    assert persona.bundled_skills is None
    assert "**pdf-reading**" in prompt
    assert "**monitor**" in prompt
    assert "project-helper" not in prompt


def test_render_filters_bundled_and_local_skills(tmp_path):
    path = _write_persona(
        tmp_path,
        "custom",
        tools="[load_skill]",
        skills="skills:\n  bundled: [pdf-reading]\n  local: false\n",
        body="{{kia:skills}}",
    )
    skills = {
        "pdf-reading": {"description": "Read PDFs", "source": "bundled"},
        "monitor": {"description": "Monitor jobs", "source": "bundled"},
        "project-helper": {"description": "Project helper", "source": "project"},
        "personal-helper": {"description": "Personal helper", "source": "personal"},
    }

    prompt = read_persona(path).build(PersonaContext(skills=skills))

    assert "**pdf-reading**" in prompt
    assert "monitor" not in prompt
    assert "project-helper" not in prompt
    assert "personal-helper" not in prompt


def test_read_persona_requires_skill_policy(tmp_path):
    path = _write_persona(tmp_path, "custom")
    persona_md = path / "PERSONA.md"
    persona_md.write_text(
        persona_md.read_text(encoding="utf-8").replace(
            "skills:\n  bundled: []\n  local: false\n", ""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'skills' is required"):
        read_persona(path)


def test_read_persona_validates_markers(tmp_path):
    path = _write_persona(tmp_path, "bad", body="Before {{kia:current-context}} after")

    with pytest.raises(ValueError, match="must occupy its own line"):
        read_persona(path)

    (path / "PERSONA.md").write_text(
        "---\nname: bad\ndescription: Bad\ntools: []\n"
        "skills:\n  bundled: []\n  local: false\n---\n"
        "{{kia:unknown}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown marker"):
        read_persona(path)


def test_exec_mode_forbids_nested_subagents(tmp_path):
    path = _write_persona(tmp_path / "personas", "custom")
    persona = read_persona(path)

    interactive_prompt = persona.build(PersonaContext(exec_mode=False))
    exec_prompt = persona.build(PersonaContext(exec_mode=True))

    assert "subagent" not in interactive_prompt
    assert "Do not invoke the `subagent` skill" in exec_prompt
    assert "Delegation depth is limited to one" in exec_prompt


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


def test_local_project_instructions_can_import_root(tmp_path):
    work = tmp_path / "work"
    (work / ".kia").mkdir(parents=True)
    (work / "AGENTS.md").write_text("Root instructions.", encoding="utf-8")
    (work / ".kia" / "AGENTS.md").write_text(
        "@AGENTS.md\n\nLocal instructions.", encoding="utf-8"
    )
    path = _write_persona(
        tmp_path / "personas",
        "custom",
        body="{{kia:project-instructions}}",
    )

    prompt = read_persona(path).build(PersonaContext(work_dir=str(work)))

    assert prompt == "## Project Instructions\nRoot instructions.\n\nLocal instructions."


def test_local_project_instructions_replace_root(tmp_path):
    work = tmp_path / "work"
    (work / ".kia").mkdir(parents=True)
    (work / "AGENTS.md").write_text("Root instructions.", encoding="utf-8")
    (work / ".kia" / "AGENTS.md").write_text("Local only.", encoding="utf-8")
    path = _write_persona(
        tmp_path / "personas",
        "custom",
        body="{{kia:project-instructions}}",
    )

    prompt = read_persona(path).build(PersonaContext(work_dir=str(work)))

    assert prompt == "## Project Instructions\nLocal only."


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
