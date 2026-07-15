"""Tests for skill discovery issue reporting (shadowing + malformed skills)
and per-session skill load-count tracking (kiui.agent.skills / tools)."""

from pathlib import Path

from kiui.agent.skills import discover_skills
from kiui.agent.tools import ToolExecutor


def _write_skill(root: Path, agent_dir: str, name: str, body: str):
    d = root / agent_dir / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d


def _valid(name: str, desc: str = "does a thing, use when relevant") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\nInstructions for {name}.\n"


# ----- discovery issue reporting -------------------------------------------

def test_discover_reports_shadowing(tmp_path):
    _write_skill(tmp_path, ".kia", "dup", _valid("dup", "kia copy"))
    _write_skill(tmp_path, ".claude", "dup", _valid("dup", "claude copy"))

    issues = {}
    skills = discover_skills(tmp_path, issues=issues)

    assert "dup" in skills
    assert skills["dup"]["description"] == "kia copy"  # .kia wins over .claude
    assert len(issues["shadowed"]) == 1
    sh = issues["shadowed"][0]
    assert sh["name"] == "dup"
    assert ".claude" in sh["path"]
    assert ".kia" in sh["shadowed_by"]


def test_discover_reports_malformed(tmp_path):
    _write_skill(tmp_path, ".kia", "broken", "no frontmatter here\n")
    _write_skill(tmp_path, ".kia", "ok", _valid("ok"))

    issues = {}
    skills = discover_skills(tmp_path, issues=issues)

    assert "ok" in skills
    assert "broken" not in skills
    assert len(issues["errors"]) == 1
    assert issues["errors"][0]["name"] == "broken"


def test_discover_no_issues_when_clean(tmp_path):
    _write_skill(tmp_path, ".kia", "solo", _valid("solo"))
    issues = {}
    skills = discover_skills(tmp_path, issues=issues)
    assert "solo" in skills
    assert issues["shadowed"] == []
    assert issues["errors"] == []


def test_discover_issues_optional(tmp_path):
    _write_skill(tmp_path, ".kia", "solo", _valid("solo"))
    # Must not raise when issues arg is omitted.
    skills = discover_skills(tmp_path)
    assert "solo" in skills


# ----- load-count tracking -------------------------------------------------

def test_load_skill_counts(tmp_path):
    _write_skill(tmp_path, ".kia", "alpha", _valid("alpha"))
    skills = discover_skills(tmp_path)
    ex = ToolExecutor(work_dir=str(tmp_path), skills=skills)

    r1 = ex._load_skill("alpha")
    assert r1["success"] and "content" in r1
    assert ex._skill_loads["alpha"] == 1
    assert "alpha" in ex._loaded_skills

    # Redundant re-load still increments the usage counter.
    r2 = ex._load_skill("alpha")
    assert r2["success"] and "content" not in r2
    assert ex._skill_loads["alpha"] == 2


def test_load_missing_skill_not_counted(tmp_path):
    skills = discover_skills(tmp_path)  # none defined here (project scope empty)
    ex = ToolExecutor(work_dir=str(tmp_path), skills={})
    r = ex._load_skill("nope")
    assert not r["success"]
    assert ex._skill_loads == {}
