"""Tests for skill discovery issue reporting (shadowing + malformed skills)
and per-session skill load-count tracking (kiui.agent.skills / tools)."""

from pathlib import Path

from kiui.agent import skills as skills_module
from kiui.agent.skills import build_skills_prompt_section, discover_skills
from kiui.agent.tools import ToolExecutor


def _write_skill(root: Path, agent_dir: str, name: str, body: str):
    d = root / agent_dir / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d


def _valid(name: str, desc: str = "does a thing, use when relevant") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\nInstructions for {name}.\n"


# ----- discovery issue reporting -------------------------------------------

def test_bundled_skills_load_from_package_and_shadow_stale_copies(tmp_path, monkeypatch):
    bundled = tmp_path / "package-skills"
    project = tmp_path / "project"
    (bundled / "alpha").mkdir(parents=True)
    (bundled / "alpha" / "SKILL.md").write_text(
        _valid("alpha", "current bundled"), encoding="utf-8"
    )
    _write_skill(project, ".kia", "alpha", _valid("alpha", "stale project copy"))
    monkeypatch.setattr(skills_module, "BUNDLED_SKILLS_DIR", bundled)

    issues = {}
    skills = discover_skills(project, issues=issues)

    assert skills["alpha"]["description"] == "current bundled"
    assert Path(skills["alpha"]["dir"]) == bundled / "alpha"
    assert issues["shadowed"][0]["path"] == str(project / ".kia" / "skills" / "alpha" / "SKILL.md")


def test_bundled_skill_library_documents_kib_commands(tmp_path):
    skills = discover_skills(tmp_path)

    skill = skills["skill-library"]
    assert "list available or installed library skills" in skill["description"]
    assert "`kib list`" in skill["body"]
    assert "`kib list --local`" in skill["body"]
    assert "`kib install <name>`" in skill["body"]
    assert "`kib upload <name>`" in skill["body"]
    assert "`kib remove <name>`" in skill["body"]
    assert "`/skills reload`" in skill["body"]


def test_discover_bundled_skills_does_not_create_project_copy(tmp_path, monkeypatch):
    bundled = tmp_path / "package-skills"
    project = tmp_path / "project"
    (bundled / "alpha").mkdir(parents=True)
    (bundled / "alpha" / "SKILL.md").write_text(_valid("alpha"), encoding="utf-8")
    monkeypatch.setattr(skills_module, "BUNDLED_SKILLS_DIR", bundled)

    assert "alpha" in discover_skills(project)
    assert not (project / ".kia").exists()


def test_discover_ignores_external_agent_dirs(tmp_path):
    _write_skill(tmp_path, ".kia", "native", _valid("native", "kia skill"))
    _write_skill(tmp_path, ".claude", "foreign", _valid("foreign", "claude skill"))

    issues = {}
    skills = discover_skills(tmp_path, issues=issues)

    assert "native" in skills
    assert "foreign" not in skills
    assert not any(issue["name"] == "foreign" for issue in issues["shadowed"])
    assert not any(issue["name"] == "foreign" for issue in issues["errors"])


def test_project_skill_shadows_personal_skill(tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = tmp_path / "home"
    # Path.home() follows HOME on POSIX, USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _write_skill(project, ".kia", "dup", _valid("dup", "project copy"))
    _write_skill(home, ".kia", "dup", _valid("dup", "personal copy"))

    issues = {}
    skills = discover_skills(project, issues=issues)

    assert skills["dup"]["description"] == "project copy"
    assert len(issues["shadowed"]) == 1
    assert issues["shadowed"][0]["name"] == "dup"


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


def test_discover_tolerates_spec_deviations(tmp_path):
    _write_skill(tmp_path, ".kia", "wrong-dir", _valid("Display Name"))
    _write_skill(
        tmp_path,
        ".kia",
        "long-description",
        _valid("long-description", "x" * 1025),
    )

    issues = {}
    skills = discover_skills(tmp_path, issues=issues)

    assert set(skills) >= {"wrong-dir", "long-description"}
    assert issues["errors"] == []
    assert skills["wrong-dir"]["warnings"]
    assert skills["long-description"]["warnings"]


def test_discover_rejects_missing_description(tmp_path):
    _write_skill(tmp_path, ".kia", "broken", "---\nname: broken\n---\nbody\n")
    issues = {}
    skills = discover_skills(tmp_path, issues=issues)
    assert "broken" not in skills
    assert len(issues["errors"]) == 1


def test_skills_prompt_requires_loading_before_work_and_guides_creation():
    section = build_skills_prompt_section({
        "skill-creator": {"description": "Create skills when asked.", "active": True},
    })
    assert "load_skill** before doing" in section
    assert "asks to create or" in section


def test_prompt_advertises_discovered_kia_skills(tmp_path):
    _write_skill(tmp_path, ".kia", "native", _valid("native", "kia skill"))

    skills = discover_skills(tmp_path)
    section = build_skills_prompt_section(skills)

    assert "native" in skills
    assert skills["native"]["active"] is True
    assert "**native**" in section


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


def test_load_skill_header_only_lists_present_resources(tmp_path):
    skill_dir = _write_skill(tmp_path, ".kia", "alpha", _valid("alpha"))
    (skill_dir / "references").mkdir()
    (skill_dir / "assets").mkdir()

    ex = ToolExecutor(work_dir=str(tmp_path), skills=discover_skills(tmp_path))
    content = ex._load_skill("alpha")["content"]

    assert f"Its directory is {skill_dir}" in content
    assert "references/…" in content
    assert "assets/…" in content
    assert "scripts/…" not in content


def test_load_skill_header_omits_directory_without_resources(tmp_path):
    _write_skill(tmp_path, ".kia", "alpha", _valid("alpha"))
    ex = ToolExecutor(work_dir=str(tmp_path), skills=discover_skills(tmp_path))

    content = ex._load_skill("alpha")["content"]

    assert content.startswith("[Skill 'alpha' loaded.]\n\n")
    assert "Its directory" not in content


def test_load_missing_skill_not_counted(tmp_path):
    skills = discover_skills(tmp_path)  # none defined here (project scope empty)
    ex = ToolExecutor(work_dir=str(tmp_path), skills={})
    r = ex._load_skill("nope")
    assert not r["success"]
    assert ex._skill_loads == {}
