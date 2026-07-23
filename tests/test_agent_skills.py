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


# ----- skill-provided tools ------------------------------------------------

_TOOLS_PY = '''\
def ping(executor, value="pong"):
    return {"echo": value, "success": True}

TOOLS = [
    {
        "permission": "safe",
        "run": ping,
        "schema": {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Echo a value back.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": [],
                },
            },
        },
    },
]
'''


def test_loading_skill_injects_its_tools(tmp_path):
    d = _write_skill(tmp_path, ".kia", "pinger", _valid("pinger"))
    (d / "tools.py").write_text(_TOOLS_PY, encoding="utf-8")
    skills = discover_skills(tmp_path)
    ex = ToolExecutor(work_dir=str(tmp_path), skills=skills)

    # The tool is neither advertised nor callable before the skill is loaded.
    assert "ping" not in {s["function"]["name"] for s in ex.skill_tool_schemas()}
    assert not ex.execute("ping", {})["success"]

    ex._load_skill("pinger")

    names = {s["function"]["name"] for s in ex.skill_tool_schemas()}
    assert "ping" in names
    assert ex.registry.permission("ping") == "safe"
    result = ex.execute("ping", {"value": "hi"})
    assert result["success"] and result["echo"] == "hi"

    # Unregistering removes both the schema and the dispatch entry.
    ex.unregister_skill_tools("pinger")
    assert "ping" not in {s["function"]["name"] for s in ex.skill_tool_schemas()}
    assert not ex.execute("ping", {})["success"]


def test_skill_tool_cannot_shadow_builtin(tmp_path):
    import pytest

    d = _write_skill(tmp_path, ".kia", "shadow", _valid("shadow"))
    (d / "tools.py").write_text(
        _TOOLS_PY.replace('"name": "ping"', '"name": "read_file"'),
        encoding="utf-8",
    )
    skills = discover_skills(tmp_path)
    ex = ToolExecutor(work_dir=str(tmp_path), skills=skills)
    with pytest.raises(ValueError):
        ex.register_skill_tools("shadow", __import__(
            "kiui.agent.skills", fromlist=["load_skill_tools"]
        ).load_skill_tools(d))


def test_two_skills_cannot_define_same_tool(tmp_path):
    import pytest

    a = _write_skill(tmp_path, ".kia", "alpha", _valid("alpha"))
    (a / "tools.py").write_text(_TOOLS_PY, encoding="utf-8")
    b = _write_skill(tmp_path, ".kia", "bravo", _valid("bravo"))
    (b / "tools.py").write_text(_TOOLS_PY, encoding="utf-8")  # also defines "ping"
    skills = discover_skills(tmp_path)
    ex = ToolExecutor(work_dir=str(tmp_path), skills=skills)

    assert ex._load_skill("alpha")["success"]
    # bravo's "ping" collides with alpha's; the load fails and leaves alpha intact.
    result = ex._load_skill("bravo")
    assert not result["success"]
    assert "bravo" not in ex._loaded_skills
    assert ex.registry.get("ping").source == "alpha"
    assert ex.execute("ping", {"value": "hi"})["echo"] == "hi"


# A tools.py whose second tool shadows a built-in: registration must be atomic,
# so the first (valid) tool must not leak into the surface.
_TWO_TOOLS_PY = _TOOLS_PY.replace(
    "TOOLS = [",
    '''\
def grab(executor):
    return {"success": True}

TOOLS = [
    {
        "permission": "safe",
        "run": grab,
        "schema": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Shadows a built-in.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    },''',
)


def test_load_skill_with_broken_tools_fails_atomically(tmp_path):
    d = _write_skill(tmp_path, ".kia", "broken", _valid("broken"))
    (d / "tools.py").write_text(_TWO_TOOLS_PY, encoding="utf-8")
    skills = discover_skills(tmp_path)
    ex = ToolExecutor(work_dir=str(tmp_path), skills=skills)

    result = ex._load_skill("broken")
    # Load fails, skill is not marked loaded, and no tools leaked in.
    assert not result["success"]
    assert "broken" not in ex._loaded_skills
    assert ex.skill_tool_schemas() == []
    assert "ping" not in {s["function"]["name"] for s in ex.skill_tool_schemas()}
    assert not ex.execute("ping", {})["success"]
