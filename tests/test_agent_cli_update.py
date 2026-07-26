"""Tests for ``kia --update``."""

from types import SimpleNamespace

from kiui.agent import cli


class FakeDistribution:
    def __init__(self, direct_url: str | None):
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self.direct_url


def test_update_editable_install_runs_git_pull(monkeypatch, tmp_path):
    direct_url = (
        '{"url": "' + tmp_path.as_uri() + '", '
        '"dir_info": {"editable": true}}'
    )
    commands = []
    monkeypatch.setattr(cli, "distribution", lambda name: FakeDistribution(direct_url))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_update() == 0
    assert commands == [["git", "-C", str(tmp_path), "pull"]]


def test_update_regular_install_reinstalls_from_source(monkeypatch):
    commands = []
    monkeypatch.setattr(cli, "distribution", lambda name: FakeDistribution(None))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_update() == 0
    assert commands == [
        [cli.sys.executable, "-m", "pip", "uninstall", "-y", "kiui"],
        [
            cli.sys.executable,
            "-m",
            "pip",
            "install",
            "kiui[kia] @ git+https://github.com/ashawkey/kiuikit.git",
        ],
    ]


def test_update_stops_and_reports_command_failure(monkeypatch, tmp_path):
    direct_url = (
        '{"url": "' + tmp_path.as_uri() + '", '
        '"dir_info": {"editable": true}}'
    )
    monkeypatch.setattr(cli, "distribution", lambda name: FakeDistribution(direct_url))
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command: SimpleNamespace(returncode=1)
    )

    assert cli.cmd_update() == 1
