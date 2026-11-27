"""The Dependabot configuration (.github/dependabot.yml) matches the project."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))


def _name(requirement: str) -> str:
    """``"onnxruntime-gpu[cuda,cudnn]>=1.21"`` -> ``"onnxruntime-gpu"`` (normalised)."""
    name = re.match(r"[A-Za-z0-9._-]+", requirement).group()
    return re.sub(r"[-_.]+", "-", name).lower()


def _dependencies() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = list(project["project"]["dependencies"])
    for extra in project["project"].get("optional-dependencies", {}).values():
        requirements += extra
    for group in project.get("dependency-groups", {}).values():
        requirements += [r for r in group if isinstance(r, str)]
    return {_name(r) for r in requirements}


def _update(ecosystem: str) -> dict:
    [update] = [u for u in CONFIG["updates"] if u["package-ecosystem"] == ecosystem]
    return update


def test_every_kind_of_dependency_is_watched():
    assert CONFIG["version"] == 2
    assert {u["package-ecosystem"] for u in CONFIG["updates"]} == {"uv", "github-actions"}
    assert _update("uv")["directory"] == "/"
    assert (ROOT / "pyproject.toml").is_file() and (ROOT / "uv.lock").is_file()
    assert list((ROOT / ".github" / "workflows").glob("*.yml"))
    for update in CONFIG["updates"]:
        assert update["schedule"]["interval"] == "weekly"
        assert update["cooldown"]["default-days"] >= 3  # releases settle before they come


def test_python_groups_name_real_dependencies_and_leave_majors_alone():
    known = _dependencies()
    groups = _update("uv")["groups"]
    named = [p for g in groups.values() for p in g["patterns"] if p != "*"]
    assert sorted(set(named)) == sorted(named), "a package is in two groups"
    assert set(named) <= known, f"not in pyproject.toml: {sorted(set(named) - known)}"
    for name, group in groups.items():
        # a major update may need code changes: it gets a pull request of its own
        assert group["update-types"] == ["minor", "patch"], name
    assert list(groups.values())[-1]["patterns"] == ["*"]  # the catch-all comes last
