"""Desktop packaging: the release version script, the workflow plan, the installers' scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


version_script = _load("version")
plan_script = _load("plan")


# --------------------------------------------------------------------------- version.py
@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v1.2.3", "1.2.3"),
        ("1.2.3", "1.2.3"),
        ("v1.2", "1.2"),
        ("1.0", "1.0"),
        ("v1.2.3rc1", "1.2.3rc1"),
        ("v1.2.3-rc.1", "1.2.3rc1"),
        ("v1.2.3-RC1", "1.2.3rc1"),
        ("v1.2.3-beta.2", "1.2.3b2"),
        ("v1.2.3b2", "1.2.3b2"),
        ("v1.2.3-alpha.1", "1.2.3a1"),
        ("v2.0-alpha1", "2.0a1"),
        ("v01.02.03", "1.2.3"),
    ],
)
def test_tags_become_pep440_versions(tag, version):
    assert version_script.from_tag(tag) == version


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "v",
        "latest",
        "v1",
        "v1.2.3.4",
        "v1.2.3-rc",
        "v1.2.3-dev1",
        "v1.2.3.post1",
        "v1.2.3+local",
        "release-1.2.3",
        "v1.2.3 ",
        "v1.2.3-rc.1.2",
    ],
)
def test_other_tags_are_rejected(tag):
    with pytest.raises(ValueError, match="not a release tag"):
        version_script.from_tag(tag)


PYPROJECT = """\
[project]
name = "vision-tagger"
version = "0.1.0"  # set from the tag on release builds
description = "x"

[project.optional-dependencies]
version = ["not-this"]

[tool.other]
version = "0.0.9"
"""


def test_the_version_is_written_into_pyproject(tmp_path, capsys):
    path = tmp_path / "pyproject.toml"
    path.write_bytes(PYPROJECT.replace("\n", "\r\n").encode())  # line endings are kept
    assert version_script.current(path) == "0.1.0"
    assert version_script.main(["v1.4.0-rc.2", "--pyproject", str(path)]) == 0
    assert capsys.readouterr().out == "1.4.0rc2\n"
    expected = PYPROJECT.replace('"0.1.0"', '"1.4.0rc2"').replace("\n", "\r\n")
    assert path.read_bytes().decode() == expected  # nothing else changed
    assert version_script.main(["--current", "--pyproject", str(path)]) == 0
    assert capsys.readouterr().out == "1.4.0rc2\n"


def test_dry_run_and_errors_leave_pyproject_alone(tmp_path, capsys):
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT)
    assert version_script.main(["v2.0", "--dry-run", "--pyproject", str(path)]) == 0
    assert capsys.readouterr().out == "2.0\n"
    assert version_script.main(["nightly", "--pyproject", str(path)]) == 2
    assert "not a release tag" in capsys.readouterr().err
    assert path.read_text() == PYPROJECT
    path.write_text('[project]\nname = "x"\n[tool.x]\nversion = "1"\n')
    assert version_script.main(["v2.0", "--pyproject", str(path)]) == 2
    assert "no `version" in capsys.readouterr().err


def test_the_repository_pyproject_has_a_version_line():
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", version_script.current())


# --------------------------------------------------------------------------- plan.py
def names(builds):
    return [b["name"] for b in builds]


def test_a_release_builds_everything():
    builds = plan_script.plan("release")
    assert names(builds) == ["windows-cpu", "windows-gpu", "linux-cpu", "linux-gpu", "macos-cpu"]
    runners = {b["name"]: b["runner"] for b in builds}
    assert runners["linux-gpu"] == "ubuntu-22.04" and runners["windows-cpu"] == "windows-latest"
    assert builds[-1]["cli"] == "dist/Mira.app/Contents/MacOS/vision-tagger-cli"
    assert all(b["cli"].endswith(("vision-tagger-cli", "vision-tagger-cli.exe")) for b in builds)


def test_pushes_and_pull_requests_build_cpu_only():
    for event in ("push", "pull_request"):
        assert names(plan_script.plan(event)) == ["windows-cpu", "linux-cpu", "macos-cpu"]


@pytest.mark.parametrize(
    ("platform", "flavor", "expected"),
    [
        ("all", "both", ["windows-cpu", "windows-gpu", "linux-cpu", "linux-gpu", "macos-cpu"]),
        ("windows", "both", ["windows-cpu", "windows-gpu"]),
        ("linux", "gpu", ["linux-gpu"]),
        ("all", "cpu", ["windows-cpu", "linux-cpu", "macos-cpu"]),
        ("macos", "both", ["macos-cpu"]),
    ],
)
def test_a_manual_run_builds_what_was_chosen(platform, flavor, expected):
    assert names(plan_script.plan("workflow_dispatch", platform, flavor)) == expected


def test_plan_output_and_errors(capsys):
    assert plan_script.main(["--event", "workflow_dispatch", "--platform", "linux"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("matrix=") and out.count("\n") == 1
    matrix = json.loads(out.removeprefix("matrix="))
    assert [b["name"] for b in matrix["include"]] == ["linux-cpu", "linux-gpu"]
    # empty inputs (events other than a manual run) mean everything
    assert plan_script.main(["--event", "release", "--platform", "", "--flavor", ""]) == 0
    capsys.readouterr()
    assert (
        plan_script.main(["--event", "workflow_dispatch", "--platform", "macos", "--flavor", "gpu"])
        == 2
    )
    assert "no build" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown platform"):
        plan_script.plan("workflow_dispatch", "amiga")


# --------------------------------------------------------------------------- workflow, installers
def test_desktop_workflow():
    workflow = yaml.safe_load((ROOT / ".github/workflows/desktop.yml").read_text(encoding="utf-8"))
    triggers = workflow[True]  # YAML 1.1 reads the key `on` as True
    assert triggers["release"] == {"types": ["published"]}
    assert "tags" not in triggers["push"]  # releases are made in the UI, not by tag pushes
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["platform"]["options"] == list(plan_script.PLATFORMS)
    assert inputs["flavor"]["options"] == list(plan_script.FLAVORS)
    jobs = workflow["jobs"]
    assert "plan.py" in jobs["plan"]["steps"][-1]["run"]
    steps = [s.get("run", "") for s in jobs["build"]["steps"]]
    lock_check = next(i for i, s in enumerate(steps) if "uv lock --check" in s)
    set_version = next(i for i, s in enumerate(steps) if "version.py" in s)
    sync = next(i for i, s in enumerate(steps) if "uv sync --frozen" in s)
    assert lock_check < set_version < sync  # the version is set after the lock check
    assert "--no-editable" in steps[sync]
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    assert "--clobber" in jobs["publish"]["steps"][-1]["run"]
    # compiled wheels (insightface) must come from the same runner image: the uv cache key
    # doesn't name the OS version, and ci.yml runs on a newer Ubuntu with a newer libstdc++
    setup_uv = next(s for s in jobs["build"]["steps"] if "setup-uv" in s.get("uses", ""))
    assert "matrix.runner" in setup_uv["with"]["cache-suffix"]
    assert "release" in str(setup_uv["with"]["enable-cache"])  # releases build from scratch


def test_inno_setup_messages_exist_in_every_language():
    raw = (PACKAGING / "windows/installer.iss").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 with BOM, or Inno reads it as ANSI
    text = raw.decode("utf-8-sig")
    languages = re.findall(r'^Name: "(\w+)"; MessagesFile:', text, re.MULTILINE)
    assert len(languages) == 11
    messages: dict[str, set[str]] = {}
    for lang, key, value in re.findall(r"^(\w+)\.(\w+)=(.*)$", text, re.MULTILINE):
        messages.setdefault(key, set()).add(lang)
        placeholders = set(re.findall(r"%\d", value))
        english = re.search(rf"^en\.{key}=(.*)$", text, re.MULTILINE).group(1)
        assert placeholders == set(re.findall(r"%\d", english)), (lang, key)
    used = set(re.findall(r"CustomMessage\('(\w+)'\)", text))
    assert used and used <= set(messages)
    for key, langs in messages.items():
        assert langs == set(languages), f"{key} is missing in {set(languages) - langs}"
    assert "quit-app" in text and "migrate --desktop" in text


def test_inno_setup_code_has_no_lines_the_compiler_misreads():
    """Inno Setup reads every line that starts with "[" as a section header, also indented
    inside [Code] ("Invalid section tag" broke the 1.2 release build), and every line that
    starts with "#" as a preprocessor directive."""
    text = (PACKAGING / "windows/installer.iss").read_bytes().decode("utf-8-sig")
    sections = re.findall(r"^\[(\w+)\]\s*$", text, re.MULTILINE)
    assert sections[-1] == "Code" and len(sections) == len(set(sections))
    code = text.split("\n[Code]", 1)[1].splitlines()
    for number, line in enumerate(code, start=1):
        stripped = line.strip()
        assert not stripped.startswith("["), f"[Code] line {number}: {line!r}"
        assert not stripped.startswith("#"), f"[Code] line {number}: {line!r}"
    # Pascal: begin/end and parentheses balance (outside strings and comments)
    body = re.sub(r"'[^']*'|\{[^}]*\}|//[^\n]*", "", "\n".join(code))
    assert body.count("(") == body.count(")")
    opened = re.findall(r"\b(begin|try)\b", body, re.IGNORECASE)
    assert len(opened) == len(re.findall(r"\bend\b", body, re.IGNORECASE))


def test_macos_package_files():
    distribution = ET.parse(PACKAGING / "macos/distribution.xml").getroot()
    ref = distribution.find("pkg-ref[@version]")
    assert ref.get("id") == "io.github.w1nte.mira" and ref.get("version") == "@VERSION@"
    assert distribution.find("options").get("hostArchitectures") == "arm64"
    assert distribution.find("volume-check/allowed-os-versions/os-version").get("min") == "15.0"
    for page in ("welcome", "conclusion"):
        assert (PACKAGING / "macos/resources" / distribution.find(page).get("file")).is_file()
    spec = (PACKAGING / "vision-tagger.spec").read_text(encoding="utf-8")
    assert 'BUNDLE_ID = "io.github.w1nte.mira"' in spec
    assert "identifier=io.github.w1nte.mira" in (PACKAGING / "macos/build_pkg.sh").read_text()
    for script in ("preinstall", "postinstall"):
        text = (PACKAGING / "macos/scripts" / script).read_text()
        assert "sudo -u" in text and text.rstrip().endswith("exit 0")


SCRIPTS = [
    "linux/install.sh",
    "linux/build_tarball.sh",
    "macos/build_pkg.sh",
    "macos/scripts/preinstall",
    "macos/scripts/postinstall",
]


def test_scripts_are_executable_in_git():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-s", *(f"packaging/{s}" for s in SCRIPTS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    modes = {line.split()[3]: line.split()[0] for line in out.splitlines()}
    if not modes:
        pytest.skip("the scripts are not committed yet")
    assert all(mode == "100755" for mode in modes.values()), modes


# --------------------------------------------------------------------------- install.sh
posix_only = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None, reason="install.sh is for Linux"
)

CLI_STUB = """#!/bin/sh
# stands in for the frozen vision-tagger-cli: logs how it was called
echo "$0 $*" >> "$STUB_LOG"
case "$1" in
  quit-app) echo "Mira is not running"; exit "${STUB_QUIT_RC:-0}" ;;
  migrate) echo "backup:   /library/db.bak-0003-x"; exit "${STUB_MIGRATE_RC:-0}" ;;
esac
"""


def make_archive(base: Path, version: str, files: dict[str, str]) -> Path:
    """An unpacked release archive, as build_tarball.sh makes it, with a stub app."""
    folder = base / f"Mira-{version}-linux-x86_64-cpu"
    app = folder / "app"
    (app / "_internal").mkdir(parents=True)
    for name, content in {"vision-tagger-cli": CLI_STUB, "Mira": CLI_STUB, **files}.items():
        (app / name).write_text(content)
        (app / name).chmod(0o755)
    shutil.copy(PACKAGING / "linux/install.sh", folder / "install.sh")
    shutil.copy(ROOT / "src/vision_tagger/web/static/icons/icon-512.png", folder / "mira.png")
    (folder / "VERSION").write_text(version + "\n")
    return folder


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home


def run_install(home: Path, archive: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "STUB_LOG": str(home / "calls.log"),
        "LANG": "C.UTF-8",
        **env,
    }
    return subprocess.run(
        ["sh", str(archive / "install.sh"), *args],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def calls(home: Path) -> list[str]:
    log = home / "calls.log"
    lines = log.read_text().splitlines() if log.exists() else []
    log.unlink(missing_ok=True)
    return lines


@posix_only
def test_install_update_and_uninstall(tmp_path, home):
    prefix = home / ".local/share/mira"
    first = make_archive(tmp_path / "v1", "1.0.0", {"_internal/old.txt": "old"})
    r = run_install(home, first)
    assert r.returncode == 0, r.stderr
    assert "Installing Mira 1.0.0" in r.stdout
    assert (prefix / "VERSION").read_text() == "1.0.0\n"
    assert (prefix / "app/_internal/old.txt").exists()
    assert os.access(prefix / "app/Mira", os.X_OK)
    # quit with the archive's command line first, migrate with the installed one at the end
    assert calls(home) == [
        f"{first}/app/vision-tagger-cli quit-app --timeout 120",
        f"{prefix}/app/vision-tagger-cli migrate --desktop",
    ]
    assert "backup:" in r.stdout
    launcher = home / ".local/bin/vision-tagger-cli"
    assert launcher.stat().st_mode & stat.S_IXUSR
    subprocess.run([str(launcher), "status"], env={"STUB_LOG": str(home / "calls.log")}, check=True)
    assert calls(home) == [f"{prefix}/app/vision-tagger-cli status"]
    entry = (home / ".local/share/applications/io.github.w1nte.mira.desktop").read_text()
    assert f'Exec="{prefix}/app/Mira"' in entry and "Name=Mira" in entry
    icon = home / ".local/share/icons/hicolor/512x512/apps/io.github.w1nte.mira.png"
    assert icon.is_file() and f"Icon={icon}" in entry

    # the update replaces the files (no leftovers of the old version) and migrates
    second = make_archive(tmp_path / "v2", "1.1.0", {"_internal/new.txt": "new"})
    r = run_install(home, second)
    assert r.returncode == 0, r.stderr
    assert "Updating Mira 1.0.0 → 1.1.0" in r.stdout
    assert (prefix / "VERSION").read_text() == "1.1.0\n"
    assert (prefix / "app/_internal/new.txt").exists()
    assert not (prefix / "app/_internal/old.txt").exists()
    assert sorted(p.name for p in prefix.iterdir()) == ["VERSION", "app"]  # no temp folders
    assert calls(home) == [
        f"{second}/app/vision-tagger-cli quit-app --timeout 120",
        f"{prefix}/app/vision-tagger-cli migrate --desktop",
    ]

    # uninstall: the app goes, the library stays
    library = home / ".local/share/vision-tagger"
    (library / "data").mkdir(parents=True)
    (library / "data/vision_tagger.sqlite3").write_text("db")
    r = run_install(home, second, "--uninstall")
    assert r.returncode == 0, r.stderr
    assert calls(home) == [f"{prefix}/app/vision-tagger-cli quit-app --timeout 120"]
    assert not prefix.exists()
    assert not launcher.exists() and not (home / ".local/bin/mira").exists()
    assert not icon.exists()
    assert not (home / ".local/share/applications/io.github.w1nte.mira.desktop").exists()
    assert (library / "data/vision_tagger.sqlite3").read_text() == "db"
    assert "is kept" in r.stdout


@posix_only
def test_update_stops_when_mira_does_not_quit(tmp_path, home):
    prefix = home / ".local/share/mira"
    assert run_install(home, make_archive(tmp_path / "v1", "1.0.0", {"a.txt": "1"})).returncode == 0
    calls(home)
    r = run_install(home, make_archive(tmp_path / "v2", "2.0.0", {}), STUB_QUIT_RC="1")
    assert r.returncode == 1
    assert "still running" in r.stderr
    assert (prefix / "VERSION").read_text() == "1.0.0\n"  # the old version is untouched
    assert (prefix / "app/a.txt").exists()
    assert sorted(p.name for p in prefix.iterdir()) == ["VERSION", "app"]
    assert calls(home) == [
        f"{tmp_path}/v2/Mira-2.0.0-linux-x86_64-cpu/app/vision-tagger-cli quit-app --timeout 120"
    ]


@posix_only
@pytest.mark.parametrize(("code", "message"), [("3", "newer version"), ("4", "exit code 4")])
def test_failed_migration_is_a_warning(tmp_path, home, code, message):
    r = run_install(home, make_archive(tmp_path, "1.0.0", {}), STUB_MIGRATE_RC=code)
    assert r.returncode == 0
    assert message in r.stderr
    assert (home / ".local/share/mira/VERSION").exists()


@posix_only
def test_prefix_with_spaces_and_foreign_launchers(tmp_path, home):
    (home / ".local/bin").mkdir(parents=True)
    (home / ".local/bin/mira").write_text("#!/bin/sh\necho another mira\n")
    prefix = tmp_path / "my apps" / "it's mira"
    r = run_install(home, make_archive(tmp_path, "1.0.0", {}), "--prefix", str(prefix))
    assert r.returncode == 0, r.stderr
    assert "is not Mira's" in r.stderr
    assert (home / ".local/bin/mira").read_text() == "#!/bin/sh\necho another mira\n"
    calls(home)
    launcher = home / ".local/bin/vision-tagger-cli"
    subprocess.run([str(launcher), "x"], env={"STUB_LOG": str(home / "calls.log")}, check=True)
    assert calls(home) == [f"{prefix}/app/vision-tagger-cli x"]
    r = run_install(
        home, make_archive(tmp_path / "again", "1.0.0", {}), "--uninstall", f"--prefix={prefix}"
    )
    assert r.returncode == 0, r.stderr
    assert not prefix.exists() and not launcher.exists()
    assert (home / ".local/bin/mira").exists()  # not ours, not removed
