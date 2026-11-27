"""The changelog: the repository's file, the parser and validator, the release notes script,
the "What's new" page and the release workflow step."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from vision_tagger import changelog
from vision_tagger.web.app import create_app
from vision_tagger.web.routes import changelog as changelog_route

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
BUNDLED = ROOT / "src" / "vision_tagger" / "resources" / "CHANGELOG.md"
DE = {"Accept-Language": "de"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notes_script = _load("changelog")
version_script = _load("version")

GOOD = """\
# Changelog

Some introduction, with a - dash and a [link](https://example.com).

## Unreleased

### Fixed
- Something that was broken works again.

## 1.4-rc.1 - 2026-12-02

### New
- A brand new thing, see `vision-tagger new`.

## 1.3 - 2026-11-26

### New
- First new thing.
- Second new thing with a [link](https://example.com/a?b=1&c=2).

### Changed
- Something works differently.

### Fixed
- A bug is gone.

## v1.2.1 - 2026-11-26

### Fixed
- An older bug is gone.
"""


# --------------------------------------------------------------------------- the repository file
def test_the_repository_changelog_is_valid():
    text = changelog.read()
    assert text == BUNDLED.read_text(encoding="utf-8")
    assert changelog.validate(text) == []
    releases = changelog.parse(text)
    assert releases[0].unreleased
    versions = [r.version for r in releases[1:]]
    for tag in ("1.3", "1.2.1", "1.2", "1.1", "1.0"):
        assert tag in versions, tag
    assert all(r.date is not None for r in releases[1:])


def test_every_version_is_a_release_tag_of_the_desktop_workflow():
    """packaging/version.py turns each heading into the version the app reports, and that
    version finds its section again (the page marks the running version with it)."""
    releases = changelog.releases()
    for release in releases[1:]:
        version = version_script.from_tag(release.version)
        assert changelog.find(releases, version) is release, release.version


def test_the_root_changelog_points_to_the_bundled_one():
    root = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "](src/vision_tagger/resources/CHANGELOG.md)" in root
    assert not re.search(r"^##", root, re.MULTILINE)  # entries go into the bundled file only


def test_the_changelog_is_package_data():
    """Inside the package: the wheel (hatch, packages = src/vision_tagger), the desktop app
    (PyInstaller's collect_data_files("vision_tagger")) and Docker (COPY src) all carry it."""
    assert BUNDLED.is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["src/vision_tagger"]' in pyproject
    spec = (PACKAGING / "vision-tagger.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("vision_tagger")' in spec
    # the built app's self-test opens the page, so a build without the file fails
    desktop = (ROOT / "src" / "vision_tagger" / "desktop.py").read_text(encoding="utf-8")
    assert '"changelog"' in desktop.split("def self_test", 1)[1]


def test_claude_md_makes_the_changelog_mandatory():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "## Changelog (mandatory)" in text
    assert "tests/test_changelog.py" in text


# --------------------------------------------------------------------------- parse
def test_parse():
    releases = changelog.parse(GOOD)
    assert [r.version for r in releases] == ["Unreleased", "1.4-rc.1", "1.3", "v1.2.1"]
    unreleased, rc, v13, v121 = releases
    assert unreleased.unreleased and unreleased.date is None
    assert unreleased.sections == {"Fixed": ["Something that was broken works again."]}
    assert rc.date == date(2026, 12, 2)
    assert list(v13.sections) == ["New", "Changed", "Fixed"]
    assert v13.sections["New"][1].startswith("Second new thing with a [link]")
    assert v13.items == 4 and v121.items == 1
    assert changelog.validate(GOOD) == []


def test_parse_tolerates_crlf_a_bom_and_an_empty_unreleased():
    text = "﻿" + GOOD.replace("### Fixed\n- Something that was broken works again.\n", "")
    text = text.replace("\n", "\r\n")
    assert changelog.validate(text) == []
    releases = changelog.parse(text)
    assert releases[0].unreleased and releases[0].items == 0
    assert releases[2].sections["Changed"] == ["Something works differently."]


def test_versions_match_however_the_tag_is_written():
    releases = changelog.parse(GOOD)
    assert changelog.find(releases, "1.3").version == "1.3"
    assert changelog.find(releases, "v1.3").version == "1.3"
    assert changelog.find(releases, "1.3.0").version == "1.3"
    assert changelog.find(releases, "1.4rc1").version == "1.4-rc.1"
    assert changelog.find(releases, "v1.4-RC.1").version == "1.4-rc.1"
    assert changelog.find(releases, "1.2.1").version == "v1.2.1"
    assert changelog.find(releases, "unreleased").unreleased
    for missing in ("1.4", "1.3.1", "0.1.0", "", "latest", "Unreleased-ish"):
        assert changelog.find(releases, missing) is None, missing
    assert changelog.section("1.3", GOOD).items == 4
    assert changelog.section("9.9", GOOD) is None


def test_version_order():
    key = changelog.version_key
    assert key("1.4a1") < key("1.4b2") < key("1.4rc1") < key("1.4") < key("1.4.1")
    assert key("1.10") > key("1.9") and key("2.0") > key("1.99.9")
    assert key("v1.3") == key("1.3.0") == key("1.3")
    assert key("1.2.3.4") is None and key("one") is None


# --------------------------------------------------------------------------- validate
def _with(body: str) -> str:
    return f"# Changelog\n\n## Unreleased\n\n{body}"


RELEASE = "## 1.0 - 2026-11-26\n\n### New\n- First.\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Unreleased\n", "starts with '# Changelog'"),
        ("", "starts with '# Changelog'"),
        ("# Changelog\n\n" + RELEASE, "'## Unreleased' must be the first section"),
        (_with(RELEASE + "\n## Unreleased\n"), "must be the first section, and only once"),
        (_with("## 1.2 (2026-11-26)\n### New\n- x\n"), "release heading is '## <version>"),
        (_with("## 1.2 \u2013 2026-11-26\n### New\n- x\n"), "release heading is '## <version>"),
        (_with("## latest - 2026-11-26\n### New\n- x\n"), "'latest' is not a release tag"),
        (_with("## 1.2 - 2026-02-30\n### New\n- x\n"), "'2026-02-30' is not a date"),
        (_with("## 1.2 - 26.11.2026\n### New\n- x\n"), "'26.11.2026' is not a date"),
        (_with("## 1.2 - 2026-1-5\n### New\n- x\n"), "'2026-1-5' is not a date"),
        (_with(RELEASE + "## 1.1 - 2026-11-20\n### New\n- x\n"), "1.1 is newer than 1.0"),
        (_with(RELEASE + "## 1.0.0 - 2026-11-20\n### New\n- x\n"), "1.0.0 appears twice"),
        (_with("## 1.1 - 2026-11-01\n### New\n- x\n" + RELEASE), "1.0 is dated after 1.1"),
        (_with("### Added\n- x\n"), "unknown section 'Added'"),
        (_with("### Fixed\n- x\n### New\n- y\n"), "section 'New' is out of order"),
        (_with("### New\n- x\n### New\n- y\n"), "section 'New' appears twice"),
        (_with("### New\n- \n"), "empty item"),
        (_with("### New\n-\n"), "empty item"),
        (_with("### New\n### Fixed\n- x\n"), "section 'New' has no items"),
        (_with("## 1.0 - 2026-11-26\n"), "1.0: a release lists at least one item"),
        (_with("- x\n"), "item outside a section"),
        (_with("### New\n- x\n  continued\n"), "no other bullets, no nesting"),
        (_with("### New\n- x\nmore text\n"), "text outside an item"),
        (_with("### New\n* x\n"), "no other bullets"),
        (_with("### New\n-x\n"), "write items as '- text'"),
        (_with("#### Details\n"), "only '## <version>' and '### <section>' headings"),
        ("# Changelog\n### New\n- x\n## Unreleased\n", "comes before the first release"),
        (_with("### New\n- <b>bold</b>\n"), "raw HTML is not allowed"),
        (_with("### New\n- see [here](javascript:alert(1))\n"), "links need an http(s) address"),
        (_with("### New\n- see [here](/changelog)\n"), "links need an http(s) address"),
        (_with("### New\n- run `vision-tagger\n"), "unbalanced `"),
        (_with("### New\n- " + "x" * 161 + "\n"), "keep it under 160"),
    ],
)
def test_validate_reports_problems(text, expected):
    problems = changelog.validate(text)
    assert any(expected in p for p in problems), problems


def test_validate_accepts_what_the_rules_allow():
    ok = _with(
        "### New\n- Use `<tag>` in code, `a|b`, and [a link](http://example.com/x_y).\n"
        "- Arrows → and quotes “like this” are fine, so is 3 < 4.\n\n" + RELEASE
    )
    assert changelog.validate(ok) == []


def test_validate_names_the_line():
    problems = changelog.validate(_with("### New\n- x\n### Added\n- y\n"))
    assert problems == ["line 7: unknown section 'Added'; use New, Changed, Fixed"]


# --------------------------------------------------------------------------- rendering
def test_to_markdown():
    text = changelog.to_markdown(changelog.section("1.3", GOOD))
    assert text == (
        "### New\n- First new thing.\n"
        "- Second new thing with a [link](https://example.com/a?b=1&c=2).\n\n"
        "### Changed\n- Something works differently.\n\n"
        "### Fixed\n- A bug is gone.\n"
    )
    assert changelog.to_markdown(changelog.Release("Unreleased")) == ""


def test_inline_html_escapes_everything_but_code_and_web_links():
    html = changelog.inline_html(
        'Use `<b>&"` & <script>alert(1)</script> [docs](https://example.com/?a=1&b="2) '
        "[site](https://example.com/x?a=1&b=2) [bad](javascript:alert(1)) [rel](/x)"
    )
    assert "<script" not in html and "<b>" not in html
    assert "<code>&lt;b&gt;&amp;&quot;</code>" in html
    assert "&amp; &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert (
        '<a href="https://example.com/x?a=1&amp;b=2" target="_blank" '
        'rel="noopener noreferrer">site</a>'
    ) in html
    assert "javascript" in html and 'href="javascript' not in html  # stays text
    assert 'href="/x"' not in html and "[rel](/x)" in html
    assert html.count("<a ") == 1  # the address with a quote in it is not a link either


# --------------------------------------------------------------------------- notes script
def test_notes_for_a_version(capsys):
    assert notes_script.main(["notes", "1.2.1"]) == 0
    out, err = capsys.readouterr()
    assert out == changelog.to_markdown(changelog.section("1.2.1"))
    assert out.startswith("### Fixed\n- ") and err == ""


def test_notes_for_a_tag_written_differently(tmp_path, capsys):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(GOOD, encoding="utf-8")
    assert notes_script.main(["--file", str(path), "notes", "v1.3.0"]) == 0
    out, err = capsys.readouterr()
    assert out == changelog.to_markdown(changelog.section("1.3", GOOD)) and err == ""
    assert notes_script.main(["--file", str(path), "notes", "1.4rc1"]) == 0
    assert "A brand new thing" in capsys.readouterr().out


def test_notes_fall_back_to_unreleased_with_a_warning(tmp_path, capsys, monkeypatch):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(GOOD, encoding="utf-8")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert notes_script.main(["--file", str(path), "notes", "2.0"]) == 0
    out, err = capsys.readouterr()
    assert out == "### Fixed\n- Something that was broken works again.\n"
    assert "warning: CHANGELOG.md has no section for 2.0; using the Unreleased section" in err
    monkeypatch.setenv("GITHUB_ACTIONS", "true")  # an annotation in the job log
    assert notes_script.main(["--file", str(path), "notes", "not-a-tag"]) == 0
    assert capsys.readouterr().err.startswith("::warning title=Release notes::")


def test_notes_without_a_section_and_an_empty_unreleased(tmp_path, capsys):
    path = tmp_path / "CHANGELOG.md"
    path.write_text("# Changelog\n\n## Unreleased\n\n" + RELEASE, encoding="utf-8")
    assert notes_script.main(["--file", str(path), "notes", "2.0"]) == 0
    out, err = capsys.readouterr()
    assert out == "" and "nothing under Unreleased: no notes" in err
    assert notes_script.main(["--file", str(path), "notes", "Unreleased"]) == 0
    assert capsys.readouterr().out == ""


def test_the_notes_script_refuses_a_broken_changelog(tmp_path, capsys):
    path = tmp_path / "CHANGELOG.md"
    path.write_text("# Changelog\n\n## Unreleased\n\n### Added\n- x\n", encoding="utf-8")
    assert notes_script.main(["--file", str(path), "notes", "1.0"]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "unknown section 'Added'" in err
    assert notes_script.main(["--file", str(path), "check"]) == 1
    assert notes_script.main(["--file", str(tmp_path / "missing.md"), "check"]) == 1
    capsys.readouterr()
    assert notes_script.main(["check"]) == 0
    assert "no problems" in capsys.readouterr().out


def test_the_notes_script_needs_nothing_but_python(tmp_path):
    """The publish job runs it with the runner's python3: no venv, no site-packages (-S)."""
    run = subprocess.run(
        [sys.executable, "-I", "-S", str(PACKAGING / "changelog.py"), "notes", "1.3"],
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == changelog.to_markdown(changelog.section("1.3"))
    assert "no `ollama pull` needed" in run.stdout and "→" in run.stdout  # UTF-8 output


# --------------------------------------------------------------------------- the page
@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def _section(page: str, anchor: str) -> str:
    match = re.search(rf'<section [^>]*id="{re.escape(anchor)}".*?</section>', page, re.DOTALL)
    assert match, anchor
    return match.group(0)


def test_the_page_lists_the_releases_and_marks_the_current_version(client, monkeypatch):
    monkeypatch.setattr(changelog_route, "__version__", "1.2.1")
    page = client.get("/changelog").text
    assert "<title>What\u2019s new · Mira</title>" in page
    assert "You are using Mira 1.2.1." in page
    for release in changelog.releases()[1:]:
        assert f'id="v{release.version}"' in page
        assert f'<time datetime="{release.date.isoformat()}">' in page
    current = _section(page, "v1.2.1")
    assert 'class="card changelog-release current"' in current and "Current version" in current
    assert page.count("Current version") == 1 and page.count("changelog-release current") == 1
    assert ">Fixed</h3>" in current and "Windows setup programs" in current
    v13 = _section(page, "v1.3")
    assert ">New</h3>" in v13 and "<code>ollama pull</code>" in v13
    assert '<a href="https://ollama.com/download" target="_blank"' in v13
    # newest first
    assert page.index('id="v1.3"') < page.index('id="v1.2.1"') < page.index('id="v1.0"')


def test_an_unknown_version_marks_nothing(client, monkeypatch):
    monkeypatch.setattr(changelog_route, "__version__", "0.1.0")
    page = client.get("/changelog").text
    assert "You are using Mira 0.1.0." in page and "Current version" not in page


def test_unreleased_shows_only_when_it_lists_something(client, monkeypatch):
    sample = changelog.parse(GOOD)
    monkeypatch.setattr(changelog, "releases", lambda: sample)
    page = client.get("/changelog").text
    assert ">Unreleased</h2>" in page and "Not in a release yet" in page
    assert "Something that was broken works again." in _section(page, "unreleased")
    without = [changelog.Release("Unreleased"), *sample[1:]]
    monkeypatch.setattr(changelog, "releases", lambda: without)
    page = client.get("/changelog").text
    assert 'id="unreleased"' not in page and 'id="v1.3"' in page


def test_the_page_never_renders_html_from_entries(client, monkeypatch):
    evil = changelog.parse(
        "# Changelog\n\n## Unreleased\n\n### New\n"
        '- <script>alert(1)</script> <img src=x onerror="alert(2)"> `<b>code</b>` '
        "[click](javascript:alert(3)) [ok](https://example.com/?a=1&b=2)\n"
    )
    assert evil[0].items == 1  # parse keeps it; validate would refuse it
    monkeypatch.setattr(changelog, "releases", lambda: evil)
    page = client.get("/changelog").text
    item = _section(page, "unreleased")
    assert "<script>alert(1)" not in page and "<img src=x" not in item
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in item
    assert "&lt;img src=x onerror=&#34;alert(2)&#34;&gt;" in item or "onerror=&quot;" in item
    assert "<code>&lt;b&gt;code&lt;/b&gt;</code>" in item
    assert 'href="javascript' not in item
    assert '<a href="https://example.com/?a=1&amp;b=2" target="_blank"' in item


def test_the_page_chrome_is_translated_and_the_entries_stay_english(client, monkeypatch):
    monkeypatch.setattr(changelog_route, "__version__", "1.2")
    page = client.get("/changelog", headers=DE).text
    assert '<html lang="de"' in page
    assert "<h1>Was ist neu</h1>" in page and "Du verwendest Mira 1.2." in page
    current = _section(page, "v1.2")
    assert "Aktuelle Version" in current
    assert ">Neu</h3>" in current and ">Geändert</h3>" in current and ">Behoben</h3>" in current
    assert '<ul class="changelog-items" lang="en">' in current
    assert "Database page: export the library database" in current  # English, as written
    assert ">New</h3>" not in page and "What\u2019s new" not in page


def test_the_page_is_linked_from_the_sidebar_and_the_environment_card(client):
    for url in ("/", "/images", "/pipeline"):
        page = client.get(url).text
        assert re.search(r'<a class="foot-link[^"]*" href="/changelog"', page), url
    assert 'aria-current="page">' in client.get("/changelog").text.split('href="/changelog"')[1]
    health = client.get("/partials/health").text
    # one Version row (shared with the update check): the version links to the page
    assert health.count("<dt>Version</dt>") == 1
    assert re.search(r'<a href="/changelog" title="What\u2019s new">Mira [^<]+</a>', health)
    assert 'title="Was ist neu"' in client.get("/partials/health", headers=DE).text


# --------------------------------------------------------------------------- release workflow
def test_the_publish_job_takes_its_notes_from_the_changelog():
    workflow = yaml.safe_load((ROOT / ".github/workflows/desktop.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["publish"]
    # the upload, plus the pull request that files a forgotten release section on main
    assert job["permissions"] == {"contents": "write", "pull-requests": "write"}
    steps = job["steps"]
    checkout = next(i for i, s in enumerate(steps) if "actions/checkout" in s.get("uses", ""))
    notes = next(i for i, s in enumerate(steps) if "changelog.py notes" in s.get("run", ""))
    upload = next(i for i, s in enumerate(steps) if "gh release upload" in s.get("run", ""))
    assert checkout < notes < upload
    step = steps[notes]
    assert step["continue-on-error"] is True  # never keeps the files from being attached
    assert step["env"]["TAG"] == "${{ github.event.release.tag_name }}"
    run = step["run"]
    assert "${{" not in run  # values come in through env, not into the script
    assert 'packaging/changelog.py notes "$TAG" > notes.md' in run
    assert "--notes-file notes.md" in run and "gh release edit" in run
    # only a release without notes is changed
    body_check = run.index("tr -d '[:space:]'")
    assert body_check < run.index("gh release edit")
    assert "exit 0" in run[body_check : run.index("packaging/changelog.py")]


# --------------------------------------------------------------------------- the release step
RELEASE_SAMPLE = """# Changelog

Notes.

## Unreleased

### New
- Search understands "or".

### Fixed
- The sidebar no longer scrolls sideways.

## 1.4 - 2026-11-20

### New
- Update check.
"""


def test_rendering_a_valid_changelog_gives_it_back_unchanged():
    """The release step rewrites the file from its parsed releases: that must not reformat."""
    for text in (BUNDLED.read_text(encoding="utf-8"), RELEASE_SAMPLE):
        rendered = notes_script.render(notes_script.intro(text), changelog.parse(text))
        assert rendered == text


def test_the_release_step_files_unreleased_under_the_tag():
    text, message = notes_script.cut_release(RELEASE_SAMPLE, "v1.5", date(2026, 11, 26))
    assert message == "filed 2 entries under 1.5 - 2026-11-26"
    assert changelog.validate(text) == []
    releases = changelog.parse(text)
    assert [r.version for r in releases] == ["Unreleased", "1.5", "1.4"]  # "v" dropped
    assert releases[0].items == 0
    assert releases[1].sections == {
        "New": ['Search understands "or".'],
        "Fixed": ["The sidebar no longer scrolls sideways."],
    }
    # once is enough: a tag with a section changes nothing
    again, message = notes_script.cut_release(text, "1.5", date(2026, 11, 27))
    assert again == text and "already has a section" in message


def test_the_release_step_keeps_entries_added_after_the_release():
    """main may already have new entries: only those of the released commit move."""
    released = RELEASE_SAMPLE
    main = RELEASE_SAMPLE.replace(
        '- Search understands "or".\n', '- Search understands "or".\n- NSFW photos are covered.\n'
    )
    text, _ = notes_script.cut_release(main, "1.5", date(2026, 11, 26), entries_from=released)
    unreleased, release_15 = changelog.parse(text)[:2]
    assert unreleased.sections == {"New": ["NSFW photos are covered."]}
    assert release_15.sections["New"] == ['Search understands "or".']


def test_two_forgotten_release_steps_file_each_entry_once():
    """1.6 and 1.7 were released without the step: 1.7's commit still lists 1.6's entries
    under Unreleased. Filing 1.6, then 1.7, must not put 1.6's entries under 1.7 as well."""
    at_16 = RELEASE_SAMPLE
    at_17 = RELEASE_SAMPLE.replace(
        "### Fixed\n", "- Similar photos.\n\n### Changed\n- Search ranks visually.\n\n### Fixed\n"
    )
    main = at_17.replace("### Fixed\n", "### Fixed\n- Merged after 1.7.\n")
    text, _ = notes_script.cut_release(main, "v1.6", date(2026, 11, 26), entries_from=at_16)
    text, message = notes_script.cut_release(text, "v1.7", date(2026, 11, 26), entries_from=at_17)
    assert message == "filed 2 entries under 1.7 - 2026-11-26"
    assert changelog.validate(text) == []
    unreleased, r17, r16 = changelog.parse(text)[:3]
    assert unreleased.sections == {"Fixed": ["Merged after 1.7."]}
    assert r17.sections == {"New": ["Similar photos."], "Changed": ["Search ranks visually."]}
    assert r16.sections == {
        "New": ['Search understands "or".'],
        "Fixed": ["The sidebar no longer scrolls sideways."],
    }
    # a released commit with nothing new for the tag changes nothing
    again, message = notes_script.cut_release(text, "1.8", date(2026, 11, 27), entries_from=at_16)
    assert again == text and "nothing under Unreleased" in message


def test_the_release_step_with_nothing_unreleased_changes_nothing():
    empty = (
        "# Changelog\n\nNotes.\n\n## Unreleased\n\n"
        "## 1.4 - 2026-11-20\n\n### New\n- Update check.\n"
    )
    text, message = notes_script.cut_release(empty, "1.5", date(2026, 11, 26))
    assert text == empty and "nothing under Unreleased" in message


def test_a_patch_of_an_older_line_goes_below_the_newer_releases():
    newer = RELEASE_SAMPLE.replace(
        "## 1.4 - 2026-11-20", "## 1.5 - 2026-11-27\n\n### New\n- X.\n\n## 1.4 - 2026-11-20"
    )
    text, _ = notes_script.cut_release(newer, "1.4.1", date(2026, 11, 26))
    assert [r.version for r in changelog.parse(text)] == ["Unreleased", "1.5", "1.4.1", "1.4"]
    assert changelog.validate(text) == []
    # dated after the newer 1.5 the format can't hold it: unchanged, with a warning, no crash
    # (the build goes on; the notes fall back to Unreleased)
    text, message = notes_script.cut_release(newer, "1.4.1", date(2026, 11, 28))
    assert text == newer and "left unchanged" in message


def test_the_release_command_writes_the_file(tmp_path, capsys):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(RELEASE_SAMPLE, encoding="utf-8")
    argv = ["--file", str(path), "release", "1.5", "--date", "2026-11-26"]
    assert notes_script.main(argv) == 0
    assert "## 1.5 - 2026-11-26" in path.read_text(encoding="utf-8")
    assert "filed 2 entries" in capsys.readouterr().out
    released = tmp_path / "released.md"
    released.write_text("not a changelog", encoding="utf-8")
    argv = ["--file", str(path), "release", "1.6", "--entries-from", str(released)]
    assert notes_script.main(argv) == 1  # a broken source is refused


def test_the_workflow_files_the_release_in_builds_notes_and_main():
    """Even when the release step was forgotten, the app, the notes and main get the section."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/desktop.yml").read_text(encoding="utf-8"))
    build = [s.get("run", "") for s in workflow["jobs"]["build"]["steps"]]
    cut = next(i for i, run in enumerate(build) if "changelog.py release" in run)
    install = next(i for i, run in enumerate(build) if "uv sync --frozen" in run)
    assert cut < install  # the install copies the package, changelog included
    publish = workflow["jobs"]["publish"]
    assert publish["permissions"] == {"contents": "write", "pull-requests": "write"}
    runs = [s.get("run", "") for s in publish["steps"]]
    cut = next(i for i, run in enumerate(runs) if 'changelog.py release "$TAG"' in run)
    notes = next(i for i, run in enumerate(runs) if "changelog.py notes" in run)
    pull = next(i for i, run in enumerate(runs) if "gh pr create" in run)
    assert cut < notes < pull
    assert "--entries-from" in runs[pull] and "released-changelog.md" in runs[cut]
    for step in (cut, pull):  # neither may keep the files from being attached
        assert publish["steps"][step].get("continue-on-error") is True
