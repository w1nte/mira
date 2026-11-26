"""Release notes from the changelog (src/vision_tagger/resources/CHANGELOG.md).

    python packaging/changelog.py notes 1.2.1    # prints the notes of that release (Markdown)
    python packaging/changelog.py check          # reports format problems, exit code 1 if any
    python packaging/changelog.py release 1.5    # files the Unreleased entries under ## 1.5

``notes`` takes the section of the tag (``v1.3`` finds ``## 1.3``, ``1.4-rc.1`` finds
``## 1.4rc1``). Without one it takes ``## Unreleased`` and warns: in GitHub Actions as a
``::warning::`` annotation, else on stderr. When that is empty too it prints nothing. A
changelog with format problems is refused (exit code 1).

``release`` does the release step of CLAUDE.md ("Changelog"): the entries under Unreleased move
into a new ``## <tag> - <date>`` section (today, or ``--date``), and Unreleased stays, empty. With
``--entries-from FILE`` only the entries that were under Unreleased in FILE move (the changelog
of the released commit), so entries added to main after the release stay Unreleased. A tag that
already has a section changes nothing. The desktop workflow runs it for every release, so the
app it builds and the release notes show the version even when the step was forgotten, and it
proposes the result for main in a pull request.

The desktop workflow's publish job uses ``notes`` for releases published without notes. The
script needs nothing but Python: the parser
(src/vision_tagger/changelog.py) is loaded by its path, without the package and its
dependencies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import io
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "vision_tagger"
CHANGELOG = SRC / "resources" / "CHANGELOG.md"


def _parser_module():
    spec = importlib.util.spec_from_file_location("vt_changelog", SRC / "changelog.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while they are made
    spec.loader.exec_module(module)
    return module


changelog = _parser_module()


def warn(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning title=Release notes::{message}", file=sys.stderr, flush=True)
    else:
        print(f"changelog.py: warning: {message}", file=sys.stderr, flush=True)


def notes(text: str, tag: str) -> str:
    """The release notes of ``tag``: its section, else Unreleased (with a warning)."""
    releases = changelog.parse(text)
    release = changelog.find(releases, tag)
    if release is None or release.unreleased:
        release = changelog.find(releases, changelog.UNRELEASED)
        if release is None or not release.items:
            warn(f"CHANGELOG.md has no section for {tag} and nothing under Unreleased: no notes")
            return ""
        warn(f"CHANGELOG.md has no section for {tag}; using the Unreleased section")
    return changelog.to_markdown(release)


def intro(text: str) -> str:
    """Everything before the first ``##`` line: the title and the introduction, as written."""
    lines = text.replace("\r\n", "\n").split("\n")
    for index, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[:index]).rstrip("\n")
    return text.rstrip("\n")


def render(head: str, releases: list) -> str:
    """A changelog from its introduction and releases (renders a valid file as it was)."""
    parts = [head.rstrip("\n") + "\n"]
    for release in releases:
        if release.unreleased:
            title = f"## {changelog.UNRELEASED}"
        else:
            title = f"## {release.version} - {release.date.isoformat()}"
        body = changelog.to_markdown(release)
        parts.append(title + "\n" + ("\n" + body if body else ""))
    return "\n".join(parts)


def cut_release(
    text: str, tag: str, day: dt.date, entries_from: str | None = None
) -> tuple[str, str]:
    """``text`` with the Unreleased entries filed under ``tag``, and what was done."""
    releases = changelog.parse(text)
    released = [r for r in releases if not r.unreleased]
    if changelog.find(released, tag) is not None:
        return text, f"the changelog already has a section for {tag}"
    source = changelog.parse(entries_from) if entries_from is not None else releases
    moved = changelog.find(source, changelog.UNRELEASED)
    if moved is None or not moved.items:
        return text, f"nothing under Unreleased to file under {tag}"
    current = changelog.find(releases, changelog.UNRELEASED)
    kept: dict[str, list[str]] = {}
    for kind in changelog.KINDS:
        items = [
            item
            for item in (current.sections.get(kind, []) if current is not None else [])
            if item not in moved.sections.get(kind, [])
        ]
        if items:
            kept[kind] = items
    version = tag.strip()
    if version[:1] in "vV" and changelog.version_key(version[1:]) is not None:
        version = version[1:]  # like the sections before it: ## 1.5, not ## v1.5
    new = changelog.Release(version=version, date=day, sections=dict(moved.sections))
    key = changelog.version_key(version)
    # newest first: a patch of an older line (1.4.1 after 1.5) goes below the newer releases
    position = next(
        (i for i, r in enumerate(released) if (changelog.version_key(r.version) or ()) < key),
        len(released),
    )
    ordered = [*released[:position], new, *released[position:]]
    unreleased = changelog.Release(version=changelog.UNRELEASED, sections=kept)
    result = render(intro(text), [unreleased, *ordered])
    problems = changelog.validate(result)
    if problems:  # e.g. a patch of an older line dated after a newer release: leave it alone
        warn(f"could not file Unreleased under {version}: {'; '.join(problems)}")
        return text, f"left unchanged: the section for {version} would break the format"
    return result, f"filed {new.items} entries under {version} - {day.isoformat()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", type=Path, default=CHANGELOG, help="the changelog to read")
    commands = parser.add_subparsers(dest="command", required=True)
    notes_cmd = commands.add_parser("notes", help="print the release notes of a tag")
    notes_cmd.add_argument("tag", help="the release tag, e.g. 1.2.1 or v1.3")
    commands.add_parser("check", help="report format problems")
    release_cmd = commands.add_parser(
        "release", help="file the Unreleased entries under a new section for the tag"
    )
    release_cmd.add_argument("tag", help="the release tag, e.g. 1.5 or v1.5")
    release_cmd.add_argument(
        "--date", type=dt.date.fromisoformat, help="the release date (default: today, UTC)"
    )
    release_cmd.add_argument(
        "--entries-from",
        type=Path,
        help="move only the entries that are under Unreleased in this changelog",
    )
    args = parser.parse_args(argv)
    try:
        text = args.file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"changelog.py: {exc}", file=sys.stderr)
        return 1
    problems = changelog.validate(text)
    if problems:
        print(f"changelog.py: {args.file} breaks the format:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if args.command == "notes":
        sys.stdout.write(notes(text, args.tag))
    elif args.command == "release":
        source = None
        if args.entries_from is not None:
            source = args.entries_from.read_text(encoding="utf-8")
            if changelog.validate(source):
                print(f"changelog.py: {args.entries_from} breaks the format", file=sys.stderr)
                return 1
        day = args.date or dt.datetime.now(dt.UTC).date()
        result, message = cut_release(text, args.tag, day, source)
        if result != text:
            args.file.write_text(result, encoding="utf-8", newline="\n")
        print(f"{args.file.name}: {message}")
    else:
        released = [r for r in changelog.parse(text) if not r.unreleased]
        print(f"{args.file.name}: Unreleased and {len(released)} releases, no problems")
    return 0


if __name__ == "__main__":
    # the notes are UTF-8 (arrows, quotes), also when Windows redirects them to a file
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
