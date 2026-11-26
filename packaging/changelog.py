"""Release notes from the changelog (src/vision_tagger/resources/CHANGELOG.md).

    python packaging/changelog.py notes 1.2.1    # prints the notes of that release (Markdown)
    python packaging/changelog.py check          # reports format problems, exit code 1 if any

``notes`` takes the section of the tag (``v1.3`` finds ``## 1.3``, ``1.4-rc.1`` finds
``## 1.4rc1``). Without one it takes ``## Unreleased`` and warns: in GitHub Actions as a
``::warning::`` annotation, else on stderr. When that is empty too it prints nothing. A
changelog with format problems is refused (exit code 1). The desktop workflow's publish job
uses it for releases published without notes; it needs nothing but Python: the parser
(src/vision_tagger/changelog.py) is loaded by its path, without the package and its
dependencies.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", type=Path, default=CHANGELOG, help="the changelog to read")
    commands = parser.add_subparsers(dest="command", required=True)
    notes_cmd = commands.add_parser("notes", help="print the release notes of a tag")
    notes_cmd.add_argument("tag", help="the release tag, e.g. 1.2.1 or v1.3")
    commands.add_parser("check", help="report format problems")
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
    else:
        released = [r for r in changelog.parse(text) if not r.unreleased]
        print(f"{args.file.name}: Unreleased and {len(released)} releases, no problems")
    return 0


if __name__ == "__main__":
    # the notes are UTF-8 (arrows, quotes), also when Windows redirects them to a file
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
