"""The app version from a release tag (PEP 440), written into pyproject.toml.

    python packaging/version.py v1.2.3               # prints 1.2.3 and sets [project] version
    python packaging/version.py v1.2.3-rc.1 --dry-run   # only prints 1.2.3rc1
    python packaging/version.py --current            # prints the version in pyproject.toml

Accepted tags: ``1.2.3`` or ``1.2``, with or without a leading ``v``, optionally followed by a
pre-release: ``a1``/``alpha1``, ``b1``/``beta1`` or ``rc1``, also written ``-alpha.1``,
``-beta.2``, ``-rc.1``. Anything else is rejected. The desktop workflow runs this before it
installs the app, so the built app (and its setup) carries the tag's version; it needs nothing
but Python.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

_TAG = re.compile(
    r"""
    v?
    (?P<release>\d+\.\d+(?:\.\d+)?)
    (?:
        [-._]?
        (?P<pre>alpha|a|beta|b|rc|c)
        [-._]?
        (?P<number>\d+)
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)
_PRE = {"alpha": "a", "a": "a", "beta": "b", "b": "b", "rc": "rc", "c": "rc"}
_VERSION_LINE = re.compile(r'^(\s*version\s*=\s*)"([^"]*)"(\s*(?:#.*)?)$')


def from_tag(tag: str) -> str:
    """``v1.2.3-rc.1`` -> ``1.2.3rc1``; ValueError for anything that is not a release tag."""
    m = _TAG.fullmatch(tag)
    if m is None:
        raise ValueError(
            f"{tag!r} is not a release tag; use e.g. v1.2.3, v1.2, v1.2.3-rc.1, v1.2.3-beta.2 "
            "or v1.2.3-alpha.1"
        )
    version = ".".join(str(int(part)) for part in m["release"].split("."))
    if m["pre"]:
        version += _PRE[m["pre"].lower()] + str(int(m["number"]))
    return version


def _project_version_line(lines: list[str]) -> tuple[int, re.Match[str]]:
    table = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            table = stripped
            continue
        m = _VERSION_LINE.match(line.rstrip("\r\n"))
        if table == "[project]" and m:
            return i, m
    raise ValueError('pyproject.toml has no `version = "..."` line in [project]')


def current(pyproject: Path = PYPROJECT) -> str:
    lines = pyproject.read_bytes().decode("utf-8").splitlines(keepends=True)
    return _project_version_line(lines)[1].group(2)


def set_version(version: str, pyproject: Path = PYPROJECT) -> None:
    """Replace the ``[project]`` version, keeping everything else byte for byte."""
    lines = pyproject.read_bytes().decode("utf-8").splitlines(keepends=True)
    i, m = _project_version_line(lines)
    ending = lines[i][len(lines[i].rstrip("\r\n")) :]
    lines[i] = f'{m.group(1)}"{version}"{m.group(3)}{ending}'
    pyproject.write_bytes("".join(lines).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", nargs="?", help="release tag, e.g. v1.2.3")
    parser.add_argument("--dry-run", action="store_true", help="only print the version")
    parser.add_argument("--current", action="store_true", help="print pyproject's version")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.current:
            print(current(args.pyproject))
            return 0
        if not args.tag:
            parser.error("give a tag (or --current)")
        version = from_tag(args.tag)
        if not args.dry_run:
            set_version(version, args.pyproject)
    except (ValueError, OSError) as exc:
        print(f"version.py: {exc}", file=sys.stderr)
        return 2
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
