"""The changelog (``resources/CHANGELOG.md``): parse it, check its format, render it.

The file is short, user-facing and English: what's new, what changed, which bugs were fixed.
Its format is strict (in the spirit of Keep a Changelog)::

    # Changelog

    Any introduction (plain text).

    ## Unreleased

    ### New
    - One short line per item.

    ### Changed
    - ...

    ### Fixed
    - ...

    ## 1.2.1 - 2026-11-26
    ...

* ``## Unreleased`` comes first, always, possibly empty. Releases follow newest first, each
  ``## <tag> - <YYYY-MM-DD>``; the tag is a release tag as the desktop workflow accepts it
  (``1.2.1``, ``v1.3``, ``1.4-rc.1``; see ``packaging/version.py``).
* The only sections are ``### New``, ``### Changed`` and ``### Fixed``, in that order, each
  optional but never empty. A release lists at least one item.
* An item is one line, ``- text``. Inline code (```code```) and links (``[text](https://…)``)
  are the only markup; nothing is ever taken as HTML.

The same module serves the "What's new" page, the tests and the release notes of the desktop
workflow (``packaging/changelog.py``, which loads this file by its path). It therefore needs
nothing but the standard library and must not import anything from the package.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as Date
from functools import lru_cache
from importlib import resources

UNRELEASED = "Unreleased"
TITLE = "# Changelog"
#: the allowed sections, in their order
KINDS = ("New", "Changed", "Fixed")
#: an item longer than this is not a "short line"
MAX_ITEM = 160

RESOURCE = "resources/CHANGELOG.md"

# a release tag (the grammar of packaging/version.py): 1.2, 1.2.3, v1.2.3, 1.2.3rc1, 1.2.3-rc.1
_VERSION = re.compile(
    r"v?(?P<release>\d+\.\d+(?:\.\d+)?)"
    r"(?:[-._]?(?P<pre>alpha|a|beta|b|rc|c)[-._]?(?P<number>\d+))?",
    re.IGNORECASE,
)
_PRE_RANK = {"alpha": 0, "a": 0, "beta": 1, "b": 1, "rc": 2, "c": 2}
_RELEASE = re.compile(r"## (?P<version>\S+) - (?P<date>\S+)")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\[\]]+)\]\(([^()\s]+)\)")
_LINK_TARGET = re.compile(r"\[[^\[\]]*\]\(([^)\s]*)")  # what follows "[text](", for checks
_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[A-Za-z/!?]")


@dataclass(frozen=True)
class Release:
    """One ``##`` block: ``Unreleased`` or a version with its date."""

    version: str  # as written: "Unreleased", "1.2.1", "v1.3"
    date: Date | None = None
    #: kind ("New", "Changed", "Fixed") -> items, in the order of KINDS; empty kinds are left out
    sections: dict[str, list[str]] = field(default_factory=dict)

    @property
    def unreleased(self) -> bool:
        return self.version == UNRELEASED

    @property
    def items(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def matches(self, version: str) -> bool:
        """``version`` names this release: ``Unreleased``, or the same version however the tag
        is written (``v1.3`` and ``1.3.0`` are ``1.3``, ``1.4-rc.1`` is ``1.4rc1``)."""
        if self.unreleased or version.strip().lower() == UNRELEASED.lower():
            return version.strip().lower() == self.version.lower()
        mine, theirs = version_key(self.version), version_key(version)
        return mine is not None and mine == theirs


def version_key(version: str) -> tuple | None:
    """A sort key of a release tag (PEP 440 order), or None if it is not one."""
    m = _VERSION.fullmatch(version.strip())
    if m is None:
        return None
    release = [int(part) for part in m["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:  # 1.3.0 is 1.3
        release.pop()
    pre = (_PRE_RANK[m["pre"].lower()], int(m["number"])) if m["pre"] else (9, 0)
    return (tuple(release), pre)


# --------------------------------------------------------------------------- parsing
def _scan(text: str) -> tuple[list[Release], list[str]]:
    """The releases and every format problem (``"line N: ..."``)."""
    problems: list[str] = []
    releases: list[Release] = []
    current: Release | None = None
    kind: str | None = None
    seen_title = False
    seen_versions: dict[tuple, str] = {}

    def problem(number: int, message: str) -> None:
        problems.append(f"line {number}: {message}")

    lines = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line:
            continue
        if not seen_title:
            seen_title = True
            if line == TITLE:
                continue
            problem(number, f"the file starts with {TITLE!r}")
        if line.startswith("## "):
            kind = None
            current = _release_heading(line, number, releases, seen_versions, problem)
            if current is None:  # a broken heading: its items are checked, not kept
                current = Release(line[3:].strip())
            else:
                releases.append(current)
            continue
        if line.startswith("### "):
            name = line[4:].strip()
            if current is None:
                problem(number, f"{line!r} comes before the first release ('## {UNRELEASED}')")
                continue
            if name not in KINDS:
                problem(number, f"unknown section {name!r}; use {', '.join(KINDS)}")
                kind = ""  # its items are checked, not kept
                continue
            last = next(reversed(current.sections), None)
            if name in current.sections:
                problem(number, f"section {name!r} appears twice in {current.version}")
            elif last is not None and KINDS.index(name) < KINDS.index(last):
                problem(number, f"section {name!r} is out of order ({', '.join(KINDS)})")
            current.sections.setdefault(name, [])
            kind = name
            continue
        if line.startswith("#"):
            problem(number, "only '## <version>' and '### <section>' headings are allowed")
            continue
        if current is None:
            continue  # the introduction
        if line.startswith(("* ", "+ ")) or raw.startswith((" ", "\t")):
            problem(number, "an item is one line starting with '- ' (no other bullets, no nesting)")
            continue
        if not line.startswith("-"):
            problem(number, "text outside an item; an item is one line starting with '- '")
            continue
        item = line[1:].strip()
        if not line.startswith("- ") or not item:
            problem(number, "empty item" if not item else "write items as '- text'")
            if not item:
                continue
        if kind is None:
            problem(number, f"item outside a section ({', '.join(KINDS)})")
            continue
        for message in item_problems(item):
            problem(number, message)
        if kind:
            current.sections[kind].append(item)

    if not seen_title:
        problems.append(f"line 1: the file starts with {TITLE!r}")
    if not releases or not releases[0].unreleased:
        problems.append(f"'## {UNRELEASED}' must be the first section, even when it is empty")
    for release in releases:
        for name, items in list(release.sections.items()):
            if not items:
                problems.append(f"{release.version}: section {name!r} has no items")
                del release.sections[name]
        if not release.unreleased and not release.items:
            problems.append(f"{release.version}: a release lists at least one item")
    return releases, problems


def _release_heading(
    line: str,
    number: int,
    releases: list[Release],
    seen_versions: dict[tuple, str],
    problem: Callable[[int, str], None],
) -> Release | None:
    """The release of a ``##`` line, or None (reported) when it is not a valid heading."""
    if line == f"## {UNRELEASED}":
        if releases:
            problem(number, f"'## {UNRELEASED}' must be the first section, and only once")
            return None
        return Release(UNRELEASED)
    m = _RELEASE.fullmatch(line)
    if m is None:
        problem(number, f"a release heading is '## <version> - <YYYY-MM-DD>', not {line!r}")
        return None
    version, written = m["version"], m["date"]
    key = version_key(version)
    if key is None:
        problem(number, f"{version!r} is not a release tag (e.g. 1.2.3, 1.3, 1.4-rc.1)")
        return None
    try:
        if not _ISO_DATE.fullmatch(written):
            raise ValueError
        day = Date.fromisoformat(written)
    except ValueError:
        problem(number, f"{written!r} is not a date (YYYY-MM-DD)")
        day = None
    if key in seen_versions:
        problem(number, f"{version} appears twice (also as {seen_versions[key]})")
    seen_versions.setdefault(key, version)
    previous = next((r for r in reversed(releases) if not r.unreleased), None)
    if previous is not None:
        above = version_key(previous.version) or key
        if key > above:
            problem(number, f"{version} is newer than {previous.version} above it (newest first)")
        elif key < above and day and previous.date and day > previous.date:
            problem(number, f"{version} is dated after {previous.version} above it")
    return Release(version, day)


def item_problems(item: str) -> list[str]:
    """What is wrong with the text of one item (markup, length)."""
    out = []
    if item.count("`") % 2:
        out.append("unbalanced ` (inline code)")
    plain = _CODE.sub("", item)
    if _HTML_TAG.search(plain):
        out.append("raw HTML is not allowed; put tags and paths in `inline code`")
    for target in _LINK_TARGET.findall(plain):
        if not _URL.match(target):
            out.append(f"links need an http(s) address, not {target!r}")
    if len(item) > MAX_ITEM:
        out.append(f"the item is {len(item)} characters long; keep it under {MAX_ITEM}")
    return out


def parse(text: str) -> list[Release]:
    """The releases of a changelog, newest first (``Unreleased`` first when present).

    Lenient: lines that break the format are skipped (see :func:`validate`)."""
    return _scan(text)[0]


def validate(text: str) -> list[str]:
    """Every problem with the format, as ``"line N: ..."`` texts; empty when it is fine."""
    return _scan(text)[1]


def find(releases: list[Release], version: str) -> Release | None:
    """The release ``version`` names (see :meth:`Release.matches`), or None."""
    return next((r for r in releases if r.matches(version)), None)


# --------------------------------------------------------------------------- the bundled file
def read() -> str:
    """The changelog shipped with the package (wheel, desktop app and source tree)."""
    return resources.files("vision_tagger").joinpath(RESOURCE).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def releases() -> list[Release]:
    """The releases of the bundled changelog (parsed once per process)."""
    return parse(read())


def section(version: str, text: str | None = None) -> Release | None:
    """The release of ``version`` (a tag or ``Unreleased``) in ``text``, else in the bundled
    changelog; None if there is none."""
    return find(parse(text) if text is not None else releases(), version)


# --------------------------------------------------------------------------- rendering
def to_markdown(release: Release) -> str:
    """The sections of a release as Markdown, e.g. for the notes of a GitHub release."""
    blocks = []
    for kind in KINDS:
        items = release.sections.get(kind)
        if items:
            blocks.append(f"### {kind}\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def inline_html(text: str) -> str:
    """One item as safe HTML: everything escaped, then ```code``` and ``[text](https://…)``
    links (http and https only, opened in a new tab). Nothing else is markup."""
    out: list[str] = []
    pos = 0
    for m in _CODE.finditer(text):
        out.append(_links_html(text[pos : m.start()]))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        pos = m.end()
    out.append(_links_html(text[pos:]))
    return "".join(out)


def _links_html(text: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _LINK.finditer(text):
        label, url = m.group(1), m.group(2)
        if not _URL.fullmatch(url):
            continue  # not a web address: stays text
        out.append(html.escape(text[pos : m.start()]))
        out.append(
            f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">'
            f"{html.escape(label)}</a>"
        )
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)
