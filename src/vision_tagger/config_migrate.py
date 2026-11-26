"""Migrations of ``config.toml``: users keep their config file across versions, too.

A config file carries ``config_version``. Files without it were written before config versions
existed and count as version 1. Every step below brings a file one version further; it works on
the parsed document (tomlkit), so the user's comments, order and formatting survive.

Loading a config runs the pending steps (``load_config``). The updated file replaces the old one
atomically after a backup copy was written next to it (``config.toml.bak-v1-<time>``). If the
file can't be written (read-only mount, locked by another program), the migrated settings are
used in memory and a warning asks to run ``vision-tagger migrate`` once it is writable.

How to add a migration (see CLAUDE.md):

1. write ``_vN(doc)`` that changes the document in place and returns a short note for the user,
   or ``None`` when the file needed no change for this step;
2. register it in ``STEPS`` under N and set ``CONFIG_VERSION = N``;
3. set ``config_version = N`` in ``resources/config.example.toml``;
4. add a test to ``tests/test_config_migrate.py``.

Never change a released step; fix mistakes with a new one. Steps must not import application
settings classes: like database migrations they stay frozen in time.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.items import Table
from tomlkit.toml_document import TOMLDocument

log = logging.getLogger(__name__)

#: the version of config files this app writes and understands
CONFIG_VERSION = 2
KEY = "config_version"


# --------------------------------------------------------------------------- steps
#: ``pipeline.stages`` as ``vision-tagger init`` and the desktop app copied it from the example
#: config of earlier versions (``ages`` was added automatically with ``faces`` at load time)
_OLD_DEFAULT_STAGES = (
    ("faces", "vlm"),
    ("faces", "vlm", "embed"),
    ("faces", "ages", "vlm", "embed"),
    ("faces", "ages", "animals", "vlm", "embed"),
)


def _v2(doc: TOMLDocument) -> str | None:
    """A copied default stage list no longer pins the stages.

    The example config listed ``pipeline.stages``, and new config files were copies of it, so
    every stage added later (ages, the search index, animals) stayed switched off. An untouched
    default list is removed, so the current default (all stages) applies; a list the user
    changed is kept.
    """
    pipeline = doc.get("pipeline")
    if not isinstance(pipeline, Table) or "stages" not in pipeline:
        return None
    stages = pipeline["stages"]
    if not isinstance(stages, list) or tuple(str(s) for s in stages) not in _OLD_DEFAULT_STAGES:
        return None
    del pipeline["stages"]
    pipeline.add(tomlkit.comment("stages = [...] lists the stages to run; default: all of them"))
    return "pipeline.stages: the copied default list was removed, so all stages run"


#: target version -> step from the version before
STEPS: dict[int, Callable[[TOMLDocument], str | None]] = {
    2: _v2,
}


# --------------------------------------------------------------------------- running them
@dataclass
class ConfigMigration:
    """What loading a config file did to it."""

    path: Path
    before: int
    after: int
    notes: list[str] = field(default_factory=list)
    backup: Path | None = None
    written: bool = False
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.before != self.after


def file_version(doc: TOMLDocument) -> int:
    value = doc.get(KEY, 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{KEY} must be a whole number, not {value!r}") from None


def migrate_document(doc: TOMLDocument) -> tuple[int, list[str]]:
    """Run the pending steps on ``doc`` in place. Returns the version before and the notes."""
    before = file_version(doc)
    notes: list[str] = []
    for version in range(before + 1, CONFIG_VERSION + 1):
        note = STEPS[version](doc)
        if note:
            notes.append(note)
    if before < CONFIG_VERSION:
        _set_version(doc, CONFIG_VERSION)
    return before, notes


def _set_version(doc: TOMLDocument, version: int) -> None:
    if KEY in doc:
        doc[KEY] = version
        return
    # tomlkit puts a new top-level key above the first table, after the leading comments
    item = tomlkit.item(version)
    item.comment("format version of this file; Mira updates older files itself")
    doc[KEY] = item


def _backup_path(path: Path, version: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.bak-v{version}-{stamp}")
    n = 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.name}.bak-v{version}-{stamp}-{n}")
    return candidate


def _write(path: Path, text: str, before: int) -> Path:
    backup = _backup_path(path, before)
    shutil.copy2(path, backup)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
        raise
    return backup


def load_config(path: Path, *, write: bool = True) -> tuple[dict[str, Any], ConfigMigration]:
    """The settings of ``path`` at the current config version, updating the file if needed."""
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"  # kept as the user's editor wrote them
    doc = tomlkit.parse(raw.decode("utf-8-sig"))  # Notepad may have added a BOM
    before = file_version(doc)
    result = ConfigMigration(path=path, before=before, after=before)
    if before > CONFIG_VERSION:
        log.warning(
            "%s has config version %s, newer than this app knows (%s); settings it doesn't "
            "know are ignored",
            path,
            before,
            CONFIG_VERSION,
        )
        return doc.unwrap(), result
    if before == CONFIG_VERSION:
        return doc.unwrap(), result

    _before, result.notes = migrate_document(doc)
    result.after = CONFIG_VERSION
    if write:
        new_text = tomlkit.dumps(doc).replace("\r\n", "\n")
        if newline == "\r\n":
            new_text = new_text.replace("\n", "\r\n")
        try:
            result.backup = _write(path, new_text, before)
            result.written = True
            log.info(
                "updated %s from config version %s to %s (backup: %s)%s",
                path,
                before,
                CONFIG_VERSION,
                result.backup,
                "".join(f"\n  - {note}" for note in result.notes),
            )
        except OSError as exc:
            result.error = str(exc)
            log.warning(
                "%s is at config version %s and could not be updated (%s); the updated settings "
                "are used for now. Run `vision-tagger migrate` once the file is writable.",
                path,
                before,
                exc,
            )
    return doc.unwrap(), result
