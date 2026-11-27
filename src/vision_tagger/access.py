"""Which files Mira may open as the file of a photo.

Photo paths come from the database, and a database can come from elsewhere (an import, see
:mod:`vision_tagger.db.transfer`). A stored path is therefore never trusted on its own: before
Mira opens the file of a photo for the web UI, a ZIP download, the location check or a portrait,
it asks :func:`photo_file`.

* Only image files, judged by the extension of the path itself (the configured extensions and
  the defaults), never by what the database claims (``images.ext``). Keys, databases,
  configuration and other files on the server cannot be read as a "photo".
* In remote mode only files below the upload folder (reference photos may also be below the
  reference folder). The server's other folders stay out of reach, as ``docs/remote.md``
  promises.
* A file on a network share (``\\\\host\\share\\...``) only if a library folder is on that share.
  Merely looking at such a path makes Windows connect to the host and sign in with the user's
  account (NTLM), so a crafted database must not be able to name any host it likes.

The check is lexical (``os.path.abspath``, the way the indexer stores paths): it never touches
the disk or the network, so ``..`` cannot escape and a refused path is not even looked at.
"""

from __future__ import annotations

import ntpath
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_EXTENSIONS

if TYPE_CHECKING:
    from .config import Settings
    from .db.session import Database

PathLike = str | os.PathLike[str]


def image_extensions(settings: Settings) -> set[str]:
    """Lower-case extensions (with the dot) of the files Mira treats as photos."""
    configured = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in settings.index.extensions
    }
    return configured | set(DEFAULT_EXTENSIONS)


def is_below(path: PathLike, folder: PathLike) -> bool:
    """``path`` is inside ``folder``, compared lexically after normalising both."""
    key = os.path.normcase(os.path.abspath(path))
    base = os.path.normcase(os.path.abspath(folder)).rstrip("\\/")
    return key.startswith(base + os.sep)


def network_share(path: PathLike) -> str | None:
    """``\\\\host\\share`` (lower case) of a path on a network share, else ``None``."""
    drive = ntpath.splitdrive(os.fspath(path))[0]
    if drive[:2] not in ("\\\\", "//"):
        return None
    return drive.replace("/", "\\").lower()


def known_shares(roots: Iterable[PathLike]) -> set[str]:
    """The network shares the library folders ``roots`` are on."""
    return {share for root in roots if (share := network_share(root)) is not None}


def photo_file(
    settings: Settings,
    path: PathLike | None,
    *,
    roots: Iterable[PathLike] = (),
    extra_roots: Iterable[PathLike] = (),
) -> Path | None:
    """The file of a photo at ``path`` if Mira may open it, else ``None`` (see the module).

    ``roots`` are the library folders (:func:`vision_tagger.indexer.stored_roots`); only
    needed for paths on network shares.
    """
    if not path:
        return None
    text = os.fspath(path)
    if "\x00" in text:
        return None
    if os.path.splitext(text)[1].lower() not in image_extensions(settings):
        return None
    folders: list[PathLike] = [*roots, *extra_roots]
    if settings.remote_mode:
        try:
            allowed: list[PathLike] = [settings.upload_dir, *extra_roots]
        except ValueError:  # remote mode without an upload folder: nothing is readable
            return None
        if not any(is_below(text, folder) for folder in allowed):
            return None
        folders += allowed
    share = network_share(text)
    if share is not None and share not in known_shares(folders):
        return None
    return Path(text)


def checker(
    settings: Settings, roots: Iterable[PathLike] = ()
) -> Callable[[PathLike | None], bool]:
    """:func:`photo_file` as a yes/no function, for code that has no settings."""
    folders = list(roots)
    return lambda path: photo_file(settings, path, roots=folders) is not None


def library_checker(db: Database, settings: Settings) -> Callable[[PathLike | None], bool]:
    """:func:`checker` with the library folders of ``db``."""
    from .indexer import stored_roots  # the indexer uses this module

    return checker(settings, stored_roots(db, settings))
