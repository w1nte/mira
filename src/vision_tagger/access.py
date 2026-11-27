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

The check is lexical (``os.path.abspath``, the way the indexer stores paths): it never touches
the disk or the network, so ``..`` cannot escape and a refused path is not even looked at.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_EXTENSIONS

if TYPE_CHECKING:
    from .config import Settings

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


def photo_file(
    settings: Settings, path: PathLike | None, *, extra_roots: Iterable[PathLike] = ()
) -> Path | None:
    """The file of a photo at ``path`` if Mira may open it, else ``None`` (see the module)."""
    if not path:
        return None
    text = os.fspath(path)
    if "\x00" in text:
        return None
    if os.path.splitext(text)[1].lower() not in image_extensions(settings):
        return None
    if settings.remote_mode:
        try:
            roots: list[PathLike] = [settings.upload_dir, *extra_roots]
        except ValueError:  # remote mode without an upload folder: nothing is readable
            return None
        if not any(is_below(text, root) for root in roots):
            return None
    return Path(text)


def checker(settings: Settings) -> Callable[[PathLike | None], bool]:
    """:func:`photo_file` as a yes/no function, for code that has no settings."""
    return lambda path: photo_file(settings, path) is not None
