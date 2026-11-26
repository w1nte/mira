"""Show a file or folder in the file manager: Explorer, Finder or the Linux desktop's.

Used by the "Show in folder" buttons of the web UI. They only make sense when the server runs
on the computer in front of the user, so the web layer offers them only in local mode and to
requests from this computer (``web.routes.reveal``). Paths are never taken from a request:
the caller resolves them from the database or the settings.

* Windows: ``explorer /select,"<file>"`` selects a file (one command line: explorer parses
  the quotes itself), ``os.startfile`` opens a folder.
* macOS: ``open -R <file>`` reveals a file in Finder, ``open <folder>`` opens a folder.
* Linux: ``xdg-open <folder>``; there is no standard way to select a file, so its folder opens.

The program is started detached and never waited for in the request. It must not load Mira's
own libraries: a frozen build gives it :func:`vision_tagger.desktop.child_environment`, a source
install drops its environment's folders from the library path (see :func:`environment`).
"""

from __future__ import annotations

import logging
import ntpath
import os
import subprocess
import sys
import sysconfig
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Launch:
    """What to start: a command (an argument list, or one command line on Windows), or a
    folder for ``os.startfile`` (Windows)."""

    command: list[str] | str | None = None
    startfile: str | None = None

    def __str__(self) -> str:
        if self.startfile is not None:
            return f"startfile {self.startfile}"
        return self.command if isinstance(self.command, str) else " ".join(self.command or [])


Launcher = Callable[[Launch], None]


class RevealError(Exception):
    """The file manager could not be started."""


def has_desktop(platform: str | None = None, environ: dict[str, str] | None = None) -> bool:
    """Whether a file manager can show up: always on Windows and macOS; on Linux only in a
    graphical session (not on a headless server reached through an SSH tunnel)."""
    platform = platform or sys.platform
    if platform in ("win32", "darwin"):
        return True
    env = os.environ if environ is None else environ
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _explorer() -> str:
    return ntpath.join(os.environ.get("SYSTEMROOT") or "C:\\Windows", "explorer.exe")


def plan(path: Path, *, select: bool, platform: str | None = None) -> Launch:
    """The command that shows ``path``: selected in its folder (``select``) or opened."""
    platform = platform or sys.platform
    target = os.fspath(path)
    if platform == "win32":
        if select:
            # explorer reads its own command line: the path must be quoted after the comma, and
            # a list (subprocess quoting) would quote "/select,<path>" as a whole
            return Launch(command=f'"{_explorer()}" /select,"{target}"')
        return Launch(startfile=target)
    if platform == "darwin":
        return Launch(command=["open", "-R", target] if select else ["open", target])
    return Launch(command=["xdg-open", os.fspath(path.parent) if select else target])


LIBRARY_PATH_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


def _python_lib_dirs() -> list[Path]:
    """``…/lib/python3.x`` of this Python: the packages' bundled libraries live below it."""
    paths = sysconfig.get_paths()
    return sorted({Path(paths[key]).resolve().parent for key in ("purelib", "platlib")})


def environment(own: list[Path] | None = None) -> dict[str, str] | None:
    """The environment for the file manager; ``None``: inherit as it is.

    A frozen build gets :func:`vision_tagger.desktop.child_environment`. Otherwise folders of
    Mira's own Python packages (``own``) are dropped from the library paths: importing OpenCV
    prepends its bundled libraries there, for child processes too, and a file manager that
    loads them may crash.
    """
    from .desktop import child_environment

    env = child_environment()
    if env is not None:
        return env
    bases = _python_lib_dirs() if own is None else own

    def foreign(entry: str) -> bool:
        path = Path(entry).resolve()
        return not any(path.is_relative_to(base) for base in bases)

    changed = dict(os.environ)
    for var in LIBRARY_PATH_VARS:
        value = changed.get(var)
        if not value:
            continue
        keep = [p for p in value.split(os.pathsep) if p and foreign(p)]
        if keep:
            changed[var] = os.pathsep.join(keep)
        else:
            del changed[var]
    return None if changed == dict(os.environ) else changed


def launch(action: Launch) -> None:
    """Start the file manager detached; raises :class:`RevealError` if it cannot start."""
    try:
        if action.startfile is not None:
            os.startfile(action.startfile)  # type: ignore[attr-defined]  # Windows only
            return
        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": environment(),
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(action.command, **kwargs)
    except (OSError, ValueError) as exc:
        raise RevealError(str(exc)) from exc
    # collect the exit status in the background (no zombie processes, no waiting here)
    threading.Thread(target=proc.wait, name="reveal-wait", daemon=True).start()


class FileManager:
    """Shows files and folders; ``launcher`` can be replaced (tests)."""

    def __init__(self, launcher: Launcher | None = None, platform: str | None = None) -> None:
        self.launcher: Launcher = launcher or launch
        self.platform = platform

    def show(self, path: Path, *, select: bool) -> Launch:
        action = plan(path, select=select, platform=self.platform)
        log.info("showing in the file manager: %s", action)
        self.launcher(action)
        return action
