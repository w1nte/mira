"""Update check: is there a newer Mira release on GitHub?

The installed desktop app asks the GitHub API about once a day for the newest release of
``w1nte/vision-tagger`` and shows a notice when it is newer than the running version. Nothing
is downloaded or installed by itself: the user runs the new setup, which keeps the library and
migrates its database.

* :func:`check` makes one request and returns a :class:`CheckResult`. Network errors, rate
  limits and unexpected answers make the result "unknown"; it never raises for them.
* :class:`UpdateChecker` is the app's side: the setting ``updates.check`` (``auto`` is on only
  in the installed app), a background thread (about 15 s after the start, then once a day), the
  last result in ``<data>/update-check.json`` (so frequent starts don't ask GitHub more often),
  "Check now", and the version the user dismissed on the Home page.

Release tags are read like ``packaging/version.py`` writes the app version: ``1.2``,
``v1.2.3``, ``v1.2.3-rc.1`` (-> ``1.2.3rc1``). Tags that don't parse are ignored.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import platform as platform_module
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from . import APP_NAME, __version__
from .config import Settings

log = logging.getLogger(__name__)

REPOSITORY = "w1nte/vision-tagger"
API_URL = "https://api.github.com"
STATE_FILE = "update-check.json"
FIRST_CHECK_DELAY = 15.0  # seconds after the start: the app is busy starting before that
CHECK_INTERVAL = 24 * 3600.0
RETRY_INTERVAL = 2 * 3600.0  # after a failed check (offline at the start, rate limit)
TIMEOUT = 10.0

#: what :attr:`CheckResult.status` says
AVAILABLE, UP_TO_DATE, UNKNOWN = "available", "current", "unknown"
#: why a check failed (language-neutral; the web UI translates them)
NETWORK, RATE_LIMITED, HTTP_ERROR, INVALID = "network", "rate-limited", "http", "invalid"

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
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


# --------------------------------------------------------------------------- versions
def version_from_tag(tag: str) -> str | None:
    """``v1.2.3-rc.1`` -> ``1.2.3rc1`` (as packaging/version.py makes it); None: no release tag."""
    m = _TAG.fullmatch(tag.strip())
    if m is None:
        return None
    text = ".".join(str(int(part)) for part in m["release"].split("."))
    if m["pre"]:
        text += _PRE[m["pre"].lower()] + str(int(m["number"]))
    return text


def parse_version(text: str | None) -> Version | None:
    if not text:
        return None
    try:
        return Version(text)
    except InvalidVersion:
        return None


def is_newer(candidate: str | None, current: str | None) -> bool:
    """True if ``candidate`` is a later version than ``current`` (PEP 440 ordering)."""
    a, b = parse_version(candidate), parse_version(current)
    return a is not None and b is not None and a > b


# --------------------------------------------------------------------------- this computer
def installed_flavour() -> str:
    """``gpu`` when the CUDA build of ONNX Runtime is installed, else ``cpu``."""
    try:
        distribution("onnxruntime-gpu")
    except PackageNotFoundError:
        return "cpu"
    return "gpu"


def pick_download(
    assets: Iterable[dict[str, Any]], version: str, platform: str, flavour: str, machine: str
) -> tuple[str, str] | None:
    """``(url, file name)`` of the installer for this computer, if it is a single file.

    None when the release has none, or only in parts: a Windows setup too big for one file comes
    with ``.bin`` slices (the gpu setup of 1.2.1 did), and big Linux archives are split into
    ``.partNN`` files. Those are downloaded from the release page.
    """
    files = {
        a["name"]: a["browser_download_url"]
        for a in assets
        if isinstance(a, dict)
        and isinstance(a.get("name"), str)
        and _https(a.get("browser_download_url"))
    }
    v = re.escape(version)
    machine = machine.lower()
    if platform == "win32":
        wanted = rf"Mira-setup-{v}-{flavour}\.exe"
    elif platform == "darwin":
        if machine not in ("arm64", "aarch64"):
            return None  # Apple silicon only
        wanted = rf"Mira-{v}-macos-arm64\.pkg"
    elif platform.startswith("linux"):
        if machine not in ("x86_64", "amd64"):
            return None
        wanted = rf"Mira-{v}-linux-x86_64-{flavour}\.tar\.gz"
    else:
        return None
    name = next((n for n in files if re.fullmatch(wanted, n)), None)
    if name is None:
        return None
    if platform == "win32":
        slices = re.compile(re.escape(name[: -len(".exe")]) + r"-\d+[a-z]?\.bin")
        if any(slices.fullmatch(n) for n in files):
            return None
    return files[name], name


def _https(value: object) -> bool:
    return isinstance(value, str) and value.startswith("https://")


# --------------------------------------------------------------------------- the check
@dataclass(frozen=True)
class Release:
    version: str  # PEP 440, e.g. "1.3rc1"
    tag: str
    url: str  # the release page (what's new, every file)
    published: str | None = None  # YYYY-MM-DD
    prerelease: bool = False
    download_url: str | None = None  # the installer for this computer, if it is one file
    download_name: str | None = None


@dataclass(frozen=True)
class CheckResult:
    current: str
    latest: Release | None = None
    error: str | None = None  # NETWORK, RATE_LIMITED, HTTP_ERROR or INVALID

    @property
    def status(self) -> str:
        if self.error or parse_version(self.current) is None:
            return UNKNOWN
        if self.latest is not None and is_newer(self.latest.version, self.current):
            return AVAILABLE
        return UP_TO_DATE

    @property
    def latest_version(self) -> str | None:
        return self.latest.version if self.latest else None

    @property
    def release_url(self) -> str | None:
        return self.latest.url if self.latest else None

    @property
    def download_url(self) -> str | None:
        return self.latest.download_url if self.latest else None

    @property
    def published(self) -> str | None:
        return self.latest.published if self.latest else None


def headers(current: str) -> dict[str, str]:
    """No token, no cookies, nothing about the user: a generic client name and version."""
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"Mira-update-check/{current}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def make_client() -> httpx.Client:
    """The HTTP client of the checks (tests replace it with one on a ``MockTransport``)."""
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True)


def check(
    current: str,
    *,
    client: httpx.Client,
    include_prereleases: bool = False,
    api_url: str = API_URL,
    platform: str | None = None,
    flavour: str | None = None,
    machine: str | None = None,
) -> CheckResult:
    """Ask GitHub for the newest release; one request, never raises for network trouble.

    Without pre-releases this is ``GET /repos/<repo>/releases/latest`` (GitHub's latest
    release is never a draft or a pre-release); with them, the list of releases, of which the
    highest version that is not a draft wins.
    """
    base = f"{api_url.rstrip('/')}/repos/{REPOSITORY}/releases"
    url = f"{base}?per_page=30" if include_prereleases else f"{base}/latest"
    try:
        response = client.get(url, headers=headers(current), timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        log.debug("update check: %s: %s", url, exc)
        return CheckResult(current, error=NETWORK)
    if response.status_code in (403, 429):
        log.debug("update check: %s: rate limited (%s)", url, response.status_code)
        return CheckResult(current, error=RATE_LIMITED)
    if response.status_code == 404 and not include_prereleases:
        return CheckResult(current)  # no release published yet: nothing newer
    if response.status_code != 200:
        log.debug("update check: %s: HTTP %s", url, response.status_code)
        return CheckResult(current, error=HTTP_ERROR)
    try:
        data = response.json()
    except ValueError as exc:
        log.debug("update check: %s: not JSON: %s", url, exc)
        return CheckResult(current, error=INVALID)
    items = data if include_prereleases else [data]
    if not isinstance(items, list):
        log.debug("update check: %s: unexpected answer", url)
        return CheckResult(current, error=INVALID)
    best: tuple[Version, str, dict[str, Any]] | None = None
    for item in items:
        if not isinstance(item, dict) or item.get("draft"):
            continue
        if item.get("prerelease") and not include_prereleases:
            continue
        tag, page = item.get("tag_name"), item.get("html_url")
        if not isinstance(tag, str) or not _https(page):
            continue
        text = version_from_tag(tag)
        parsed = parse_version(text)
        if text is None or parsed is None:
            log.debug("update check: ignoring tag %r", tag)
            continue
        if best is None or parsed > best[0]:
            best = (parsed, text, item)
    if best is None:
        if not items:
            return CheckResult(current)
        log.debug("update check: %s: no release with a version tag", url)
        return CheckResult(current, error=INVALID)
    _, text, item = best
    assets = item.get("assets")
    download = pick_download(
        assets if isinstance(assets, list) else [],
        text,
        platform or sys.platform,
        flavour or installed_flavour(),
        machine if machine is not None else platform_module.machine(),
    )
    published = item.get("published_at")
    release = Release(
        version=text,
        tag=item["tag_name"],
        url=item["html_url"],
        published=published[:10] if isinstance(published, str) and _DATE.match(published) else None,
        prerelease=bool(item.get("prerelease")),
        download_url=download[0] if download else None,
        download_name=download[1] if download else None,
    )
    return CheckResult(current, latest=release)


# --------------------------------------------------------------------------- the app's side
def resolve(mode: str, frozen: bool | None = None) -> bool:
    """Whether the setting ``updates.check`` means "check": ``auto`` only in the installed app."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(getattr(sys, "frozen", False)) if frozen is None else frozen


@dataclass(frozen=True)
class State:
    """What ``update-check.json`` holds: the last check and the dismissed version."""

    checked_at: float | None = None  # seconds since the epoch, of the last attempt
    error: str | None = None  # of the last attempt
    current: str | None = None  # the version that made it
    prereleases: bool = False
    release: Release | None = None  # the newest release seen (kept when a later check fails)
    dismissed: str | None = None  # "Not now" on the Home page, for this version only

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: object) -> State:
        """Whatever the file holds: anything unexpected is left out (a fresh check follows)."""
        if not isinstance(data, dict):
            return cls()

        def text(key: str) -> str | None:
            value = data.get(key)
            return value if isinstance(value, str) else None

        checked = data.get("checked_at")
        if not isinstance(checked, int | float) or isinstance(checked, bool):
            checked = None
        return cls(
            checked_at=None if checked is None else float(checked),
            error=text("error"),
            current=text("current"),
            prereleases=data.get("prereleases") is True,
            release=_release_from_json(data.get("release")),
            dismissed=text("dismissed"),
        )


def _release_from_json(raw: object) -> Release | None:
    if not isinstance(raw, dict):
        return None
    values = {f.name: raw.get(f.name) for f in dataclasses.fields(Release)}

    def optional(key: str, valid: Callable[[object], bool]) -> bool:
        return values[key] is None or valid(values[key])

    if not (
        isinstance(values["version"], str)
        and parse_version(values["version"]) is not None
        and isinstance(values["tag"], str)
        and _https(values["url"])
        and optional("published", lambda v: isinstance(v, str))
        and optional("download_url", _https)
        and optional("download_name", lambda v: isinstance(v, str))
    ):
        return None
    return Release(**{**values, "prerelease": values["prerelease"] is True})


@dataclass(frozen=True)
class Offer:
    """A newer version to show (sidebar, Home, Environment card)."""

    version: str
    current: str
    url: str
    published: str | None
    download_url: str | None
    download_name: str | None
    dismissed: bool


@dataclass(frozen=True)
class Status:
    """The Environment card: the version, the setting, the last check and its result."""

    current: str
    enabled: bool
    mode: str
    checked_at: datetime | None
    error: str | None
    offer: Offer | None
    up_to_date: bool


class UpdateChecker:
    """The update check of the running app (one per ``web.context.AppContext``)."""

    def __init__(
        self,
        settings: Settings,
        *,
        current: str | None = None,
        frozen: bool | None = None,
        clock: Callable[[], float] = time.time,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        cfg = settings.updates
        self.current = current or __version__
        self.mode = cfg.check
        self.enabled = resolve(cfg.check, frozen)
        self.prereleases = cfg.prereleases
        self.api_url = cfg.api_url
        # a server's platform is not the viewer's: remote mode links to the release page
        self.direct_downloads = not settings.remote_mode
        self.path = settings.data_dir / STATE_FILE
        self.clock = clock
        self.client_factory = client_factory
        self._check_lock = threading.Lock()  # one request at a time
        self._save_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> State:
        try:
            return State.from_json(json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return State()
        except (OSError, ValueError) as exc:  # locked by a virus scanner, broken file, ...
            log.debug("update check: cannot read %s: %s", self.path, exc)
            return State()

    def _update(self, **changes: Any) -> None:
        """Change the state (readers always see a whole one) and save it; errors are logged."""
        with self._save_lock:  # a check and "Not now" at the same time keep both changes
            state = dataclasses.replace(self.state, **changes)
            self.state = state
            text = json.dumps(state.to_json(), indent=2)
            tmp = self.path.with_name(self.path.name + ".tmp")
            for attempt in range(3):
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_text(text, encoding="utf-8")
                    tmp.replace(self.path)
                    return
                except OSError as exc:  # Windows: a scanner or viewer holds the file
                    if attempt == 2:
                        log.debug("update check: cannot write %s: %s", self.path, exc)
                    else:
                        time.sleep(0.1 * (attempt + 1))

    # ------------------------------------------------------------------ checking
    def seconds_until_due(self) -> float:
        """0 when the next check is due: never checked, another version or setting, or the
        last check is a day old (two hours after a failed one)."""
        s = self.state
        now = self.clock()
        if (
            s.checked_at is None
            or s.current != self.current
            or s.prereleases != self.prereleases
            or s.checked_at > now + 300  # the clock was put back
        ):
            return 0.0
        interval = RETRY_INTERVAL if s.error else CHECK_INTERVAL
        return max(0.0, s.checked_at + interval - now)

    def check(self, *, force: bool = False) -> CheckResult | None:
        """Ask GitHub if a check is due (``force``: "Check now"). None: off, or not due."""
        if not self.enabled:
            return None
        with self._check_lock:
            if not force and self.seconds_until_due() > 0:
                return None
            client = (self.client_factory or make_client)()
            try:
                result = check(
                    self.current,
                    client=client,
                    include_prereleases=self.prereleases,
                    api_url=self.api_url,
                )
            finally:
                client.close()
            changes: dict[str, Any] = {
                "checked_at": self.clock(),
                "error": result.error,
                "current": self.current,
                "prereleases": self.prereleases,
            }
            if not result.error:  # a failed check keeps the release seen before
                changes["release"] = result.latest
            self._update(**changes)
            if result.status == AVAILABLE:
                log.info("update check: %s %s is available", APP_NAME, result.latest_version)
            return result

    def dismiss(self, version: str) -> bool:
        """The Home card's "Not now": hide it for this version only (the sidebar keeps it)."""
        if parse_version(version) is None or len(version) > 40:
            return False
        self._update(dismissed=version)
        return True

    # ------------------------------------------------------------------ what to show
    def offer(self) -> Offer | None:
        """The newer version to tell the user about, if any (cheap: from memory)."""
        if not self.enabled:
            return None
        s = self.state
        release = s.release
        if release is None or (release.prerelease and not self.prereleases):
            return None
        if not is_newer(release.version, self.current):
            return None
        direct = self.direct_downloads and release.download_url
        return Offer(
            version=release.version,
            current=self.current,
            url=release.url,
            published=release.published,
            download_url=release.download_url if direct else None,
            download_name=release.download_name if direct else None,
            dismissed=s.dismissed == release.version,
        )

    def status(self) -> Status:
        s = self.state
        fresh = s.current == self.current and s.prereleases == self.prereleases
        checked = datetime.fromtimestamp(s.checked_at) if s.checked_at and fresh else None
        offer = self.offer()
        return Status(
            current=self.current,
            enabled=self.enabled,
            mode=self.mode,
            checked_at=checked,
            error=s.error if checked else None,
            offer=offer,
            up_to_date=checked is not None and not s.error and offer is None,
        )

    # ------------------------------------------------------------------ background thread
    def start(self) -> None:
        """Check about 15 s after the start, then whenever the last check is a day old."""
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vt-update-check", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        delay = FIRST_CHECK_DELAY
        while not self._stop.wait(delay):
            try:
                self.check()
                delay = min(CHECK_INTERVAL, max(60.0, self.seconds_until_due()))
            except Exception:  # never let the thread die; try again later
                log.exception("update check failed")
                delay = RETRY_INTERVAL

    def stop(self) -> None:
        self._stop.set()
