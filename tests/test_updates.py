"""The update check: versions, the GitHub answer, the file for this computer, caching, the
setting, the web UI (sidebar, Home, Environment card) and ``vision-tagger update-check``.

No network: GitHub is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import vision_tagger
from vision_tagger import config as config_mod
from vision_tagger import updates
from vision_tagger.cli import app as cli_app
from vision_tagger.config import Settings
from vision_tagger.web.app import create_app

PACKAGING = Path(__file__).resolve().parent.parent / "packaging"
DE = {"Accept-Language": "de-DE,de;q=0.9"}
NOTES = "<script>alert('release notes')</script> RELEASE-NOTES-MARKER"


def release(tag, *, assets=(), prerelease=False, draft=False, body=NOTES, published=None):
    return {
        "tag_name": tag,
        "name": f"Mira {tag}",
        "html_url": f"https://github.com/w1nte/vision-tagger/releases/tag/{tag}",
        "published_at": published or "2026-11-20T10:00:00Z",
        "prerelease": prerelease,
        "draft": draft,
        "body": body,
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://github.com/w1nte/vision-tagger/releases/"
                f"download/{tag}/{name}",
            }
            for name in assets
        ],
    }


def all_assets(version, *, win_gpu_slices=True, linux_gpu_split=False):
    names = [
        f"Mira-setup-{version}-cpu.exe",
        f"Mira-setup-{version}-gpu.exe",
        f"Mira-{version}-macos-arm64.pkg",
        f"Mira-{version}-linux-x86_64-cpu.tar.gz",
    ]
    if win_gpu_slices:
        names.append(f"Mira-setup-{version}-gpu-1.bin")
    if linux_gpu_split:
        names += [f"Mira-{version}-linux-x86_64-gpu.tar.gz.part0{i}" for i in range(3)]
    else:
        names.append(f"Mira-{version}-linux-x86_64-gpu.tar.gz")
    return names


class FakeGitHub:
    """Answers like api.github.com; ``answer`` may be a JSON value, a Response or an exception."""

    def __init__(self, latest=None, releases=None):
        self.latest = latest
        self.releases = releases if releases is not None else []
        self.requests: list[httpx.Request] = []
        self.answer = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.answer, Exception):
            raise self.answer
        if isinstance(self.answer, httpx.Response):
            return self.answer
        path = request.url.path
        if path == "/repos/w1nte/vision-tagger/releases/latest":
            if self.latest is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=self.latest)
        if path == "/repos/w1nte/vision-tagger/releases":
            return httpx.Response(200, json=self.releases)
        return httpx.Response(404, json={"message": "Not Found"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub(latest=release("1.3", assets=all_assets("1.3")))
    monkeypatch.setattr(updates, "make_client", fake.client)
    return fake


def run_check(fake, current="1.2", **kwargs):
    kwargs.setdefault("platform", "win32")
    kwargs.setdefault("flavour", "cpu")
    kwargs.setdefault("machine", "AMD64")
    with fake.client() as client:
        return updates.check(current, client=client, **kwargs)


# --------------------------------------------------------------------------- versions
@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("1.2", "1.2"),
        ("v1.2.3", "1.2.3"),
        ("V1.2.3", "1.2.3"),
        ("v1.2.3-rc.1", "1.2.3rc1"),
        ("1.3rc1", "1.3rc1"),
        ("v2.0-beta.2", "2.0b2"),
        ("v2.0.0-alpha.1", "2.0.0a1"),
        ("v01.02", "1.2"),
        ("nightly", None),
        ("1", None),
        ("v1.2.3.4", None),
        ("release-1.2", None),
        ("", None),
    ],
)
def test_tags_become_versions(tag, version):
    assert updates.version_from_tag(tag) == version


def test_tags_are_read_like_the_release_build_writes_them():
    """The same rules as packaging/version.py, which gives the built app its version."""
    spec = importlib.util.spec_from_file_location("packaging_version", PACKAGING / "version.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for tag in ["1.2", "v1.2.3", "v1.2.3-rc.1", "1.2.3rc1", "v1.2-beta.2", "v3.0.0-alpha.7"]:
        assert updates.version_from_tag(tag) == module.from_tag(tag), tag
    for tag in ["nightly", "v1", "1.2.3.4"]:
        assert updates.version_from_tag(tag) is None
        with pytest.raises(ValueError):
            module.from_tag(tag)


def test_versions_compare_as_pep440():
    order = ["1.2", "1.2.1", "1.3a1", "1.3b2", "1.3rc1", "1.3", "1.10"]
    for older, newer in itertools.pairwise(order):
        assert updates.is_newer(newer, older), (newer, older)
        assert not updates.is_newer(older, newer), (older, newer)
    assert not updates.is_newer("1.2", "1.2.0")  # the same version
    assert not updates.is_newer("1.3", "1.3")
    assert not updates.is_newer("garbage", "1.2") and not updates.is_newer("1.3", "garbage")
    assert not updates.is_newer(None, "1.2")


# --------------------------------------------------------------------------- the GitHub answer
def test_latest_release_with_the_download_for_this_computer():
    fake = FakeGitHub(latest=release("v1.3", assets=all_assets("1.3")))
    result = run_check(fake)
    assert result.status == updates.AVAILABLE
    assert (result.current, result.latest_version) == ("1.2", "1.3")
    assert result.release_url == "https://github.com/w1nte/vision-tagger/releases/tag/v1.3"
    assert result.download_url.endswith("/Mira-setup-1.3-cpu.exe")
    assert result.published == "2026-11-20"
    (request,) = fake.requests
    assert request.method == "GET"
    assert str(request.url) == "https://api.github.com/repos/w1nte/vision-tagger/releases/latest"
    assert request.headers["accept"] == "application/vnd.github+json"
    assert request.headers["user-agent"] == "Mira-update-check/1.2"
    assert "authorization" not in request.headers and "cookie" not in request.headers


def test_same_or_older_release_is_up_to_date():
    fake = FakeGitHub(latest=release("1.2", assets=all_assets("1.2")))
    assert run_check(fake, current="1.2").status == updates.UP_TO_DATE
    assert run_check(fake, current="1.2.1").status == updates.UP_TO_DATE
    assert run_check(fake, current="1.3rc1").status == updates.UP_TO_DATE
    assert run_check(fake, current="1.2rc1").status == updates.AVAILABLE


def test_no_release_yet_means_up_to_date():
    assert run_check(FakeGitHub(latest=None)).status == updates.UP_TO_DATE


def test_prereleases_come_from_the_list_and_drafts_are_skipped():
    fake = FakeGitHub(
        latest=release("1.3"),
        releases=[
            release("v1.5.0", draft=True),  # a draft never counts
            release("v1.4.0-rc.2", prerelease=True, assets=all_assets("1.4.0rc2")),
            release("nightly", prerelease=True),  # not a version: ignored
            release("v1.4.0-rc.1", prerelease=True),
            release("1.3"),
            release("1.2"),
        ],
    )
    result = run_check(fake, include_prereleases=True)
    assert result.latest_version == "1.4.0rc2" and result.latest.prerelease
    assert result.download_url.endswith("/Mira-setup-1.4.0rc2-cpu.exe")
    assert fake.requests[-1].url.path == "/repos/w1nte/vision-tagger/releases"
    # without pre-releases: GitHub's latest release
    assert run_check(fake).latest_version == "1.3"
    assert fake.requests[-1].url.path.endswith("/releases/latest")


def test_a_final_release_beats_its_candidates():
    fake = FakeGitHub(releases=[release("v1.4.0-rc.1", prerelease=True), release("v1.4.0")])
    result = run_check(fake, current="1.4.0rc1", include_prereleases=True)
    assert result.latest_version == "1.4.0" and result.status == updates.AVAILABLE


def test_latest_release_that_is_a_prerelease_or_draft_or_no_version_is_ignored():
    for latest in (release("1.4rc1", prerelease=True), release("1.4", draft=True)):
        assert run_check(FakeGitHub(latest=latest)).status == updates.UNKNOWN
    assert run_check(FakeGitHub(latest=release("nightly"))).error == updates.INVALID


@pytest.mark.parametrize(
    ("platform", "machine", "flavour", "expected"),
    [
        ("win32", "AMD64", "cpu", "Mira-setup-1.3-cpu.exe"),
        ("win32", "AMD64", "gpu", None),  # the setup comes with .bin slices
        ("darwin", "arm64", "cpu", "Mira-1.3-macos-arm64.pkg"),
        ("darwin", "arm64", "gpu", "Mira-1.3-macos-arm64.pkg"),  # one build for every Mac
        ("darwin", "x86_64", "cpu", None),  # Intel Macs: no build
        ("linux", "x86_64", "cpu", "Mira-1.3-linux-x86_64-cpu.tar.gz"),
        ("linux", "x86_64", "gpu", "Mira-1.3-linux-x86_64-gpu.tar.gz"),
        ("linux", "aarch64", "cpu", None),
        ("freebsd14", "amd64", "cpu", None),
    ],
)
def test_the_download_for_each_computer(platform, machine, flavour, expected):
    fake = FakeGitHub(latest=release("1.3", assets=all_assets("1.3")))
    result = run_check(fake, platform=platform, machine=machine, flavour=flavour)
    assert result.status == updates.AVAILABLE
    if expected is None:
        assert result.download_url is None  # the release page instead
    else:
        assert result.latest.download_name == expected
        assert result.download_url.endswith(f"/download/1.3/{expected}")


def test_split_or_sliced_installers_link_to_the_release_page():
    assets = all_assets("1.3", win_gpu_slices=False, linux_gpu_split=True)
    fake = FakeGitHub(latest=release("1.3", assets=assets))
    gpu = {"flavour": "gpu"}
    # a gpu setup without slices is one file
    assert run_check(fake, **gpu).latest.download_name == "Mira-setup-1.3-gpu.exe"
    assert run_check(fake, platform="linux", machine="x86_64", **gpu).download_url is None
    cpu = run_check(fake, platform="linux", machine="x86_64", flavour="cpu")
    assert cpu.latest.download_name == "Mira-1.3-linux-x86_64-cpu.tar.gz"
    many = [*all_assets("1.3"), "Mira-setup-1.3-gpu-2.bin", "Mira-setup-1.3-gpu-3a.bin"]
    assert run_check(FakeGitHub(latest=release("1.3", assets=many)), **gpu).download_url is None


def test_files_of_another_version_or_without_https_are_not_offered():
    stale = release("1.3", assets=["Mira-setup-1.2-cpu.exe"])  # left over from 1.2
    assert run_check(FakeGitHub(latest=stale)).download_url is None
    plain = release("1.3", assets=["Mira-setup-1.3-cpu.exe"])
    plain["assets"][0]["browser_download_url"] = "http://example.com/Mira-setup-1.3-cpu.exe"
    assert run_check(FakeGitHub(latest=plain)).download_url is None
    page = release("1.3")
    page["html_url"] = "javascript:alert(1)"
    assert run_check(FakeGitHub(latest=page)).status == updates.UNKNOWN


def test_the_flavour_follows_the_installed_onnx_runtime(monkeypatch):
    installed = set()

    def distribution(name):
        if name not in installed:
            raise updates.PackageNotFoundError(name)
        return object()

    monkeypatch.setattr(updates, "distribution", distribution)
    assert updates.installed_flavour() == "cpu"
    installed.add("onnxruntime-gpu")
    assert updates.installed_flavour() == "gpu"


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (httpx.ConnectError("no route to host"), updates.NETWORK),
        (httpx.ReadTimeout("too slow"), updates.NETWORK),
        (httpx.Response(403, json={"message": "API rate limit exceeded"}), updates.RATE_LIMITED),
        (httpx.Response(429, text="slow down"), updates.RATE_LIMITED),
        (httpx.Response(500, text="oops"), updates.HTTP_ERROR),
        (httpx.Response(200, text="<html>captive portal</html>"), updates.INVALID),
        (httpx.Response(200, json=["not", "an", "object"]), updates.INVALID),
        (httpx.Response(200, json={"tag_name": 13}), updates.INVALID),
    ],
)
def test_errors_and_rate_limits_are_unknown_not_exceptions(answer, error, caplog):
    fake = FakeGitHub(latest=release("1.3"))
    fake.answer = answer
    with caplog.at_level("DEBUG", logger="vision_tagger.updates"):
        result = run_check(fake)
    assert result.status == updates.UNKNOWN and result.error == error
    assert result.latest is None and result.download_url is None
    assert all(r.levelname == "DEBUG" for r in caplog.records)  # quietly


def test_a_version_that_cannot_be_compared_is_unknown():
    result = run_check(FakeGitHub(latest=release("1.3")), current="not-a-version")
    assert result.status == updates.UNKNOWN


# --------------------------------------------------------------------------- the setting
def test_auto_is_on_only_in_the_installed_app(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert not updates.resolve("auto")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert updates.resolve("auto")
    assert updates.resolve("on") and not updates.resolve("off")
    assert Settings().updates.check == "auto" and not Settings().updates.prereleases


def test_the_setting_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("VT_UPDATES__CHECK", "off")
    monkeypatch.setenv("VT_UPDATES__PRERELEASES", "true")
    settings = Settings()
    assert settings.updates.check == "off" and settings.updates.prereleases


def checker(settings, fake, *, clock=None, current="1.2", frozen=None, **cfg):
    settings.updates = settings.updates.model_copy(update={"check": "on", **cfg})
    return updates.UpdateChecker(
        settings,
        current=current,
        frozen=frozen,
        clock=clock or time.time,
        client_factory=fake.client,
    )


def test_off_asks_nobody(settings, monkeypatch):
    fake = FakeGitHub(latest=release("1.3"))
    c = checker(settings, fake, check="off")
    assert not c.enabled and c.check(force=True) is None and c.offer() is None
    monkeypatch.delattr(sys, "frozen", raising=False)
    c = checker(settings, fake, check="auto")
    c.start()
    assert not c.enabled and c._thread is None
    assert c.check() is None and c.check(force=True) is None
    assert fake.requests == []
    assert not (settings.data_dir / updates.STATE_FILE).exists()
    # the installed app with "auto" does ask
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert checker(settings, fake, check="auto").check().status == updates.AVAILABLE
    assert len(fake.requests) == 1


# --------------------------------------------------------------------------- caching
class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_one_request_a_day_and_check_now_bypasses_it(settings):
    fake = FakeGitHub(latest=release("1.3", assets=all_assets("1.3")))
    clock = Clock()
    c = checker(settings, fake, clock=clock)
    assert c.check().status == updates.AVAILABLE
    clock.now += 23 * 3600
    assert c.check() is None  # not due
    # a restart reads the cached result: still no request
    again = checker(settings, fake, clock=clock)
    assert again.check() is None and again.offer().version == "1.3"
    assert len(fake.requests) == 1
    assert again.check(force=True).status == updates.AVAILABLE  # Check now
    assert len(fake.requests) == 2
    clock.now += 24 * 3600 + 1
    assert again.check() is not None
    assert len(fake.requests) == 3
    saved = json.loads((settings.data_dir / updates.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["release"]["version"] == "1.3" and saved["current"] == "1.2"
    assert "body" not in json.dumps(saved) and "RELEASE-NOTES-MARKER" not in json.dumps(saved)


def test_a_failed_check_keeps_the_last_release_and_retries_sooner(settings):
    fake = FakeGitHub(latest=release("1.3"))
    clock = Clock()
    c = checker(settings, fake, clock=clock)
    c.check()
    fake.answer = httpx.ConnectError("offline")
    clock.now += 25 * 3600
    assert c.check().status == updates.UNKNOWN
    assert c.offer().version == "1.3"  # still known
    assert c.status().error == updates.NETWORK
    clock.now += 3600
    assert c.check() is None
    clock.now += updates.RETRY_INTERVAL
    fake.answer = None
    assert c.check().status == updates.AVAILABLE
    assert c.status().error is None


def test_a_new_version_or_setting_checks_again_and_never_offers_the_running_one(settings):
    fake = FakeGitHub(latest=release("1.3"))
    clock = Clock()
    checker(settings, fake, clock=clock, current="1.2").check()
    # the user installed 1.3: the cached "1.3 is available" is not shown, and it checks again
    updated = checker(settings, fake, clock=clock, current="1.3")
    assert updated.offer() is None and updated.seconds_until_due() == 0
    assert updated.status().checked_at is None  # the last check was made by another version
    assert updated.check().status == updates.UP_TO_DATE
    assert updated.status().up_to_date
    # turning pre-releases on (or off) asks again as well
    assert checker(settings, fake, clock=clock, current="1.3", prereleases=True).check()


def test_prereleases_seen_earlier_are_hidden_when_they_are_turned_off(settings):
    fake = FakeGitHub(releases=[release("v1.4.0-rc.1", prerelease=True), release("1.3")])
    clock = Clock()
    checker(settings, fake, clock=clock, prereleases=True).check()
    assert checker(settings, fake, clock=clock, prereleases=True).offer().version == "1.4.0rc1"
    assert checker(settings, fake, clock=clock, prereleases=False).offer() is None


def test_a_broken_or_tampered_state_file_is_ignored(settings):
    path = settings.data_dir / updates.STATE_FILE
    fake = FakeGitHub(latest=release("1.3"))
    path.write_text("{not json", encoding="utf-8")
    assert checker(settings, fake).state == updates.State()
    tampered = {
        "checked_at": time.time(),
        "current": "1.2",
        "release": {"version": "9.0", "tag": "9.0", "url": "javascript:alert(1)"},
        "dismissed": 5,
    }
    path.write_text(json.dumps(tampered), encoding="utf-8")
    c = checker(settings, fake)
    assert c.state.release is None and c.state.dismissed is None and c.offer() is None
    tampered["release"] = {
        "version": "9.0",
        "tag": "9.0",
        "url": "https://github.com/w1nte/vision-tagger/releases/tag/9.0",
        "download_url": "file:///C:/Windows/evil.exe",
    }
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert checker(settings, fake).offer() is None


def test_a_locked_state_file_does_not_break_the_check(settings, monkeypatch):
    fake = FakeGitHub(latest=release("1.3"))
    c = checker(settings, fake)

    def locked(self, target):
        raise PermissionError("in use by another process")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(updates.time, "sleep", lambda s: None)
    assert c.check().status == updates.AVAILABLE
    assert c.offer().version == "1.3"  # kept in memory


def test_dismissing_hides_the_home_card_for_that_version_only(settings):
    fake = FakeGitHub(latest=release("1.3"))
    clock = Clock()
    c = checker(settings, fake, clock=clock)
    c.check()
    assert not c.offer().dismissed
    assert c.dismiss("1.3") and c.offer().dismissed
    assert checker(settings, fake, clock=clock).offer().dismissed  # remembered
    assert not c.dismiss("<script>") and not c.dismiss("")
    fake.latest = release("1.4")
    c.check(force=True)
    assert c.offer().version == "1.4" and not c.offer().dismissed


def test_remote_mode_links_to_the_release_page(settings):
    settings.mode = "remote"
    fake = FakeGitHub(latest=release("1.3", assets=all_assets("1.3")))
    c = checker(settings, fake)
    c.check()
    offer = c.offer()
    assert offer.download_url is None and offer.url.endswith("/releases/tag/1.3")


def test_the_background_thread_checks_soon_after_the_start_and_stops(settings, monkeypatch):
    monkeypatch.setattr(updates, "FIRST_CHECK_DELAY", 0.01)
    fake = FakeGitHub(latest=release("1.3"))
    checked = threading.Event()
    c = checker(settings, fake)
    original = c.check

    def check(**kwargs):
        try:
            return original(**kwargs)
        finally:
            checked.set()

    c.check = check
    c.start()
    c.start()  # once
    assert checked.wait(10)
    assert c.offer().version == "1.3"
    c.stop()
    c._thread.join(10)
    assert not c._thread.is_alive() and len(fake.requests) == 1


# --------------------------------------------------------------------------- web UI
@pytest.fixture
def web(settings, github, monkeypatch):
    """The web app with the check on, running version 1.2, and a fake GitHub."""
    monkeypatch.setattr(updates, "__version__", "1.2")
    monkeypatch.setattr(updates, "FIRST_CHECK_DELAY", 3600.0)  # the tests check themselves
    settings.updates = settings.updates.model_copy(update={"check": "on"})
    app = create_app(settings)
    with TestClient(app) as client:
        client.ctx = app.state.ctx
        yield client


def sidebar(page):
    return page.split('<aside class="sidebar"', 1)[1].split("</aside>", 1)[0]


def check_now(web, **headers):
    r = web.post("/updates/check", headers={"Origin": "http://testserver", **headers})
    assert r.status_code == 200, r.text
    return r


def test_nothing_is_shown_before_a_check(web, github):
    page = web.get("/").text
    assert "update-note" not in page and "update-nudge" not in page
    page = web.get("/pipeline").text
    assert "Mira 1.2" in page and "Not checked yet." in page and "Check now" in page
    assert github.requests == []  # the first check waits ~15 s after the start


def test_a_newer_release_shows_in_the_sidebar_home_and_environment(web, github):
    r = check_now(web, **{"HX-Request": "true"})
    assert 'id="update-status"' in r.text and "Update available" in r.text
    assert "Mira 1.3 is available (you have 1.2)" in r.text and "Last check:" in r.text
    page = web.get("/").text
    note = sidebar(page)
    assert "Update available: Mira 1.3" in note
    download = "https://github.com/w1nte/vision-tagger/releases/download/1.3/"
    assert f'href="{download}' in note and 'target="_blank" rel="noopener"' in note
    card = page.split('id="update-nudge"', 1)[1].split('<div hx-get="/partials/models', 1)[0]
    assert "Mira 1.3 is available (you have 1.2)" in card
    assert 'href="https://github.com/w1nte/vision-tagger/releases/tag/1.3"' in card
    assert "What\u2019s new" in card and "</svg>Download</a>" in card
    assert 'action="/updates/dismiss"' in card and 'name="version" value="1.3"' in card
    assert card.count('target="_blank" rel="noopener"') == 2
    # every page has the sidebar notice, the Home card is on Home only
    page = web.get("/images").text
    assert "Update available: Mira 1.3" in page and "update-nudge" not in page
    health = web.get("/partials/health").text
    assert "Released on 2026-11-20" in health and "Mira 1.3 is available" in health


@pytest.mark.parametrize("tag", ["1.2", "v1.2.0", "1.1", "1.2rc1"])
def test_the_same_or_an_older_release_shows_nothing(web, github, tag):
    github.latest = release(tag, assets=all_assets(updates.version_from_tag(tag)))
    check_now(web)
    page = web.get("/").text
    assert "update-note" not in page and "update-nudge" not in page
    assert "up to date" in web.get("/partials/health").text


def test_an_unreachable_github_shows_nothing_and_no_error(web, github):
    github.answer = httpx.ConnectError("offline")
    check_now(web, **{"HX-Request": "true"})
    page = web.get("/").text
    assert "update-note" not in page and "update-nudge" not in page
    assert 'class="flash err"' not in page
    health = web.get("/partials/health").text
    assert "Could not reach GitHub" in health and "Mira tries again later." in health
    r = web.post("/updates/check", headers={"Origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303 and "err=" in r.headers["location"]


def test_not_now_hides_the_home_card_for_that_version_only(web, github):
    check_now(web)
    r = web.post(
        "/updates/dismiss",
        data={"version": "1.3"},
        headers={"Origin": "http://testserver", "HX-Request": "true"},
    )
    assert r.status_code == 200 and r.text == ""  # htmx removes the card
    page = web.get("/").text
    assert "update-nudge" not in page
    assert "Update available: Mira 1.3" in sidebar(page)  # the sidebar keeps it
    # without JavaScript: a form post and back to the page
    r = web.post(
        "/updates/dismiss",
        data={"version": "1.3"},
        headers={"Origin": "http://testserver", "Referer": "http://testserver/"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/"
    # the next version shows the card again
    github.latest = release("1.4", assets=all_assets("1.4"))
    check_now(web)
    page = web.get("/").text
    assert "Mira 1.4 is available (you have 1.2)" in page


def test_update_actions_are_post_only_and_same_origin(web, github):
    assert web.get("/updates/check").status_code == 405
    assert web.get("/updates/dismiss").status_code == 405
    evil = {"Origin": "http://evil.example"}
    assert web.post("/updates/check", headers=evil).status_code == 403
    assert web.post("/updates/dismiss", data={"version": "1.3"}, headers=evil).status_code == 403
    assert github.requests == []
    check_now(web)
    web.post("/updates/dismiss", data={"version": "1.3"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert "update-nudge" in web.get("/").text  # not dismissed by another site


def test_release_notes_are_never_rendered(web, github):
    check_now(web)
    pages = [web.get(url).text for url in ("/", "/pipeline", "/partials/health", "/images")]
    pages.append(check_now(web, **{"HX-Request": "true"}).text)
    for page in pages:
        assert "RELEASE-NOTES-MARKER" not in page and "alert('release notes')" not in page


def test_no_direct_download_links_to_the_release_page(web, github, monkeypatch):
    monkeypatch.setattr(updates, "installed_flavour", lambda: "gpu")
    monkeypatch.setattr(updates.sys, "platform", "win32")
    check_now(web)
    page = web.get("/").text
    note = sidebar(page)
    assert 'href="https://github.com/w1nte/vision-tagger/releases/tag/1.3"' in note
    assert "Open the release page of Mira 1.3" in note
    card = page.split('id="update-nudge"', 1)[1]
    assert "Get the setup for this computer from the release page" in card
    assert "/releases/download/" not in page


def test_the_environment_card_says_how_to_turn_the_check_on(settings, github, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    for mode, reason in [
        ("auto", "Only the installed app looks for updates by itself."),
        ("off", "Automatic update checks are off."),
    ]:
        settings.updates = settings.updates.model_copy(update={"check": mode})
        with TestClient(create_app(settings)) as client:
            health = client.get("/partials/health").text
            assert reason in health and '<code>check = "on"</code>' in health
            assert "<code>[updates]</code>" in health and "Check now" not in health
            r = client.post("/updates/check", headers={"Origin": "http://testserver"})
            assert "Update+checks+are+off." in str(r.url)
            page = client.get("/").text
            assert "update-note" not in page
    assert github.requests == []


def test_the_update_check_in_german(web, github):
    check_now(web)
    page = web.get("/", headers=DE).text
    assert "Update verfügbar: Mira 1.3" in page
    assert "Mira 1.3 ist da (du hast 1.2)" in page and "Nicht jetzt" in page
    assert "Was ist neu" in page and "Herunterladen" in page
    health = web.get("/partials/health", headers=DE).text
    assert "Jetzt prüfen" in health and "Zuletzt geprüft:" in health
    assert "Veröffentlicht am 2026-11-20" in health
    github.answer = httpx.Response(429)
    health = check_now(web, **{"HX-Request": "true"}, **DE).text
    assert "GitHub lehnt die Anfrage gerade ab" in health


# --------------------------------------------------------------------------- command line
@pytest.fixture
def cli(tmp_path, monkeypatch, github):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setattr(vision_tagger, "__version__", "1.2")
    monkeypatch.setattr(updates, "installed_flavour", lambda: "cpu")
    monkeypatch.setattr(updates.sys, "platform", "linux")
    monkeypatch.setattr(updates.platform_module, "machine", lambda: "x86_64")
    monkeypatch.setenv("VT_PATHS__DATA_DIR", str(tmp_path / "data"))

    def run(*args):
        return CliRunner().invoke(cli_app, ["update-check", *args])

    return run


def test_cli_reports_an_update_with_exit_code_1(cli, github):
    result = cli()
    assert result.exit_code == 1, result.output
    assert "current:  Mira 1.2" in result.output
    assert "latest:   Mira 1.3 (released 2026-11-20)" in result.output
    assert "download: https://github.com/w1nte/vision-tagger/releases/download/1.3/" in (
        result.output
    )
    assert "Mira-1.3-linux-x86_64-cpu.tar.gz" in result.output


def test_cli_up_to_date_is_exit_code_0(cli, github):
    github.latest = release("1.2")
    result = cli()
    assert result.exit_code == 0, result.output
    assert "Up to date." in result.output


def test_cli_unknown_is_exit_code_2(cli, github):
    github.answer = httpx.ConnectError("offline")
    result = cli()
    assert result.exit_code == 2, result.output
    assert "latest:   unknown (could not reach GitHub)" in result.output
    github.answer = httpx.Response(403)
    assert cli().exit_code == 2


def test_cli_prereleases_and_the_setting_off(cli, github, monkeypatch):
    github.releases = [release("v1.4.0-rc.1", prerelease=True), release("1.3")]
    result = cli("--prereleases")
    assert result.exit_code == 1 and "Mira 1.4.0rc1" in result.output
    monkeypatch.setenv("VT_UPDATES__CHECK", "off")  # the command asks anyway: it was asked to
    assert cli().exit_code == 1
