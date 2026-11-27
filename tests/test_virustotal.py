"""VirusTotal scans of the release files: the script (packaging/virustotal.py, with a fake API)
and the workflows that run it after a release."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while they are made
    spec.loader.exec_module(module)
    return module


vt = _load("virustotal")


class FakeVirusTotal:
    """The API as the script uses it: answers queued per (method, path), requests recorded."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], list[tuple[int, dict | bytes]]] = {}
        self.requests: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.now = 0.0
        self.sleeps: list[float] = []

    def answer(self, method: str, path: str, *answers: tuple[int, dict | bytes]) -> None:
        self.answers.setdefault((method, path), []).extend(answers)

    def transport(self, method, url, headers, body):
        if body is not None and not isinstance(body, bytes):
            body = b"".join(body)
        self.requests.append((method, url, headers, body))
        path = url.removeprefix(vt.API)
        queue = self.answers.get((method, path))
        assert queue, f"unexpected request {method} {url}"
        status, data = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(data, Exception):
            raise data
        return status, data if isinstance(data, bytes) else json.dumps(data).encode()

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now

    def client(self, interval: float = 0.0):
        return vt.Client(
            "secret",
            interval=interval,
            transport=self.transport,
            sleep=self.sleep,
            clock=self.clock,
        )

    def paths(self) -> list[str]:
        return [f"{m} {u.removeprefix(vt.API)}" for m, u, _, _ in self.requests]


def _file(tmp_path: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


NOT_FOUND = (404, {"error": {"code": "NotFoundError", "message": "not found"}})


# --------------------------------------------------------------------------- scanning
def test_a_known_file_is_not_uploaded_again(tmp_path):
    path, digest = _file(tmp_path, "Mira-setup-1.8-cpu.exe", b"setup")
    api = FakeVirusTotal()
    known = {"data": {"attributes": {"last_analysis_stats": {"malicious": 0, "undetected": 70}}}}
    api.answer("GET", f"/files/{digest}", (200, known))
    result = vt.scan_file(api.client(), path)
    assert api.paths() == [f"GET /files/{digest}"]
    assert (result.status, result.sha256, result.size) == ("scanned", digest, 5)
    assert result.link == f"https://www.virustotal.com/gui/file/{digest}"
    assert vt.describe(result) == "0 of 70 engines flag it"
    assert api.requests[0][2]["x-apikey"] == "secret"  # the key goes in a header


def test_a_small_unknown_file_is_uploaded_as_a_form(tmp_path):
    content = b"\x00binary\r\n--not-a-boundary\xff" * 100
    path, digest = _file(tmp_path, "Mira-1.8-macos-arm64.pkg", content)
    api = FakeVirusTotal()
    api.answer("GET", f"/files/{digest}", NOT_FOUND)
    api.answer("POST", "/files", (200, {"data": {"type": "analysis", "id": "an-1"}}))
    result = vt.scan_file(api.client(), path)
    assert api.paths() == [f"GET /files/{digest}", "POST /files"]
    assert (result.status, result.analysis) == ("scanned", "an-1")
    _, _, headers, body = api.requests[1]
    boundary = headers["Content-Type"].split("boundary=", 1)[1]
    assert int(headers["Content-Length"]) == len(body)
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="Mira-1.8-macos-arm64.pkg"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    assert body == head + content + f"\r\n--{boundary}--\r\n".encode()


def test_a_big_file_goes_to_an_upload_url(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "DIRECT_LIMIT", 10)
    path, digest = _file(tmp_path, "Mira-setup-1.8-gpu.exe", b"x" * 11)
    upload = "https://www.virustotal.com/_ah/upload/abc"
    api = FakeVirusTotal()
    api.answer("GET", f"/files/{digest}", NOT_FOUND)
    api.answer("GET", "/files/upload_url", (200, {"data": upload}))
    api.answer("POST", upload, (200, {"data": {"type": "analysis", "id": "an-2"}}))
    result = vt.scan_file(api.client(), path)
    assert api.paths() == [f"GET /files/{digest}", "GET /files/upload_url", f"POST {upload}"]
    assert result.analysis == "an-2"
    assert api.requests[2][2]["x-apikey"] == "secret"


def test_a_file_over_the_limit_is_only_hashed(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "UPLOAD_LIMIT", 10)
    path, digest = _file(tmp_path, "Mira-1.8-linux-x86_64-gpu.tar.gz.part00", b"x" * 11)
    api = FakeVirusTotal()
    result = vt.scan_file(api.client(), path)
    assert api.requests == []
    assert (result.status, result.sha256) == ("too-large", digest)


def test_quota_and_server_errors_are_retried(tmp_path):
    path, digest = _file(tmp_path, "a.exe", b"a")
    api = FakeVirusTotal()
    quota = (429, {"error": {"code": "QuotaExceededError", "message": "quota"}})
    timeout = (0, TimeoutError("slow"))
    api.answer("GET", f"/files/{digest}", quota, (503, b"unavailable"), timeout, NOT_FOUND)
    api.answer("POST", "/files", (200, {"data": {"id": "an-3"}}))
    result = vt.scan_file(api.client(), path)
    assert result.status == "scanned" and result.analysis == "an-3"
    assert api.sleeps == [vt.QUOTA_PAUSE, 20.0, 40.0]  # a minute after 429, else backing off


def test_a_refused_upload_is_reported_not_raised(tmp_path):
    path, digest = _file(tmp_path, "a.exe", b"a")
    api = FakeVirusTotal()
    api.answer("GET", f"/files/{digest}", NOT_FOUND)
    api.answer("POST", "/files", (400, {"error": {"code": "BadRequestError", "message": "no"}}))
    result = vt.scan_file(api.client(), path)
    assert result.status == "failed"
    assert result.detail == "HTTP 400: BadRequestError no"
    assert len(api.requests) == 2  # a client error is not retried


def test_giving_up_after_the_retries(tmp_path):
    path, digest = _file(tmp_path, "a.exe", b"a")
    api = FakeVirusTotal()
    api.answer("GET", f"/files/{digest}", (500, b"boom"))
    result = vt.scan_file(api.client(), path)
    assert result.status == "failed" and "gave up after" in result.detail
    assert len(api.requests) == vt.RETRIES + 1


def test_requests_keep_the_interval():
    api = FakeVirusTotal()
    api.answer("GET", "/files/x", (200, {}))
    client = api.client(interval=15.0)
    for _ in range(3):
        client.request("GET", "/files/x")
    assert api.sleeps == [15.0, 15.0]  # 4 requests a minute on the public API


def test_waiting_for_the_analyses(tmp_path):
    api = FakeVirusTotal()
    queued = {"data": {"attributes": {"status": "queued"}}}
    done = {"data": {"attributes": {"status": "completed", "stats": {"malicious": 1}}}}
    api.answer("GET", "/analyses/an-1", (200, queued), (200, done))
    api.answer("GET", "/analyses/an-2", (200, queued))
    one = vt.Result("one.exe", 1, "a" * 64, analysis="an-1")
    two = vt.Result("two.exe", 1, "b" * 64, analysis="an-2")
    known = vt.Result("known.exe", 1, "c" * 64)  # nothing to wait for
    vt.wait_for_analyses(api.client(), [one, two, known], minutes=2.5, poll=60)
    assert one.stats == {"malicious": 1} and vt.describe(one) == "1 of 1 engines flag it"
    assert two.stats == {} and vt.describe(two) == "analysing"  # still queued: link stays
    assert two.status == "scanned"
    assert api.paths().count("GET /analyses/an-2") == 3  # at 0, 60 and 120 s, then gave up
    api.requests.clear()
    vt.wait_for_analyses(api.client(), [two], minutes=0)
    assert api.requests == []


# --------------------------------------------------------------------------- the notes section
def _results():
    return [
        vt.Result("Mira-setup-1.8-gpu.exe", 1, "b" * 64),
        vt.Result("Mira-1.8-linux-x86_64-gpu.tar.gz", 1, "c" * 64, status="too-large"),
        vt.Result("Mira-setup-1.8-cpu.exe", 1, "a" * 64, status="failed", detail="HTTP 400"),
    ]


def test_the_section_lists_every_file_with_its_checksum():
    text = vt.section(_results())
    assert text.startswith(vt.START + "\n### Downloads\n") and text.endswith(vt.END + "\n")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert rows == [  # sorted by name
        f"| `Mira-1.8-linux-x86_64-gpu.tar.gz` | `{'c' * 64}` | too large to scan (over 650 MB) |",
        f"| `Mira-setup-1.8-cpu.exe` | `{'a' * 64}` | not scanned |",
        f"| `Mira-setup-1.8-gpu.exe` | `{'b' * 64}` | "
        f"[scan results](https://www.virustotal.com/gui/file/{'b' * 64}) |",
    ]
    assert "sha256sum" in text and "certutil" in text


def test_merge_appends_the_section_once():
    section = vt.section(_results())
    notes = '### New\n- Search understands "or".\n'
    merged = vt.merge(notes, section)
    assert merged == notes + "\n" + section
    assert vt.merge(merged, section) == merged  # a second run replaces it
    newer = vt.section(_results()[:1])
    again = vt.merge(merged, newer)
    assert again == notes + "\n" + newer and again.count(vt.START) == 1
    assert vt.merge("", section) == section  # a release without notes
    assert vt.merge("  \r\n", section) == section


def test_merge_keeps_text_around_the_section():
    section = vt.section(_results()[:1])
    old = vt.section(_results())
    body = f"Intro written on GitHub.\r\n\r\n{old}\r\nThanks to everyone!\r\n"
    merged = vt.merge(body, section)
    assert merged.startswith("Intro written on GitHub.\r\n\r\n" + section.rstrip("\n"))
    assert merged.endswith("\r\nThanks to everyone!\r\n")
    assert merged.count(vt.START) == 1 and "c" * 64 not in merged
    # the end marker was deleted by hand: the old section ran to the end
    broken = "Intro.\n\n" + old.replace(vt.END, "")
    assert vt.merge(broken, section) == "Intro.\n\n" + section


# --------------------------------------------------------------------------- command line
def test_scan_needs_the_key_from_the_environment(tmp_path, monkeypatch):
    path, _ = _file(tmp_path, "a.exe", b"a")
    monkeypatch.delenv("VT_API_KEY", raising=False)
    assert vt.main(["scan", str(path)]) == 2


def test_scan_writes_the_section_and_fails_for_unscanned_files(tmp_path, monkeypatch):
    good, good_digest = _file(tmp_path, "good.exe", b"good")
    bad, bad_digest = _file(tmp_path, "bad.exe", b"bad")
    api = FakeVirusTotal()
    api.answer("GET", f"/files/{good_digest}", (200, {"data": {"attributes": {}}}))
    api.answer("GET", f"/files/{bad_digest}", (403, {"error": {"code": "ForbiddenError"}}))
    monkeypatch.setenv("VT_API_KEY", "secret")
    monkeypatch.setattr(vt, "urllib_transport", api.transport)
    monkeypatch.setattr(vt.time, "sleep", api.sleep)
    out = tmp_path / "section.md"
    argv = ["scan", str(good), str(bad), "--output", str(out), "--interval", "0", "--wait", "0"]
    assert vt.main(argv) == 1  # one file not scanned
    text = out.read_text(encoding="utf-8")
    assert f"`{good_digest}` | [scan results]" in text
    assert f"`{bad_digest}` | not scanned" in text


def test_merge_command(tmp_path, capsys):
    body = tmp_path / "body.md"
    body.write_text("Notes.\n", encoding="utf-8")
    section = tmp_path / "section.md"
    section.write_text(vt.section(_results()), encoding="utf-8")
    assert vt.main(["merge", str(body), str(section)]) == 0
    assert capsys.readouterr().out == vt.merge("Notes.\n", vt.section(_results()))


# --------------------------------------------------------------------------- workflows
def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_the_desktop_workflow_scans_after_attaching_the_files():
    job = _workflow("desktop.yml")["jobs"]["virustotal"]
    assert job["needs"] == "publish"
    assert job["uses"] == "./.github/workflows/virustotal.yml"
    assert job["with"] == {"tag": "${{ github.event.release.tag_name }}"}
    assert job["secrets"] == {"VIRUSTOTAL_API_KEY": "${{ secrets.VIRUSTOTAL_API_KEY }}"}
    assert job["permissions"] == {"contents": "write"}
    # releases only, once the files are attached
    assert "github.event_name == 'release'" in job["if"]
    assert "needs.publish.result == 'success'" in job["if"]


def test_the_virustotal_workflow():
    workflow = _workflow("virustotal.yml")
    triggers = workflow[True]  # YAML 1.1 reads the key `on` as True
    assert triggers["workflow_call"]["inputs"]["tag"]["required"] is True
    assert triggers["workflow_call"]["secrets"]["VIRUSTOTAL_API_KEY"]["required"] is False
    assert triggers["workflow_dispatch"]["inputs"]["tag"]["required"] is True  # a re-scan
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["scan"]
    assert job["permissions"] == {"contents": "write"}
    assert job["env"]["TAG"] == "${{ inputs.tag }}"
    steps = job["steps"]
    runs = [s.get("run", "") for s in steps]
    for run in runs:
        assert "${{" not in run  # values come in through env, not into the script
    download = next(i for i, r in enumerate(runs) if "gh release download" in r)
    scan = next(i for i, r in enumerate(runs) if "virustotal.py scan" in r)
    edit = next(i for i, r in enumerate(runs) if "gh release edit" in r)
    assert download < scan < edit < len(steps) - 1  # failures are reported after the edit
    assert "virustotal.py merge" in runs[edit] and "--notes-file" in runs[edit]
    assert steps[scan]["env"] == {"VT_API_KEY": "${{ secrets.VIRUSTOTAL_API_KEY }}"}
    assert "|| status=$?" in runs[scan]  # a failed file still gets its row in the notes
    assert "steps.scan.outputs.status != '0'" in steps[-1]["if"]
    # without the secret the job warns instead of failing
    assert "::warning" in runs[0] and "present=false" in runs[0]
    for step in steps[1:]:
        assert "steps.key.outputs.present == 'true'" in step["if"]


@pytest.mark.parametrize("doc", ["docs/development.md", "docs/installation.md"])
def test_the_docs_mention_the_scans(doc):
    assert "VirusTotal" in (ROOT / doc).read_text(encoding="utf-8")
