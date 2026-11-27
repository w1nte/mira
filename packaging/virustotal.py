"""VirusTotal scans and SHA-256 checksums of a release's files, for its notes.

    VT_API_KEY=... python packaging/virustotal.py scan files/* --output section.md
    python packaging/virustotal.py merge body.md section.md > notes.md

``scan`` hashes each file and has VirusTotal scan it: a file VirusTotal already knows (by its
SHA-256) is not uploaded again, files up to 32 MB go to ``POST /files``, bigger ones up to
650 MB through an upload URL, and bigger ones still are only hashed. It then waits (``--wait``
minutes) until the uploads are analysed, so the links show results, and writes a Markdown
section with a table: file, SHA-256, link to the scan results. The free (public) API allows 4
requests a minute, hence ``--interval`` seconds between requests (0 for a premium key). The API
key comes from the environment variable ``VT_API_KEY``, never from the command line. Exit code 1
when a file could not be scanned; the section lists it anyway, with its checksum.

``merge`` prints the release notes with the section: a section of an earlier run is replaced
(between the ``<!-- virustotal -->`` markers), else it is appended. Everything else stays.

The workflow .github/workflows/virustotal.yml runs both after the desktop workflow has attached
the files to a release. The script needs nothing but Python.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

API = "https://www.virustotal.com/api/v3"
RESULTS = "https://www.virustotal.com/gui/file/"
DIRECT_LIMIT = 32 * 1000 * 1000  # POST /files takes files up to 32 MB
UPLOAD_LIMIT = 650 * 1000 * 1000  # an upload URL takes files up to 650 MB
INTERVAL = 15.0  # seconds between requests: the public API allows 4 a minute
QUOTA_PAUSE = 60.0  # after "429 Too Many Requests"
RETRIES = 4  # for 429, server errors and broken connections
START = "<!-- virustotal -->"
END = "<!-- /virustotal -->"

# (method, url, headers, body) -> (status, response body); body: bytes or an iterable of bytes
Transport = Callable[[str, str, dict[str, str], bytes | Iterable[bytes] | None], tuple[int, bytes]]


class VirusTotalError(Exception):
    pass


def log(message: str) -> None:
    print(f"virustotal.py: {message}", file=sys.stderr, flush=True)


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | Iterable[bytes] | None
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Client:
    """The VirusTotal API v3: paced (``interval``), with retries for passing failures."""

    def __init__(
        self,
        key: str,
        interval: float = INTERVAL,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.key = key
        self.interval = interval
        self.transport = transport or urllib_transport
        self.sleep = sleep or time.sleep
        self.clock = clock or time.monotonic
        self._last: float | None = None

    def _pace(self) -> None:
        if self._last is not None:
            wait = self._last + self.interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def request(
        self,
        method: str,
        url: str,
        body: Callable[[], bytes | Iterable[bytes]] | None = None,
        headers: dict[str, str] | None = None,
        ok: tuple[int, ...] = (200,),
    ) -> tuple[int, dict]:
        """``(status, JSON)`` for a status in ``ok``; ``body`` makes the body (once per try)."""
        if not url.startswith("https://"):
            url = API + url
        all_headers = {"x-apikey": self.key, "accept": "application/json", **(headers or {})}
        for attempt in range(RETRIES + 1):
            self._pace()
            try:
                status, raw = self.transport(
                    method, url, all_headers, body() if body is not None else None
                )
            except (OSError, http.client.HTTPException) as exc:  # timeouts, resets, DNS
                status, raw, problem = None, b"", str(exc)
            else:
                problem = f"HTTP {status}: {_error_message(raw)}"
            if status in ok:
                try:
                    return status, json.loads(raw) if raw.strip() else {}
                except ValueError as exc:
                    raise VirusTotalError(f"HTTP {status} without JSON: {exc}") from exc
            if status is not None and status != 429 and status < 500:
                raise VirusTotalError(problem)
            if attempt == RETRIES:
                raise VirusTotalError(f"{problem} (gave up after {RETRIES + 1} tries)")
            pause = QUOTA_PAUSE if status == 429 else 10.0 * 2**attempt
            log(f"{method} {url.split('?')[0]}: {problem}; again in {pause:.0f} s")
            self.sleep(pause)
        raise AssertionError("unreachable")


def _error_message(raw: bytes) -> str:
    try:
        error = json.loads(raw)["error"]
        return f"{error.get('code', '')} {error.get('message', '')}".strip()
    except (ValueError, KeyError, TypeError, AttributeError):
        return raw[:200].decode("utf-8", "replace").strip() or "no details"


@dataclass
class Result:
    name: str
    size: int
    sha256: str
    status: str = "scanned"  # scanned (known or uploaded), too-large, failed
    detail: str = ""
    analysis: str | None = None  # the analysis of an upload
    stats: dict[str, int] = field(default_factory=dict)  # detections, once known

    @property
    def link(self) -> str:
        return RESULTS + self.sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def multipart(path: Path) -> tuple[Callable[[], Iterator[bytes]], dict[str, str]]:
    """A form with the file as field ``file``, streamed (the setups have hundreds of MB)."""
    boundary = uuid.uuid4().hex
    name = path.name.replace('"', "%22").replace("\r", "").replace("\n", "")
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    def chunks() -> Iterator[bytes]:
        yield head
        with path.open("rb") as file:
            while chunk := file.read(1 << 20):
                yield chunk
        yield tail

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(head) + path.stat().st_size + len(tail)),
    }
    return chunks, headers


def _stats(data: dict) -> dict[str, int]:
    attributes = data.get("data", {}).get("attributes", {})
    stats = attributes.get("last_analysis_stats") or attributes.get("stats") or {}
    return {k: v for k, v in stats.items() if isinstance(v, int)}


def scan_file(client: Client, path: Path) -> Result:
    result = Result(name=path.name, size=path.stat().st_size, sha256=sha256(path))
    if result.size > UPLOAD_LIMIT:
        result.status = "too-large"
        log(f"{path.name}: {result.size} bytes, over VirusTotal's limit; hashed only")
        return result
    try:
        status, data = client.request("GET", f"/files/{result.sha256}", ok=(200, 404))
        if status == 200:
            result.stats = _stats(data)
            log(f"{path.name}: known to VirusTotal, not uploaded again")
            return result
        url = "/files"
        if result.size > DIRECT_LIMIT:
            _, data = client.request("GET", "/files/upload_url")
            url = data["data"]
            if not isinstance(url, str) or not url.startswith("https://"):
                raise VirusTotalError(f"unexpected upload URL {url!r}")
        body, headers = multipart(path)
        _, data = client.request("POST", url, body=body, headers=headers)
        result.analysis = data["data"]["id"]
        log(f"{path.name}: uploaded, analysis {result.analysis}")
    except (VirusTotalError, KeyError, TypeError, ValueError) as exc:
        result.status = "failed"
        result.detail = str(exc) or type(exc).__name__
        log(f"{path.name}: not scanned: {result.detail}")
    return result


def wait_for_analyses(
    client: Client, results: list[Result], minutes: float, poll: float = 60.0
) -> None:
    """Until VirusTotal has analysed the uploads, at most ``minutes``; a file still queued
    then keeps its link (the results appear there later)."""
    pending = [r for r in results if r.analysis and r.status == "scanned"]
    if minutes <= 0:
        return
    deadline = client.clock() + minutes * 60
    while pending:
        for result in list(pending):
            try:
                _, data = client.request("GET", f"/analyses/{result.analysis}")
            except VirusTotalError as exc:
                log(f"{result.name}: analysis not readable ({exc}); its link stays")
                pending.remove(result)
                continue
            if data.get("data", {}).get("attributes", {}).get("status") == "completed":
                result.stats = _stats(data)
                pending.remove(result)
        if pending:
            if client.clock() + poll > deadline:
                names = ", ".join(r.name for r in pending)
                log(f"still analysing after {minutes:g} minutes: {names}; their links stay")
                return
            client.sleep(poll)


def describe(result: Result) -> str:
    if result.status == "too-large":
        return "too large to scan (over 650 MB)"
    if result.status == "failed":
        return f"not scanned: {result.detail}"
    if not result.stats:
        return "analysing" if result.analysis else "scanned"
    flagged = result.stats.get("malicious", 0) + result.stats.get("suspicious", 0)
    total = sum(result.stats.get(k, 0) for k in ("malicious", "suspicious", "undetected"))
    return f"{flagged} of {total} engines flag it"


def section(results: list[Result]) -> str:
    """The Markdown section for the release notes, between the markers."""
    lines = [
        START,
        "### Downloads",
        "",
        "SHA-256 checksums of this release's files and their scans by"
        " [VirusTotal](https://www.virustotal.com):",
        "",
        "| File | SHA-256 | VirusTotal |",
        "| --- | --- | --- |",
    ]
    for result in sorted(results, key=lambda r: r.name.lower()):
        name = result.name.replace("|", "\\|").replace("`", "'")
        if result.status == "scanned":
            scan = f"[scan results]({result.link})"
        elif result.status == "too-large":
            scan = "too large to scan (over 650 MB)"
        else:
            scan = "not scanned"
        lines.append(f"| `{name}` | `{result.sha256}` | {scan} |")
    lines += [
        "",
        "Check a download with `certutil -hashfile <file> SHA256` (Windows),"
        " `shasum -a 256 <file>` (macOS) or `sha256sum <file>` (Linux).",
        END,
    ]
    return "\n".join(lines) + "\n"


def merge(body: str, new: str) -> str:
    """``body`` with ``new`` in place of an earlier section, else appended."""
    new = new.strip("\n")
    start = body.find(START)
    if start >= 0:
        end = body.find(END, start)
        # an end marker deleted by hand: the section was last, it runs to the end
        rest = body[end + len(END) :] if end >= 0 else ""
        return body[:start] + new + (rest if rest.strip() else "\n")
    body = body.rstrip()
    return f"{body}\n\n{new}\n" if body else f"{new}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scan_cmd = commands.add_parser("scan", help="scan files, write the notes section")
    scan_cmd.add_argument("files", nargs="+", type=Path)
    scan_cmd.add_argument("--output", type=Path, help="the section's file (default: stdout)")
    scan_cmd.add_argument(
        "--interval", type=float, default=INTERVAL, help="seconds between API requests"
    )
    scan_cmd.add_argument(
        "--wait", type=float, default=20.0, help="minutes to wait for the analyses"
    )
    merge_cmd = commands.add_parser("merge", help="print release notes with the section")
    merge_cmd.add_argument("body", type=Path, help="the release notes as they are")
    merge_cmd.add_argument("section", type=Path, help="the section written by scan")
    args = parser.parse_args(argv)

    if args.command == "merge":
        body = args.body.read_text(encoding="utf-8")
        sys.stdout.write(merge(body, args.section.read_text(encoding="utf-8")))
        return 0

    key = os.environ.get("VT_API_KEY", "").strip()
    if not key:
        log("the environment variable VT_API_KEY (the VirusTotal API key) is not set")
        return 2
    files = [f for f in args.files if f.is_file()]
    if not files:
        log("no files to scan")
        return 2
    client = Client(key, interval=args.interval)
    results = [scan_file(client, path) for path in files]
    wait_for_analyses(client, results, args.wait)
    for result in results:
        log(f"{result.name}: {result.sha256}  {describe(result)}")
    text = section(results)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
