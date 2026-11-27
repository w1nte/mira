"""Slow pages and a stuck event loop are logged with what every thread was doing."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from vision_tagger.web.app import create_app
from vision_tagger.web.watchdog import Watchdog, _short, thread_stacks

LOGGER = "vision_tagger.web.watchdog"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def waiting_for_ollama(release: threading.Event) -> None:  # shows up in the stacks
    release.wait(10)


@pytest.fixture
def busy_thread():
    release = threading.Event()
    thread = threading.Thread(target=waiting_for_ollama, args=(release,), name="test-runner")
    thread.start()
    yield thread
    release.set()
    thread.join()


def test_a_slow_page_is_reported_once_with_the_threads(caplog, busy_thread):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    clock = Clock()
    dog = Watchdog(slow=10, report_every=60, clock=clock)
    fast = dog.begin("GET", "/partials/activity")
    slow = dog.begin("GET", "/images")
    clock.now += 2
    dog.end(fast)  # answered in time: never mentioned
    clock.now += 9
    dog.check()
    first = messages(caplog)
    assert len(first) == 1
    assert first[0].startswith("slow page: GET /images (11 s) not answered yet")
    assert "What the threads are doing:" in first[0]
    assert "test-runner:" in first[0] and "in waiting_for_ollama" in first[0]
    assert "/partials/activity" not in first[0]

    clock.now += 5
    dog.check()  # reported already
    assert len(messages(caplog)) == 1
    dog.end(slow)
    assert messages(caplog)[-1] == "GET /images answered after 16 s"


def test_stacks_at_most_once_a_minute(caplog, busy_thread):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    clock = Clock()
    dog = Watchdog(slow=10, report_every=60, clock=clock)
    dog.begin("GET", "/")
    clock.now += 10
    dog.check()
    dog.begin("GET", "/overview")
    clock.now += 10
    dog.check()  # 10 s after the first report: one line, no stacks
    second = messages(caplog)[-1]
    assert second == "slow page: GET /overview (10 s) not answered yet"
    dog.begin("GET", "/properties")
    clock.now += 50
    dog.check()  # a minute after the first report: stacks again
    assert "What the threads are doing:" in messages(caplog)[-1]


def test_stacks_list_threads_that_do_the_same_once():
    release = threading.Event()
    twins = [
        threading.Thread(target=waiting_for_ollama, args=(release,), name=f"twin-{i}")
        for i in range(3)
    ]
    for t in twins:
        t.start()
    try:
        time.sleep(0.05)
        text = thread_stacks()
    finally:
        release.set()
        for t in twins:
            t.join()
    assert "3 threads (twin-" in text
    assert text.count("in waiting_for_ollama") == 1  # their stack once
    assert "in thread_stacks" not in text  # the calling thread is left out


@pytest.mark.parametrize(
    ("filename", "short"),
    [
        (
            r"C:\Users\anna\mira\.venv\Lib\site-packages\httpx\_client.py",
            "httpx/_client.py",
        ),
        (r"C:\Program Files\Mira\_internal\vision_tagger\web\app.py", "vision_tagger/web/app.py"),
        ("/home/anna/code/src/mira/src/vision_tagger/web/app.py", "vision_tagger/web/app.py"),
        ("/usr/lib/python3.12/threading.py", "python3.12/threading.py"),
        (r"C:\Users\anna\AppData\Local\Programs\Python\Lib\threading.py", "Lib/threading.py"),
        ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap>"),
    ],
)
def test_paths_name_the_module_not_the_folders_of_this_computer(filename, short):
    assert _short(filename) == short
    assert "anna" not in _short(filename)


# --------------------------------------------------------------------------- in the app
def _until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


@pytest.fixture
def quick_app(settings):
    app = create_app(settings)
    dog: Watchdog = app.state.watchdog
    dog.slow, dog.stall, dog.interval = 0.3, 0.3, 0.05
    return app


def test_a_slow_page_of_the_app_is_logged_without_its_query(quick_app, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    release = threading.Event()

    def slow_page() -> dict[str, bool]:  # a page waiting for something (sync: a worker thread)
        return {"released": release.wait(10)}

    quick_app.add_api_route("/test/slow", slow_page, methods=["GET"])
    with TestClient(quick_app) as client:
        results = []
        request = threading.Thread(
            target=lambda: results.append(client.get("/test/slow?q=Anna+at+the+beach"))
        )
        request.start()
        _until(lambda: any("slow page" in m for m in messages(caplog)))
        release.set()
        request.join()
        assert results[0].json() == {"released": True}
        assert client.get("/").status_code == 200  # fast pages are not mentioned
    found = messages(caplog)
    report = next(m for m in found if m.startswith("slow page"))
    headline = report.splitlines()[0]  # (the stacks below show code lines, not requests)
    assert "GET /test/slow (" in headline and "Anna" not in headline
    assert "in slow_page" in report  # the worker thread waiting in the page
    assert "worker threads in use: 1 of 40" in report
    assert any(m.startswith("GET /test/slow answered after") for m in found)
    assert not any("GET / " in m for m in found)


def test_a_stuck_event_loop_is_logged_with_what_blocks_it(quick_app, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)

    async def blocking_page() -> dict[str, bool]:  # blocks the event loop: nothing else loads
        time.sleep(1.0)
        return {"done": True}

    quick_app.add_api_route("/test/blocking", blocking_page, methods=["GET"])
    with TestClient(quick_app) as client:
        assert client.get("/test/blocking").json() == {"done": True}
        _until(lambda: any("runs again" in m for m in messages(caplog)))
    found = messages(caplog)
    assert any("event loop has not run" in m for m in found)
    # the first report carries the stacks (the slow request may come first)
    assert "in blocking_page" in next(m for m in found if "What the threads are doing" in m)
    assert any(m.startswith("the web server's event loop runs again after") for m in found)


def test_the_watchdog_stops_with_the_app(quick_app):
    with TestClient(quick_app):
        thread = quick_app.state.watchdog._thread
        assert thread is not None and thread.is_alive()
    assert not thread.is_alive()
