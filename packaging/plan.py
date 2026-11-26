"""Which desktop builds a run of .github/workflows/desktop.yml makes (its build matrix).

    python packaging/plan.py --event release
    python packaging/plan.py --event workflow_dispatch --platform linux --flavor gpu

Prints ``matrix=<json>`` for ``$GITHUB_OUTPUT``:

* a published release: everything;
* a manual run: the chosen platform(s) and flavour(s);
* a push to main or a pull request: the cpu build of every platform (a quick check).
"""

from __future__ import annotations

import argparse
import json
import sys

FOLDER_CLI = "dist/vision-tagger/vision-tagger-cli"
# the oldest supported Ubuntu runner: the app then runs on every distribution with glibc 2.35 or
# newer (insightface is compiled on the runner, against its glibc)
LINUX_RUNNER = "ubuntu-22.04"

#: every build: platform, flavour, runner, and the frozen command line (for the self-test)
TARGETS = [
    ("windows", "cpu", "windows-latest", FOLDER_CLI + ".exe"),
    ("windows", "gpu", "windows-latest", FOLDER_CLI + ".exe"),
    ("linux", "cpu", LINUX_RUNNER, FOLDER_CLI),
    ("linux", "gpu", LINUX_RUNNER, FOLDER_CLI),
    # Apple silicon only: no onnxruntime wheels for Intel Macs, and no CUDA on a Mac
    ("macos", "cpu", "macos-15", "dist/Mira.app/Contents/MacOS/vision-tagger-cli"),
]
PLATFORMS = ("all", "windows", "linux", "macos")
FLAVORS = ("both", "cpu", "gpu")


def plan(event: str, platform: str = "all", flavor: str = "both") -> list[dict[str, str]]:
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r} (one of {', '.join(PLATFORMS)})")
    if flavor not in FLAVORS:
        raise ValueError(f"unknown flavour {flavor!r} (one of {', '.join(FLAVORS)})")
    if event == "release":
        chosen = TARGETS
    elif event == "workflow_dispatch":
        chosen = [t for t in TARGETS if platform in ("all", t[0]) and flavor in ("both", t[1])]
    else:  # push to main, pull request
        chosen = [t for t in TARGETS if t[1] == "cpu"]
    if not chosen:
        raise ValueError(f"no build for platform {platform!r} and flavour {flavor!r}")
    return [
        {"name": f"{os_}-{fl}", "platform": os_, "flavor": fl, "runner": runner, "cli": cli}
        for os_, fl, runner, cli in chosen
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", required=True, help="github.event_name")
    parser.add_argument("--platform", default="all", help="|".join(PLATFORMS))
    parser.add_argument("--flavor", default="both", help="|".join(FLAVORS))
    args = parser.parse_args(argv)
    try:
        builds = plan(args.event, args.platform or "all", args.flavor or "both")
    except ValueError as exc:
        print(f"plan.py: {exc}", file=sys.stderr)
        return 2
    print("matrix=" + json.dumps({"include": builds}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
