"""The "Report a bug" button: a new GitHub issue with Mira's bug report form, partly filled in.

The form is ``.github/ISSUE_TEMPLATE/bug_report.yml``. A field with an ``id`` can be filled in
through the URL (``?template=bug_report.yml&version=…``); Mira fills in its version and how it
runs (:func:`vision_tagger.logs.system`: no paths, no names). Nothing is sent by itself: the
browser opens the form, and the user decides what to submit. The log is not attached (it holds
file and folder names); the form asks for it and says how to get it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from . import __version__, logs
from .updates import REPOSITORY

if TYPE_CHECKING:
    from .config import Settings

TEMPLATE = "bug_report.yml"
#: the fields of the form Mira fills in (their ids in bug_report.yml; a test checks them)
FIELDS = ("version", "environment")


def issue_url(settings: Settings | None = None) -> str:
    """The new-issue page of Mira's repository with the bug report form."""
    values = {"version": __version__, "environment": logs.system(settings)}
    query = urlencode({"template": TEMPLATE, **{key: values[key] for key in FIELDS}})
    return f"https://github.com/{REPOSITORY}/issues/new?{query}"
