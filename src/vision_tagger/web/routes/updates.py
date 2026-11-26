"""The update check: "Check now" (Pipeline page) and "Not now" (the Home page's card).

Both are POST (behind the Host/Origin guard of ``web.security``). Nothing is downloaded: the
links go to the release on GitHub, and the user runs the new setup.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from ...i18n import _
from ...updates import AVAILABLE, UP_TO_DATE
from ..common import back, ctx, redirect, render

router = APIRouter()


@router.post("/updates/check")
def check_now(request: Request) -> Response:
    """Ask GitHub right away, whatever the last check found (off: nothing is asked)."""
    c = ctx(request)
    result = c.updates.check(force=True)
    if request.headers.get("HX-Request"):
        return render(request, "partials/update_status.html", up=c.updates.status())
    if result is None:
        return redirect(request, back(request, "/pipeline"), err=_("Update checks are off."))
    if result.status == AVAILABLE:
        msg = _("Mira %(version)s is available.", version=result.latest_version)
    elif result.status == UP_TO_DATE:
        msg = _("Mira is up to date.")
    else:
        return redirect(request, back(request, "/pipeline"), err=_("Could not check for updates."))
    return redirect(request, back(request, "/pipeline"), msg=msg)


@router.post("/updates/dismiss")
def dismiss(request: Request, version: str = Form("")) -> Response:
    """The Home card's "Not now": no card for this version any more (the sidebar keeps it)."""
    ctx(request).updates.dismiss(version.strip())
    if request.headers.get("HX-Request"):
        return Response("")  # the card is replaced by nothing
    return redirect(request, back(request, "/"))
