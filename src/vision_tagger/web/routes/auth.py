"""Login / logout (only active when ``web.password`` is set) and the health check."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from ...i18n import _
from ..auth import COOKIE, MAX_AGE
from ..common import render

router = APIRouter()

#: seconds a wrong password waits; awaited, so no worker thread is held meanwhile
FAIL_DELAY = 1.0


def _safe_next(target: str) -> str:
    return target if target.startswith("/") and not target.startswith("//") else "/"


@router.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@router.get("/login")
def login_page(request: Request, next: str = "/") -> Response:
    if not request.app.state.auth.enabled:
        return RedirectResponse("/", status_code=303)
    return render(request, "login.html", next=_safe_next(next), error=None)


@router.post("/login")
async def login(request: Request, password: str = Form(""), next: str = Form("/")) -> Response:
    # async on purpose: locked() and check() run without a pause in between, so parallel
    # requests cannot all pass the lockout before the first wrong password is counted
    auth = request.app.state.auth
    client = request.client.host if request.client else "?"
    if auth.locked(client):
        return render(
            request,
            "login.html",
            next=_safe_next(next),
            error=_("Too many attempts. Please wait a few minutes."),
        )
    if not auth.check(password, client):
        await asyncio.sleep(FAIL_DELAY)  # slows down guessing
        return render(request, "login.html", next=_safe_next(next), error=_("Wrong password."))
    resp = RedirectResponse(_safe_next(next), status_code=303)
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    resp.set_cookie(
        COOKIE, auth.token(), max_age=MAX_AGE, httponly=True, samesite="lax", secure=secure
    )
    return resp


@router.post("/logout")
def logout(request: Request) -> Response:
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


@router.post("/logout/everywhere")
def logout_everywhere(request: Request) -> Response:
    """Ends every login, on every device (a new secret signs the cookies from now on)."""
    auth = request.app.state.auth
    if not auth.enabled:
        return RedirectResponse("/", status_code=303)
    auth.end_all(request.app.state.ctx.settings.data_dir)
    return logout(request)
