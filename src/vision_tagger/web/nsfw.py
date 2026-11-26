"""Covers for NSFW photos in the web UI (template helpers).

Wherever a thumbnail, photo or face crop of an NSFW photo appears, the page gets a black "NSFW"
tile (``/static/nsfw.svg``) instead: the ``<img>`` has no URL of the photo in ``src``, the real
one waits in ``data-nsfw-src`` and ``data-nsfw`` names the photo. A click on the cover reveals
that photo (``static/app.js``; only then does the browser ask for it, so a slow connection or a
screenshot never shows it), and it stays revealed for the browser session.

"Show NSFW photos without asking" is a cookie of the browser (:data:`COOKIE`), like the theme:
it depends on where the library is opened (at home, at work, on a phone), not on the library.

The NSFW photo ids are cached until the database changes (``db.cache``) and looked up once per
request; a page of 120 thumbnails costs one set lookup per thumbnail.
"""

from __future__ import annotations

from typing import Any

from jinja2 import pass_context
from markupsafe import Markup, escape

from .. import i18n
from ..nsfw import verdict

#: ``show``: this browser shows NSFW photos without a cover
COOKIE = "vt_nsfw"
PLACEHOLDER = "/static/nsfw.svg"


def shown(request: Any) -> bool:
    """True if this browser asked to see NSFW photos without a cover."""
    return request is not None and request.cookies.get(COOKIE) == "show"


def _lookup(request: Any, kind: str) -> Any:
    """``images`` (frozenset), ``faces`` / ``animals`` (``{id: image id}``), once per request."""
    memo = getattr(request.state, "nsfw", None)
    if memo is None:
        memo = {}
        request.state.nsfw = memo
    if kind not in memo:
        compute = {
            "images": verdict.nsfw_images,
            "faces": verdict.faces_of_nsfw_images,
            "animals": verdict.animals_of_nsfw_images,
        }[kind]
        memo[kind] = request.app.state.ctx.cache.get(("nsfw", kind), compute)
    return memo[kind]


def covered_ids(request: Any, image_ids) -> list[int]:
    """The photos among ``image_ids`` that get a cover for this request."""
    if shown(request):
        return []
    nsfw = _lookup(request, "images")
    return [i for i in image_ids if i in nsfw] if nsfw else []


def cover_for(request: Any, *, image: int | None = None, face=None, animal=None) -> int | None:
    """The photo whose cover hides this picture (``None``: no cover)."""
    if request is None or shown(request):
        return None
    if image is not None:
        return image if image in _lookup(request, "images") else None
    if face is not None:
        return _lookup(request, "faces").get(face)
    if animal is not None:
        return _lookup(request, "animals").get(animal)
    return None


def img_tag(
    src: str, cover: int | None, *, cls: str = "", alt: str = "", lazy: bool = True, title=""
) -> Markup:
    """An ``<img>``; with a cover, the placeholder and the real URL in ``data-nsfw-src``."""
    attrs: list[tuple[str, object]] = []
    if cls:
        attrs.append(("class", cls))
    if cover is None:
        if lazy:
            attrs.append(("loading", "lazy"))
        attrs += [("src", src), ("alt", alt)]
        if title:
            attrs.append(("title", title))
    else:
        attrs += [
            ("src", PLACEHOLDER),
            ("data-nsfw", cover),
            ("data-nsfw-src", src),
            ("alt", "NSFW"),
            ("title", i18n.gettext("Not safe for work – click to show")),
        ]
    return Markup("<img " + " ".join(f'{k}="{escape(v)}"' for k, v in attrs) + ">")


def _request(context) -> Any:
    return context.get("request")


@pass_context
def is_covered(context, image_id: int) -> bool:
    """Template test: does this photo get a cover on this page?"""
    return cover_for(_request(context), image=image_id) is not None


@pass_context
def nsfw_shown(context) -> bool:
    return shown(_request(context))


@pass_context
def thumb(context, image_id: int, size: int | None = None, *, cls="", alt="", lazy=True):
    """The thumbnail of a photo (``/media/thumb``), covered if it is NSFW."""
    src = f"/media/thumb/{int(image_id)}" + (f"?size={int(size)}" if size else "")
    cover = cover_for(_request(context), image=image_id)
    return img_tag(src, cover, cls=cls, alt=alt, lazy=lazy)


@pass_context
def face_img(context, face_id: int, *, cls="", alt="", lazy=True, src: str | None = None, title=""):
    """A face crop (``/media/face``, or ``src`` for another picture of the face)."""
    url = src or f"/media/face/{int(face_id)}"
    cover = cover_for(_request(context), face=face_id)
    return img_tag(url, cover, cls=cls, alt=alt, lazy=lazy, title=title)


@pass_context
def animal_img(context, animal_id: int, *, cls="", alt="", lazy=True):
    """The picture of an animal (``/media/animal``)."""
    cover = cover_for(_request(context), animal=animal_id)
    return img_tag(f"/media/animal/{int(animal_id)}", cover, cls=cls, alt=alt, lazy=lazy)


@pass_context
def safe_first(context, image_ids) -> list[int]:
    """``image_ids`` with the photos without a cover first (covers of cards and memories)."""
    covered = set(covered_ids(_request(context), image_ids))
    return sorted(image_ids, key=lambda i: i in covered) if covered else list(image_ids)


GLOBALS = {
    "is_covered": is_covered,
    "nsfw_shown": nsfw_shown,
    "thumb": thumb,
    "face_img": face_img,
    "animal_img": animal_img,
    "safe_first": safe_first,
    "nsfw_placeholder": PLACEHOLDER,
}
