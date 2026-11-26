"""Translations of the web UI.

* Strings are English in the code and templates: ``{{ _("Photos") }}``,
  ``{{ _("%(n)s photos", n=count) }}`` and ``{{ ngettext("%(num)d photo", "%(num)d photos", n) }}``.
  Templates always %-format (a literal ``%`` is written ``%%`` there); Python's ``_()`` only
  formats when variables are given.
* Catalogs live in ``resources/i18n/<lang>.json``: ``{"English text": "translation"}``; plural
  entries hold a list with one form per plural category of the language (see ``PLURALS``).
  A missing entry falls back to English. Keys starting with ``@`` are notes for translators.
  ``python -m vision_tagger.i18n_extract update`` adds new strings to every catalog,
  ``... check`` reports what is missing or broken.
* Text made outside a request (background jobs) or kept in caches shared by all requests is
  stored language-neutral: as :class:`Lazy` (translated when shown), or marked with ``N_()``
  and translated at render time.
* The language of a request is the ``vt_lang`` cookie (the switcher in the sidebar) or else the
  best match of the browser's ``Accept-Language`` header, else English. It is kept in a context
  variable, so ``_()`` works in templates and in Python code handling the request.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Callable
from functools import cache
from importlib import resources
from typing import Any

# code -> name in its own language (the order is the order of the switcher)
LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
    "ru": "Русский",
    "zh": "中文",
    "ja": "日本語",
}
DEFAULT = "en"
COOKIE = "vt_lang"


def _slavic(n: int) -> int:
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


def _polish(n: int) -> int:
    if n == 1:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


# plural form index for a count, per language (forms in the catalog are in this order)
PLURALS: dict[str, Callable[[int], int]] = {
    "fr": lambda n: 0 if n in (0, 1) else 1,
    "pl": _polish,
    "ru": _slavic,
    "zh": lambda n: 0,
    "ja": lambda n: 0,
}
PLURAL_COUNT = {"pl": 3, "ru": 3, "zh": 1, "ja": 1}  # others: 2 (one, other)

_current: contextvars.ContextVar[str] = contextvars.ContextVar("vt_lang", default=DEFAULT)


@cache
def catalog(lang: str) -> dict[str, Any]:
    if lang == DEFAULT or lang not in LANGUAGES:
        return {}
    try:
        text = (
            resources.files("vision_tagger")
            .joinpath(f"resources/i18n/{lang}.json")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError):
        return {}
    return {k: v for k, v in json.loads(text).items() if not k.startswith("@")}


def current() -> str:
    return _current.get()


def activate(lang: str) -> contextvars.Token[str]:
    return _current.set(lang if lang in LANGUAGES else DEFAULT)


def reset(token: contextvars.Token[str]) -> None:
    _current.reset(token)


def gettext(message: str) -> str:
    value = catalog(current()).get(message)
    if isinstance(value, list):
        value = value[0] if value else None
    return value or message


def ngettext(singular: str, plural: str, n: int) -> str:
    lang = current()
    forms = catalog(lang).get(singular)
    if isinstance(forms, list) and forms:
        index = PLURALS.get(lang, lambda k: 0 if k == 1 else 1)(n)
        return forms[min(index, len(forms) - 1)]
    return singular if n == 1 else plural


def _(message: str, **variables: Any) -> str:
    """Translate (and fill ``%(name)s`` placeholders) in Python code."""
    text = gettext(message)
    return text % variables if variables else text


def _n(singular: str, plural: str, n: int, **variables: Any) -> str:
    """``ngettext`` and fill the placeholders (``%(num)d`` is ``n``) in Python code."""
    return ngettext(singular, plural, n) % {"num": n, **variables}


def N_(message: str) -> str:
    """Mark a string for extraction without translating it here (it is translated later)."""
    return message


CONTEXT_SEP = "|"


def pgettext(context: str, message: str) -> str:
    """``gettext`` of a word whose translation depends on a context (catalog key
    ``"context|message"``, e.g. ``"orientation|landscape"``); else the plain translation."""
    value = catalog(current()).get(f"{context}{CONTEXT_SEP}{message}")
    if isinstance(value, list):
        value = value[0] if value else None
    return value or gettext(message)


class Lazy:
    """Text translated when it is shown (``str()``), not when it is made.

    For messages made outside a request (background jobs, which have no language) or kept in
    data shared by requests in different languages (caches): each viewer sees their language.
    Variables may be ``Lazy`` themselves.
    """

    __slots__ = ("message", "variables")

    def __init__(self, message: str, /, **variables: Any) -> None:
        self.message = message
        self.variables = variables

    def __str__(self) -> str:
        return _(self.message, **self.variables)

    def __bool__(self) -> bool:
        return bool(self.message)

    def __repr__(self) -> str:
        return f"Lazy({self.message!r}, **{self.variables!r})"


class LazyJoin(Lazy):
    """Several texts (``str`` or :class:`Lazy`) joined when shown."""

    __slots__ = ("parts", "separator")

    def __init__(self, separator: str, parts: list[Any]) -> None:
        super().__init__("")
        self.separator = separator
        self.parts = list(parts)

    def __str__(self) -> str:
        return self.separator.join(str(p) for p in self.parts)

    def __bool__(self) -> bool:
        return any(bool(p) for p in self.parts)

    def __repr__(self) -> str:
        return f"LazyJoin({self.separator!r}, {self.parts!r})"


def negotiate(accept_language: str | None, cookie: str | None = None) -> str:
    """The UI language: the switcher's cookie, else the browser's preference, else English."""
    if cookie in LANGUAGES:
        return cookie  # type: ignore[return-value]
    choices: list[tuple[float, int, str]] = []
    for position, part in enumerate((accept_language or "").split(",")):
        piece = part.strip()
        if not piece:
            continue
        tag, _sep, params = piece.partition(";")
        quality = 1.0
        for param in params.split(";"):
            name, _eq, value = param.strip().partition("=")
            if name == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        code = tag.strip().lower().replace("_", "-").split("-")[0]
        if code in LANGUAGES and quality > 0:
            choices.append((-quality, position, code))
    return min(choices)[2] if choices else DEFAULT


# strings used by static/app.js (looked up there with vtT("…"))
JS_KEYS = [
    "All time · %(n)s photos",
    "%(range)s · %(n)s photos",
    "Click the last month (Esc cancels)",
    "Drag the band or its edges",
    "Click the first month, then the last",
    "%(done)s of %(total)s files uploaded",
    "%(done)s of %(total)s files uploaded – paused",
    "Not a photo or ZIP file",
    "Connection lost – retrying…",
    "Browser storage for waiting uploads: %(used)s of %(quota)s",
    "%(n)s files skipped (not photos)",
    "Photos in this folder: %(n)s",
    "File no longer available",
    "%(n)s selected",
    "Not safe for work – click to show",
]


# strings used by static/map.js (handed to the map page as JSON)
MAP_JS_KEYS = [
    "%(n)s photos",
    "1 photo",
    "Show photos",
    "Name",
    "Save",
    "Empty: automatic name",
    "Named by you",
    "Could not load the map data.",
    "Map tiles could not be loaded – showing photos without a background map.",
]


def js_strings(keys: list[str] | None = None) -> dict[str, str]:
    """Translations the browser scripts need (current language)."""
    return {k: gettext(k) for k in (keys or JS_KEYS)}
