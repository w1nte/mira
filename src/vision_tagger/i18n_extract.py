"""Collect the translatable strings of the web UI and keep the catalogs in sync.

Usage::

    python -m vision_tagger.i18n_extract update          # sync every catalog with the sources
    python -m vision_tagger.i18n_extract check           # report problems, exit 1 if any
    python -m vision_tagger.i18n_extract check de fr -v  # some languages, every problem listed

Strings (msgids) come from

* the templates: ``_()``, ``gettext()``, ``pgettext()`` and ``ngettext()`` with literal strings;
* the Python sources: calls to ``_``, ``gettext``, ``N_``, ``Lazy`` and ``ngettext``, ``_n``
  with literal strings;
* ``i18n.JS_KEYS`` (strings of ``static/app.js``);
* the vocabulary: property keys as the ``label`` filter shows them, enum values as ``pretty``
  shows them, plus the keys and fixed values of derived and file properties.

``update`` adds new msgids to every non-English catalog with an empty translation (a list of
empty forms for plural entries), drops msgids that are no longer used, keeps the existing
translations and sorts the entries. The ``@`` keys at the top are notes for translators and
are rewritten each time: the plural forms of the language, the English plural of each plural
entry and hints for ambiguous short strings.

``check`` reports missing or empty translations, entries ``update`` would drop, placeholders
that differ from the English text (``%(name)s``, ``%(num)d``; the number of ``%%``), a lone
``%`` and characters that would break the HTML (``<``, ``>``, ``"``): template translations are
inserted as they are.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import i18n

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "web" / "templates"
CATALOG_DIR = PACKAGE_DIR / "resources" / "i18n"

SINGULAR_CALLS = {"_", "gettext", "N_", "Lazy"}
PLURAL_CALLS = {"ngettext", "_n"}

# what the forms of a plural entry are for, per language (the order of PLURALS)
PLURAL_FORMS = {
    "fr": ["one (0, 1)", "other"],
    "pl": ["one (1)", "few (2-4, 22-24, 32-34, …)", "many (0, 5-21, 25-31, …)"],
    "ru": ["one (1, 21, 31, … not 11)", "few (2-4, 22-24, … not 12-14)", "many (0, 5-20, …)"],
    "zh": ["all numbers"],
    "ja": ["all numbers"],
}
DEFAULT_PLURAL_FORMS = ["one (1)", "other"]

# hints for translators: short or ambiguous strings (written to "@context" in every catalog)
NOTES: dict[str, str] = {
    "Index": "Name of the job that scans the photo folders (shown while it runs).",
    "Recluster": "Name of the job that groups the faces again.",
    "Sidecars": "Name of the job that writes XMP sidecar files.",
    "Database import": "Name of the job that replaces the library with an imported database.",
    "Location check": "Name of the job that checks whether the photo files are still there.",
    "Clearing the library": (
        "Name of the job that deletes everything from the library database (the photo files "
        "stay), also shown as a heading above its result."
    ),
    "Removing missing photos": (
        "Name of the job that removes photos whose file is missing from the library database."
    ),
    "Danger zone": "Heading of the Database page's last card, with 'Clear the library'.",
    "Show in folder": (
        "Button on a photo's page: open Explorer (Windows) or Finder (macOS) with the photo's "
        "file selected; on Linux the file manager opens its folder."
    ),
    "Open folder": "Button next to a library folder: open it in Explorer, Finder or similar.",
    "Continue": "Button: go on to the second, last confirmation step.",
    "Back": "Button: go back to the first confirmation step.",
    "Actions": "Column heading for buttons (read by screen readers only).",
    "Runs": "Pipeline runs: each start of the pipeline, with its result (history).",
    "Database": "Sidebar link and page title: export and import of the library database.",
    "folder picker|Start": "Folder picker: label of the top level (home folder and drives).",
    "sidebar|People": (
        "Sidebar heading above Faces, Persons, Animals, Generations and Quiz; use a word that "
        "differs from the 'Persons' page name below it."
    ),
    "Up": "Folder picker: go to the parent folder.",
    "Open": "Folder picker: open the typed path.",
    "Start": "Button: start a pipeline run.",
    "all": "Placeholder of the 'limit per stage' field: no limit.",
    "any": "Filter option: no restriction (any camera, with or without faces).",
    "with": "Filter option for faces: photos with faces.",
    "without": "Filter option for faces: photos without faces.",
    "waiting": "Upload queue, followed by a number.",
    "uploading": "Upload queue, followed by a number.",
    "done": "Status word, also followed by a number in progress legends.",
    "failed": "Status word, also followed by a number in progress legends.",
    "pending": "Status of a photo in a pipeline stage, also followed by a number.",
    "errors": "Progress legend, followed by a number.",
    "Score": "Quiz scoreboard, followed by points.",
    "Streak": "Quiz scoreboard: correct answers in a row.",
    "Best": "Quiz scoreboard: the longest streak.",
    "lq": "Very short badge on a face: low quality.",
    "flat": "Export layout: all files in one folder.",
    "copy": "Export mode (copy the files).",
    "hardlink": "Export mode (hard links).",
    "symlink": "Export mode (symbolic links).",
    "slow": "Animation speed.",
    "normal": "Animation speed.",
    "fast": "Animation speed.",
    "Weihnachten mit der Familie": (
        "Search example that shows that any language works: use a language other than the "
        "UI's, e.g. English 'Christmas with the family'."
    ),
    "Pet": "Property name: the name of a pet (an animal) in the photo.",
    "Pets": "Named animals (heading in the gallery filters and on the properties page).",
    "Species": "Property name: the species of an animal in the photo (dog, cat, bird, …).",
    "bear": "Animal species (property value).",
    "%(species)s #%(id)s": "Name of an animal group that has no name yet, e.g. 'Dog #12'.",
    "Ungrouped": "Button on the Animals page: animals that are in no group, with a count.",
    "Removed": "Button on the Animals page: animals the user removed, with a count.",
    "Move to": "Button: move the selected animal photos to the group with the typed name.",
    "New group": "Button: move the selected animal photos into a new group.",
    "Not this one": "Button: take the selected photos out of this animal's group.",
    "Change": "Button: change the species of an animal group.",
    "Boxes": "Checkbox under a photo: show the boxes around faces and animals.",
    "NSFW": (
        "'Not safe for work': photos with nudity, sexual content, violence or gore. Property "
        "name; keep the abbreviation NSFW unless your language has a well-known short one."
    ),
    "Automatic": "Button: forget the user's NSFW choice for the photos; the checks decide again.",
    "First · ViT classifier": "Pipeline page: the NSFW check runs before stage 1 (faces).",
    "orientation|landscape": "Property value: picture orientation, wider than high.",
    "orientation|portrait": "Property value: picture orientation, higher than wide.",
    "orientation|square": "Property value: picture orientation.",
    "orientation|panorama": "Property value: picture orientation, very wide.",
    "landscape": "Property value: composition / genre of the photo (a landscape photo).",
    "portrait": "Property value: composition / genre of the photo (a portrait photo).",
    "Download": "Button next to an AI model: let Ollama download it.",
    "queued": "State of an AI model that waits for its download (badge).",
    "starting the download": "Phase of an AI model download, after the model name.",
    "checking the download": "Phase of an AI model download, after the model name.",
    "finishing": "Phase of an AI model download, after the model name.",
    "Downloading AI models": "Name of the job that downloads the AI models (shown while it runs).",
    "Search queries": "What an AI model is used for: understanding what the user searches for.",
    "What\u2019s new": (  # with the typographic apostrophe, like the UI text
        "Sidebar link and page title: what changed in each version of the app; also the "
        "link to the release notes of a newer version."
    ),
    "Version": "The app's version number (Environment card on the Pipeline page).",
    "Unreleased": "What's new page: heading of the changes that are in no released version yet.",
    "changelog|New": "What's new page: heading above the new features of a version.",
    "changelog|Changed": "What's new page: heading above what works differently in a version.",
    "changelog|Fixed": "What's new page: heading above the bugs fixed in a version.",
    "or": (
        "Between alternatives of a filter or search condition ('Anna or Ben'), and the small "
        "toggle in a gallery filter section that adds values as alternatives."
    ),
    "one of them": "Tooltip of a search chip with alternatives: one of them is enough.",
    "Open details": (
        "Button in the large photo preview over the gallery (lightbox): opens the photo's own "
        "page with all its details."
    ),
    "Close": "Button (an X icon) that closes the large photo preview over the gallery.",
}

PLACEHOLDER = re.compile(r"%\(\w+\)[sd]")
UNSAFE = '<>"'


@dataclass
class Message:
    msgid: str
    plural: str | None = None
    origins: set[str] = field(default_factory=set)  # "template", "python", "js", "vocabulary"


def _add(
    messages: dict[str, Message], msgid: str, plural: str | None, origin: str, where: str
) -> None:
    if not msgid:
        return
    m = messages.setdefault(msgid, Message(msgid))
    m.origins.add(origin)
    if plural is not None:
        if m.plural is not None and m.plural != plural:
            print(
                f"warning: {where}: {msgid!r} has the plurals {m.plural!r} and {plural!r}",
                file=sys.stderr,
            )
        m.plural = plural


# --------------------------------------------------------------------------- sources
def template_messages() -> Iterator[tuple[str, str | None, str]]:
    """``(msgid, plural, where)`` of the templates."""
    import jinja2

    env = jinja2.Environment(extensions=["jinja2.ext.i18n"])
    for path in sorted(TEMPLATES_DIR.rglob("*.html")):
        where = path.relative_to(PACKAGE_DIR).as_posix()
        source = path.read_text(encoding="utf-8")
        for lineno, func, message in env.extract_translations(source):
            parts = message if isinstance(message, tuple) else (message,)
            if func == "ngettext":
                if isinstance(parts[0], str) and len(parts) > 1 and isinstance(parts[1], str):
                    yield parts[0], parts[1], f"{where}:{lineno}"
            elif func == "pgettext":
                if len(parts) > 1 and isinstance(parts[0], str) and isinstance(parts[1], str):
                    yield f"{parts[0]}{i18n.CONTEXT_SEP}{parts[1]}", None, f"{where}:{lineno}"
            elif isinstance(parts[0], str):
                yield parts[0], None, f"{where}:{lineno}"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def python_messages() -> Iterator[tuple[str, str | None, str]]:
    """``(msgid, plural, where)`` of the Python sources (calls with literal strings)."""
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        where = path.relative_to(PACKAGE_DIR).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = _call_name(node)
            if name in SINGULAR_CALLS:
                msgid = _literal(node.args[0])
                if msgid is not None:
                    yield msgid, None, f"{where}:{node.lineno}"
            elif name in PLURAL_CALLS and len(node.args) >= 2:
                singular, plural = _literal(node.args[0]), _literal(node.args[1])
                if singular is not None and plural is not None:
                    yield singular, plural, f"{where}:{node.lineno}"


def vocabulary_messages() -> Iterator[str]:
    """Display texts of the property keys and fixed values (vocabulary, derived, file)."""
    from .animals import SPECIES
    from .derived import VALUES as DERIVED
    from .geo.places import PLACE_KEYS
    from .vlm.schema import Vocabulary
    from .web.common import VALUE_CONTEXTS, label_text, pretty_text

    keys: dict[str, list[str]] = {}
    for spec in Vocabulary.load().fields.values():
        if not spec.store or spec.base_type in ("integer", "text"):
            continue
        if spec.base_type in ("enum", "enum_list"):
            keys[spec.name] = list(spec.values)
        elif spec.base_type == "boolean":
            keys[spec.name] = ["yes", "no"]
        else:  # free text (lists, short text)
            keys[spec.name] = []
    for key, values in DERIVED.items():
        keys.setdefault(key, []).extend(v for v in values if v not in keys.get(key, []))
    for key in PLACE_KEYS:  # place names themselves are not translated
        keys.setdefault(key, [])
    keys.setdefault("pet", [])  # pet names are not translated, the species are
    keys.setdefault("animal", []).extend(v for v in SPECIES if v not in keys.get("animal", []))
    for key, values in keys.items():
        yield label_text(key)
        for value in values:
            text = pretty_text(value)
            yield f"{key}{i18n.CONTEXT_SEP}{text}" if key in VALUE_CONTEXTS else text


def collect() -> dict[str, Message]:
    """Every msgid of the UI."""
    messages: dict[str, Message] = {}
    for msgid, plural, where in template_messages():
        _add(messages, msgid, plural, "template", where)
    for msgid, plural, where in python_messages():
        _add(messages, msgid, plural, "python", where)
    for msgid in i18n.JS_KEYS:
        _add(messages, msgid, None, "js", "i18n.JS_KEYS")
    for msgid in i18n.MAP_JS_KEYS:
        _add(messages, msgid, None, "js", "i18n.MAP_JS_KEYS")
    for msgid in vocabulary_messages():
        _add(messages, msgid, None, "vocabulary", "vocabulary")
    return messages


# --------------------------------------------------------------------------- catalogs
def plural_count(lang: str) -> int:
    return i18n.PLURAL_COUNT.get(lang, 2)


def catalog_path(lang: str) -> Path:
    return CATALOG_DIR / f"{lang}.json"


def load(lang: str) -> dict[str, Any]:
    path = catalog_path(lang)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _sort_key(msgid: str) -> tuple[str, str]:
    return (msgid.casefold(), msgid)


def sync(lang: str, catalog: dict[str, Any], messages: dict[str, Message]) -> dict[str, Any]:
    """The catalog with exactly the given msgids; existing translations are kept."""
    n = plural_count(lang)
    generated = ("@language", "@plural_forms", "@context", "@plurals")
    out: dict[str, Any] = {
        "@language": f"{i18n.LANGUAGES.get(lang, lang)} ({lang})",
        "@plural_forms": PLURAL_FORMS.get(lang, DEFAULT_PLURAL_FORMS),
        "@context": {k: NOTES[k] for k in sorted(NOTES, key=_sort_key) if k in messages},
        "@plurals": {
            k: messages[k].plural
            for k in sorted(messages, key=_sort_key)
            if messages[k].plural is not None
        },
    }
    out.update({k: v for k, v in catalog.items() if k.startswith("@") and k not in generated})
    for msgid in sorted(messages, key=_sort_key):
        old = catalog.get(msgid)
        if messages[msgid].plural is None:
            if isinstance(old, list):
                old = old[0] if old else ""
            out[msgid] = old if isinstance(old, str) else ""
        else:
            forms = old if isinstance(old, list) else ([old] if isinstance(old, str) else [])
            forms = [f if isinstance(f, str) else "" for f in forms[:n]]
            out[msgid] = forms + [""] * (n - len(forms))
    return out


def update(
    languages: Iterable[str] | None = None, messages: dict[str, Message] | None = None
) -> list[Path]:
    """Sync the catalogs (default: every non-English one); returns the files written."""
    messages = messages if messages is not None else collect()
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for lang in languages or i18n.LANGUAGES:
        if lang == i18n.DEFAULT:
            continue
        path = catalog_path(lang)
        text = dump(sync(lang, load(lang), messages))
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            written.append(path)
    return written


# --------------------------------------------------------------------------- checks
def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def _format_problems(text: str, msgid: str, expected: set[str], optional: set[str]) -> list[str]:
    out = []
    found = placeholders(text)
    if not expected - optional <= found or not found <= expected:
        want = ", ".join(sorted(expected)) or "none"
        got = ", ".join(sorted(found)) or "none"
        out.append(f"placeholders {got} instead of {want}")
    if text.count("%%") != msgid.count("%%"):
        out.append("the number of %% differs")
    if "%" in PLACEHOLDER.sub("", text).replace("%%", ""):
        out.append("a lone % (write %% for a percent sign)")
    unsafe = [c for c in UNSAFE if c in text and c not in msgid]
    if unsafe:
        out.append(f"characters {' '.join(unsafe)} would break the HTML")
    return out


def _forms_without_num(lang: str) -> set[int]:
    """Plural forms used only for n == 1: they may say "one" instead of ``%(num)d``."""
    rule = i18n.PLURALS.get(lang, lambda k: 0 if k == 1 else 1)
    used_by_others = {rule(k) for k in range(1000) if k != 1}
    return {rule(1)} - used_by_others


def check_catalog(
    lang: str, catalog: dict[str, Any], messages: dict[str, Message]
) -> list[tuple[str, str, str]]:
    """Problems of one catalog as ``(kind, msgid, detail)``; kind: missing, obsolete, format."""
    if lang == i18n.DEFAULT:
        return []
    problems: list[tuple[str, str, str]] = []
    n = plural_count(lang)
    may_omit_num = _forms_without_num(lang)
    for msgid in sorted(messages, key=_sort_key):
        m = messages[msgid]
        value = catalog.get(msgid)
        if m.plural is None:
            if not isinstance(value, str) or not value:
                problems.append(("missing", msgid, "no translation"))
                continue
            for detail in _format_problems(value, msgid, placeholders(msgid), set()):
                problems.append(("format", msgid, detail))
            continue
        if not isinstance(value, list) or len(value) != n:
            problems.append(("missing", msgid, f"needs a list of {n} plural forms"))
            continue
        if not all(isinstance(f, str) and f for f in value):
            problems.append(("missing", msgid, "empty plural form"))
            continue
        expected = placeholders(m.plural)
        for index, form in enumerate(value):
            optional = {"%(num)d", "%(num)s"} if index in may_omit_num else set()
            for detail in _format_problems(form, m.plural, expected, optional):
                problems.append(("format", msgid, f"form {index + 1}: {detail}"))
    for key in catalog:
        if not key.startswith("@") and key not in messages:
            problems.append(("obsolete", key, "no longer used (run update)"))
    return problems


def describe(lang: str, problems: list[tuple[str, str, str]], limit: int | None = 10) -> str:
    """A readable report (``limit`` problems per kind; None = all)."""
    if not problems:
        return f"{lang}: ok"
    lines = []
    counts: dict[str, int] = {}
    for kind, _msgid, _detail in problems:
        counts[kind] = counts.get(kind, 0) + 1
    summary = ", ".join(f"{n} {kind}" for kind, n in counts.items())
    lines.append(f"{lang}: {summary}  ({catalog_path(lang).relative_to(PACKAGE_DIR.parent)})")
    shown: dict[str, int] = {}
    for kind, msgid, detail in problems:
        shown[kind] = shown.get(kind, 0) + 1
        if limit is None or shown[kind] <= limit:
            lines.append(f"  {kind}: {msgid!r}: {detail}")
        elif shown[kind] == limit + 1:
            lines.append(f"  … {counts[kind] - limit} more {kind} (use -v to list all)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vision_tagger.i18n_extract",
        description="Sync (update) or check the translation catalogs of the web UI.",
    )
    parser.add_argument("command", choices=["update", "check"])
    parser.add_argument("languages", nargs="*", help="language codes (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every problem")
    args = parser.parse_args(argv)
    unknown = [lang for lang in args.languages if lang not in i18n.LANGUAGES]
    if unknown:
        parser.error(f"unknown language(s): {', '.join(unknown)}")
    languages = [lang for lang in args.languages or i18n.LANGUAGES if lang != i18n.DEFAULT]
    if args.command == "update":
        messages = collect()
        written = update(languages, messages)
        print(f"{len(messages)} strings; {len(written)} catalog(s) written to {CATALOG_DIR}")
        return 0
    messages = collect()
    failed = False
    for lang in languages:
        problems = check_catalog(lang, load(lang), messages)
        failed = failed or bool(problems)
        print(describe(lang, problems, None if args.verbose else 10))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
