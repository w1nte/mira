"""Translations: language negotiation, German UI, catalogs and the extraction tool."""

from __future__ import annotations

import json
import re
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from vision_tagger import derived, i18n, i18n_extract
from vision_tagger.db.models import Cluster, ImageProperty, Person
from vision_tagger.i18n import Lazy, LazyJoin, _, _n
from vision_tagger.indexer import Indexer
from vision_tagger.pipeline.runner import PipelineRunner
from vision_tagger.pipeline.stages.ages import AgesStage
from vision_tagger.pipeline.stages.animals import AnimalsStage
from vision_tagger.pipeline.stages.embed import EmbedStage
from vision_tagger.pipeline.stages.faces import FacesStage
from vision_tagger.pipeline.stages.nsfw import NsfwStage
from vision_tagger.pipeline.stages.vlm import VlmStage
from vision_tagger.vlm.backend import FakeBackend
from vision_tagger.vlm.schema import Vocabulary
from vision_tagger.web.app import create_app
from vision_tagger.web.common import STATIC_DIR, TEMPLATES_DIR, label, pretty, templates
from vision_tagger.web.context import Job

from .conftest import make_image
from .helpers import (
    FakeAgeEstimator,
    FakeAnimalAnalyzer,
    FakeNsfwClassifier,
    add_face,
    add_image,
    person_centers,
)
from .test_faces_stage import FakeAnalyzer

DE = {"Accept-Language": "de-DE,de;q=0.9,en;q=0.5"}


@pytest.fixture
def lang():
    """Run a test body in a language: ``lang("de")``; English again afterwards."""
    tokens = []

    def activate(code: str) -> None:
        tokens.append(i18n.activate(code))

    yield activate
    for token in reversed(tokens):
        i18n.reset(token)


@pytest.fixture(scope="module")
def messages():
    return i18n_extract.collect()


@pytest.fixture
def web(settings, photos):
    """A small analysed library (fake models), like the one of test_web."""
    for i in range(6):
        make_image(photos / ("work" if i % 2 else "trip") / f"IMG_2021010{i}_120000.jpg", seed=i)
    app = create_app(settings)
    with TestClient(app) as c:
        ctx = app.state.ctx
        Indexer(ctx.db, settings).run()
        stages = {
            "faces": FacesStage(
                ctx.db, settings, analyzer=FakeAnalyzer(), age_estimator=FakeAgeEstimator()
            ),
            "ages": AgesStage(ctx.db, settings, estimator=FakeAgeEstimator()),
            "animals": AnimalsStage(ctx.db, settings, analyzer=FakeAnimalAnalyzer()),
            "nsfw": NsfwStage(ctx.db, settings, classifier=FakeNsfwClassifier()),
            "vlm": VlmStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
            "embed": EmbedStage(ctx.db, settings, backend=FakeBackend(Vocabulary.load())),
        }
        PipelineRunner(ctx.db, settings, stages.__getitem__).run()
        c.ctx = ctx
        yield c


@pytest.fixture
def home(settings, db):
    """Photos at the beach on today's date (a suggested collection and memories)."""
    today = date.today()
    with db.write() as s:
        cluster = Cluster(locked=True, person=Person(name="Anna"))
        s.add(cluster)
        s.flush()
        for n in range(1, 6):
            img = add_image(s, n, recorded_at=datetime(2015 + n, today.month, today.day, 12))
            add_face(s, img.id, person_centers(1)[0], cluster_id=cluster.id)
            s.add(ImageProperty(image_id=img.id, key="location_type", value="beach", source="vlm"))
    with TestClient(create_app(settings)) as c:
        yield c


def ok(resp):
    assert resp.status_code == 200, resp.text[:2000]
    return resp


# --------------------------------------------------------------------------- negotiation
@pytest.mark.parametrize(
    ("header", "cookie", "expected"),
    [
        ("de-DE,de;q=0.9,en;q=0.8", None, "de"),
        ("en;q=0.5, fr;q=0.9", None, "fr"),  # the highest quality wins, not the first
        ("fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7", None, "fr"),
        ("sv;q=1.0, pt-BR;q=0.8, es;q=0.7", None, "pt"),  # regional tag, unknown first choice
        ("zh_TW", None, "zh"),
        ("NL-be", None, "nl"),
        ("de;q=0.5, ru;q=0.5", None, "de"),  # same quality: the first one
        ("de;q=0, it", None, "it"),  # q=0 means "not this one"
        ("de;q=abc, ja;q=0.1", None, "ja"),  # a broken quality counts as 0
        ("sv, fi;q=0.5", None, "en"),  # nothing known -> English
        ("", None, "en"),
        (None, None, "en"),
        ("de", "fr", "fr"),  # the switcher's cookie wins
        ("de", "en", "en"),
        ("de", "xx", "de"),  # an unknown cookie is ignored
        (None, "ja", "ja"),
    ],
)
def test_negotiate(header, cookie, expected):
    assert i18n.negotiate(header, cookie) == expected


def test_activate_unknown_language_falls_back_to_english(lang):
    lang("xx")
    assert i18n.current() == "en"
    assert _("Photos") == "Photos"


# --------------------------------------------------------------------------- pages
PAGES = [
    "/",
    "/import",
    "/pipeline",
    "/collections",
    "/database",
    "/database/missing",
    "/partials/database-status",
    "/partials/status",
    "/partials/health",
    "/partials/activity",
    "/partials/import-status",
    "/faces",
    "/faces/unclustered",
    "/faces/ignored",
    "/persons",
    "/images",
    "/images?faces=yes&sort=faces&t=work_image:no&xt=year:1999&folder=trip",
    "/images/1",
    "/properties",
    "/properties/objects",
    "/properties/orientation",
    "/overview",
    "/overview?from=2021&to=2021",
    "/overview/table",
    "/search",
    "/search?q=trees",
    "/search/results?q=trees&deep=1",
    "/quiz",
    "/quiz?mode=name",
    "/generations",
]


def test_every_page_renders_in_german(web):
    """German pages render (no broken placeholders) and carry lang="de"."""
    with web.ctx.db.session() as s:
        cluster_id = s.query(Cluster.id).first()[0]
    pages = [*PAGES, f"/clusters/{cluster_id}"]
    for url in pages:
        page = ok(web.get(url, headers=DE)).text
        if not url.startswith(("/partials/", "/search/results")):
            assert '<html lang="de"' in page, url
    page = ok(web.get("/", headers=DE)).text
    assert "Deine Fotos" in page and ">Sammlungen<" in page and ">Bilder<" in page
    assert "Your photos" not in page and ">Collections<" not in page
    page = ok(web.get("/images", headers=DE)).text
    assert "Als Sammlung speichern" not in page  # only with a filter
    assert "Bilder" in page and "Neueste zuerst" in page and "Filter" in page
    page = ok(web.get("/pipeline", headers=DE)).text
    assert "Pipeline starten" in page and "Stufe 1b · AgeNet + InsightFace" in page
    assert "Gesichtsalter" in page and "Umgebung" in page


def test_people_pages_render_in_german(web):
    """Persons, ages, generations and the quiz with named people, in German."""
    from sqlalchemy import select

    from vision_tagger.db.models import Face
    from vision_tagger.faces import clustering as C

    with web.ctx.db.write() as s:
        cids = list(s.scalars(select(Cluster.id).order_by(Cluster.id)))
        C.name_cluster(s, cids[0], "Anna")
        C.name_cluster(s, cids[1], "Ben")
        anna = s.scalar(select(Person.id).where(Person.name == "Anna"))
        ben = s.scalar(select(Person.id).where(Person.name == "Ben"))
        face = s.scalar(select(Face).where(Face.cluster_id == cids[0]))
        face_id, image_id = face.id, face.image_id
    page = ok(web.get(f"/persons/{anna}", headers=DE)).text
    assert "Geburtsjahr setzen" in page and "Anna aufwachsen sehen" in page
    r = web.post(f"/persons/{anna}/birth-year", data={"birth_year": "1990"}, headers=DE)
    assert "Geburtsjahr auf 1990 gesetzt." in ok(r).text
    page = ok(web.get(f"/persons/{anna}", headers=DE)).text
    assert "Geboren <b>1990</b> · heute" in page and "Im Lauf der Jahre" in page
    assert "geb. 1990 · " in ok(web.get("/persons", headers=DE)).text
    for url in [
        f"/images/{image_id}",
        "/generations",
        f"/generations/{anna}",
        f"/generations/{anna}/compare/{ben}",
        "/quiz?mode=quiz",
        f"/images?p={anna}&p={ben}&pm=any&c={cids[0]}",
    ]:
        assert '<html lang="de"' in ok(web.get(url, headers=DE)).text, url
    answer = {"mode": "quiz", "face_id": face_id, "choice": ben, "image_id": image_id}
    page = ok(web.post("/quiz/answer", data=answer, headers=DE)).text
    assert "Leider nein" in page and "das war Anna" in page and "Das ist nicht Anna!" in page
    fix = {"mode": "quiz", "face_id": face_id, "action": "unassign"}
    assert "Von dieser Person entfernt" in ok(web.post("/quiz/fix", data=fix, headers=DE)).text


def test_english_stays_the_default(web):
    page = ok(web.get("/")).text
    assert '<html lang="en"' in page and "Your photos" in page and "Collections" in page
    page = ok(web.get("/", headers={"Accept-Language": "sv"})).text
    assert '<html lang="en"' in page


def test_cookie_switches_the_language(web):
    web.cookies.set(i18n.COOKIE, "de")
    page = ok(web.get("/faces", headers={"Accept-Language": "en"})).text
    assert '<html lang="de"' in page and "Gesichter" in page
    assert '<option value="de" selected>Deutsch</option>' in page
    web.cookies.set(i18n.COOKIE, "en")
    page = ok(web.get("/faces", headers=DE)).text
    assert '<html lang="en"' in page and ">Faces<" in page


def test_redirect_messages_use_the_language_of_the_request(web):
    r = web.post("/actions/run", data={}, headers=DE, follow_redirects=False)
    assert r.status_code == 303
    page = ok(web.get(r.headers["location"], headers=DE)).text
    assert "Wähle mindestens eine Stufe aus." in page


def test_property_names_and_values_are_translated_but_urls_stay_english(web):
    page = ok(web.get("/images?t=orientation:landscape", headers=DE)).text
    assert "Ausrichtung: Querformat" in page  # filter chip
    assert "t=orientation%3Alandscape" in page or "t=orientation:landscape" in page
    page = ok(web.get("/properties", headers=DE)).text
    assert ">Ausrichtung<" in page and 'value="orientation:landscape"' in page
    assert "Querformat" in page and "Arbeitsbild" in page


def test_job_messages_are_shown_in_each_viewers_language(web):
    """Background jobs have no language: their messages are translated when shown."""
    message = Lazy(
        "%(processed)s/%(candidates)s new or changed files analysed (%(found)s found)",
        processed=3,
        candidates=10,
        found=12,
    )
    web.ctx.jobs.current = Job(name="index", message=LazyJoin(" · ", [message, "x"]))
    english = ok(web.get("/partials/activity")).text
    german = ok(web.get("/partials/activity", headers=DE)).text
    assert "3/10 new or changed files analysed (12 found) · x" in english and "Index" in english
    assert "3/10 neue oder geänderte Dateien analysiert (12 gefunden) · x" in german
    assert ">Scan<" in german
    assert web.get("/api/status").json()["job"]["message"].startswith("3/10 new or changed")
    web.ctx.jobs.current = None


def test_cached_home_data_stays_language_neutral(home):
    """The home data is cached for every language: suggestion names are translated later."""
    german = ok(home.get("/", headers=DE)).text
    english = ok(home.get("/")).text
    assert "Am Strand" in german and "At the beach" not in german
    assert "At the beach" in english and "Am Strand" not in english
    assert "Dieses Jahr" not in german  # no photos this year
    assert "vor 1 Jahr" in german or "vor " in german  # "On this day" memories
    page = ok(home.get("/collections", headers=DE)).text
    assert 'name="name" value="Am Strand"' in page  # keeping a suggestion keeps its shown name


def test_js_strings_are_in_the_page(web, lang):
    page = ok(web.get("/images", headers=DE)).text
    data = json.loads(re.search(r'id="vt-i18n">(.*?)</script>', page).group(1))
    assert set(data) == set(i18n.JS_KEYS)
    assert data["%(n)s selected"] == "%(n)s ausgewählt"
    assert (
        data["Click the first month, then the last"]
        == "Klick den ersten Monat an, dann den letzten"
    )
    lang("de")
    assert all(text for text in i18n.js_strings().values())


def test_every_vtT_key_is_in_js_keys():
    """Strings looked up with vtT() in app.js or in templates must be in JS_KEYS."""
    sources = [(STATIC_DIR / "app.js").read_text(encoding="utf-8")]
    sources += [p.read_text(encoding="utf-8") for p in TEMPLATES_DIR.rglob("*.html")]
    used = set()
    for text in sources:
        used.update(re.findall(r"""vtT\(\s*"([^"]+)\"""", text))
        used.update(re.findall(r"""vtT\(\s*'([^']+)'""", text))
    assert used, "no vtT() calls found"
    assert used <= set(i18n.JS_KEYS), sorted(used - set(i18n.JS_KEYS))


# --------------------------------------------------------------------------- plurals, filters
def test_german_plural_forms(lang):
    lang("de")
    assert i18n.ngettext("%(num)d photo", "%(num)d photos", 1) == "%(num)d Foto"
    assert i18n.ngettext("%(num)d photo", "%(num)d photos", 0) == "%(num)d Fotos"
    assert _n("%(num)d photo", "%(num)d photos", 1) == "1 Foto"
    assert _n("%(num)d photo", "%(num)d photos", 21) == "21 Fotos"
    assert _n("%(num)d year ago", "%(num)d years ago", 3) == "vor 3 Jahren"
    template = templates.env.from_string("{{ ngettext('%(num)d image', '%(num)d images', n) }}")
    assert template.render(n=1) == "1 Bild" and template.render(n=2) == "2 Bilder"
    # the singular form may leave out the number in German (only used for n == 1)
    forms = i18n.catalog("de")[
        "Write an XMP sidecar file next to the %(num)d original photo? The photo is not "
        "changed; sidecars of other apps are kept."
    ]
    assert "%(num)d" not in forms[0] and "%(num)d" in forms[1]


def test_plural_rules_of_other_languages():
    assert [i18n.PLURALS["ru"](n) for n in (1, 2, 5, 11, 21, 22, 25)] == [0, 1, 2, 2, 0, 1, 2]
    assert [i18n.PLURALS["pl"](n) for n in (1, 2, 5, 12, 22, 25)] == [0, 1, 2, 2, 1, 2]
    assert [i18n.PLURALS["fr"](n) for n in (0, 1, 2)] == [0, 0, 1]
    assert i18n_extract._forms_without_num("de") == {0}
    assert i18n_extract._forms_without_num("fr") == set()  # form 1 is also used for 0
    assert i18n_extract._forms_without_num("ru") == set()  # form 1 is also used for 21
    assert i18n_extract._forms_without_num("pl") == {0}
    assert i18n_extract._forms_without_num("ja") == set()


def test_label_and_pretty_translate_the_vocabulary(lang):
    assert label("location_type") == "Location type" and pretty("meeting_room") == "meeting room"
    lang("de")
    assert label("location_type") == "Ortstyp"
    assert label("work_image") == "Arbeitsbild"  # derived
    assert label("capture_device") == "Aufnahmegerät"  # file property
    assert pretty("meeting_room") == "Besprechungsraum"
    assert pretty("meeting_room", "location_type") == "Besprechungsraum"
    assert pretty("yes", "work_image") == "ja" and pretty("maybe") == "vielleicht"
    # the same English word, two meanings
    assert pretty("landscape", "orientation") == "Querformat"
    assert pretty("landscape", "composition") == "Landschaft"
    assert pretty("portrait", "orientation") == "Hochformat"
    # free text passes through, keys without a catalog entry too
    assert pretty("zebra crossing", "objects") == "zebra crossing"
    assert label("my_custom_key") == "My custom key"


def test_derived_values_are_known():
    """The fixed values the derived rules produce are the ones the catalogs know."""
    assert {derived.orientation(w, h) for w, h in [(4, 3), (3, 4), (5, 5), (30, 10)]} == set(
        derived.VALUES["orientation"]
    )
    assert {derived.capture_device(m) for m in [{}, {"make": "Apple"}, {"make": "Canon"}]} == set(
        derived.VALUES["capture_device"]
    )


def test_lazy_text_is_translated_when_shown(lang):
    note = Lazy("unknown person '%(name)s' searched as text", name="Zoe")
    joined = LazyJoin(", ", [note, Lazy("done")])
    assert str(joined) == "unknown person 'Zoe' searched as text, done"
    lang("de")
    assert str(joined) == "unbekannte Person „Zoe“ als Text gesucht, fertig"
    assert bool(Lazy("done")) and not bool(Lazy("")) and not bool(LazyJoin(", ", []))


def test_quiz_texts_and_frame_labels_follow_the_language(lang):
    from vision_tagger.faces.generations import Frame
    from vision_tagger.faces.quiz import streak_title

    when = datetime(2015, 5, 1)
    lang("de")
    assert streak_title(3) == "Warmgelaufen"
    assert Frame(1, 1, 7, False, when, 1.0).label == "7 Jahre · 2015"
    assert Frame(1, 1, 1, True, when, 1.0).label == "~1 Jahr"


# --------------------------------------------------------------------------- the tool
def test_collect_finds_every_kind_of_string(messages):
    origins = {o for m in messages.values() for o in m.origins}
    assert origins == {"template", "python", "js", "vocabulary"}
    assert "template" in messages["Skip to content"].origins
    assert messages["%(num)d photo"].plural == "%(num)d photos"
    assert "python" in messages["At the beach"].origins  # N_() in library.py
    assert "python" in messages["Wrong password."].origins
    assert "js" in messages["%(n)s selected"].origins
    assert "Location type" in messages and "meeting room" in messages
    assert "orientation|landscape" in messages  # a value with a context
    assert "folder picker|Start" in messages  # pgettext() in a template
    assert "Capture device" in messages and "maybe" in messages  # derived / file properties
    assert not [k for k in messages if not k.strip()]


def test_sync_keeps_translations_adds_and_drops(messages):
    small = {k: messages[k] for k in ("Photos", "%(num)d photo", "Faces")}
    old = {
        "@note": "kept",
        "Photos": "Fotos",
        "%(num)d photo": "%(num)d Foto",  # was singular: becomes the first form
        "gone": "weg",
    }
    out = i18n_extract.sync("ru", old, small)
    assert out["Photos"] == "Fotos" and out["Faces"] == ""
    assert out["%(num)d photo"] == ["%(num)d Foto", "", ""]  # three forms in Russian
    assert "gone" not in out and out["@note"] == "kept"
    assert out["@plurals"] == {"%(num)d photo": "%(num)d photos"}
    keys = [k for k in out if not k.startswith("@")]
    assert keys == sorted(keys, key=i18n_extract._sort_key)
    assert list(out)[: len([k for k in out if k.startswith("@")])] == [
        k for k in out if k.startswith("@")
    ]  # notes first
    assert i18n_extract.dump(out).endswith("}\n") and "Ä" in i18n_extract.dump({"x": "Ä"})


def test_check_reports_problems(messages):
    small = {
        k: messages[k]
        for k in (
            "Photos",
            "Page %(page)s of %(pages)s",
            "%(percent)s%% match",
            "%(num)d photo",
            "Write an XMP sidecar file next to the %(num)d original photo? The photo is not "
            "changed; sidecars of other apps are kept.",
        )
    }
    sidecar = next(k for k in small if k.startswith("Write"))

    def problems(lang: str, catalog: dict) -> list[tuple[str, str, str]]:
        return i18n_extract.check_catalog(lang, catalog, small)

    good = {
        "Photos": "Fotos",
        "Page %(page)s of %(pages)s": "Seite %(page)s von %(pages)s",
        "%(percent)s%% match": "%(percent)s %% Übereinstimmung",
        "%(num)d photo": ["%(num)d Foto", "%(num)d Fotos"],
        sidecar: ["Eine Datei schreiben?", "%(num)d Dateien schreiben?"],
    }
    assert problems("de", good) == []
    assert problems("en", {}) == []  # English is the source
    kinds = [p[0] for p in problems("de", {**good, "Photos": "", "old": "alt"})]
    assert kinds == ["missing", "obsolete"]

    def details(**changes: object) -> list[str]:
        return [p[2] for p in problems("de", {**good, **changes})]

    assert details(**{"Page %(page)s of %(pages)s": "Seite %(page)s"}) == [
        "placeholders %(page)s instead of %(page)s, %(pages)s"
    ]
    assert details(**{"Page %(page)s of %(pages)s": "Seite %(seite)s von %(pages)s"})
    assert details(**{"%(percent)s%% match": "%(percent)s Übereinstimmung"}) == [
        "the number of %% differs"
    ]
    assert details(Photos="100 % Fotos") == ["a lone % (write %% for a percent sign)"]
    assert details(Photos='Fotos "neu"') == ['characters " would break the HTML']
    assert details(Photos="<b>Fotos</b>") == ["characters < > would break the HTML"]
    assert details(**{"%(num)d photo": ["%(num)d Foto"]}) == ["needs a list of 2 plural forms"]
    assert details(**{"%(num)d photo": ["%(num)d Foto", ""]}) == ["empty plural form"]
    assert details(**{"%(num)d photo": ["%(num)d Foto", "Fotos"]}) == [
        "form 2: placeholders none instead of %(num)d"
    ]
    # French uses the singular for 0 and 1: it must show the number
    fr = {**good, sidecar: ["Écrire un fichier ?", "Écrire %(num)d fichiers ?"]}
    assert [p[2] for p in problems("fr", fr)] == ["form 1: placeholders none instead of %(num)d"]


def test_main_check_exit_code(capsys):
    assert i18n_extract.main(["check", "de"]) == 0
    assert "de: ok" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        i18n_extract.main(["check", "xx"])


def test_catalog_files_are_synced(messages):
    """Every non-English catalog exists and has exactly the current msgids (run update)."""
    for code in i18n.LANGUAGES:
        if code == i18n.DEFAULT:
            assert not i18n_extract.catalog_path(code).exists()
            continue
        path = i18n_extract.catalog_path(code)
        assert path.exists(), (
            f"{path} is missing: run `python -m vision_tagger.i18n_extract update`"
        )
        text = path.read_text(encoding="utf-8")
        assert text == i18n_extract.dump(i18n_extract.sync(code, json.loads(text), messages)), (
            f"{path.name} is not in sync with the sources: run "
            "`python -m vision_tagger.i18n_extract update`"
        )


@pytest.mark.parametrize("code", list(i18n.LANGUAGES))
def test_catalog_is_complete(code, messages):
    """Every string is translated, with the right placeholders.

    Expected to fail for a language until its translation is finished; see
    ``python -m vision_tagger.i18n_extract check <lang> -v`` for the full list.
    """
    problems = i18n_extract.check_catalog(code, i18n_extract.load(code), messages)
    if problems:
        pytest.fail(
            f"{i18n_extract.describe(code, problems, limit=15)}\n\n"
            f"The {i18n.LANGUAGES[code]} translation is incomplete or broken. Fill in "
            f"src/vision_tagger/resources/i18n/{code}.json and run "
            f"`python -m vision_tagger.i18n_extract check {code} -v`.",
            pytrace=False,
        )


@pytest.mark.parametrize("lang", [code for code in i18n.LANGUAGES if code != i18n.DEFAULT])
def test_sidebar_heading_differs_from_the_persons_page(lang):
    token = i18n.activate(lang)
    try:
        heading, page = i18n.pgettext("sidebar", "People"), i18n.gettext("Persons")
    finally:
        i18n.reset(token)
    assert heading != page, f"{lang}: the 'People' heading repeats the 'Persons' link below it"
