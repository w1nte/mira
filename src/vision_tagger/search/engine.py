"""Hybrid search: structured filters + what photos show + their descriptions + properties.

1. The query is interpreted (LLM, or keyword fallback) into persons, pets, dates, required /
   preferred / excluded properties, keywords, an English "semantic" rewrite and the text that
   describes what the photos should show. Alternatives ("Anna or Ben", "beach or mountains",
   "2019 or 2021") are OR groups; groups are AND-ed.
2. The strict part (persons, pets, required properties and their alternatives, exclusions,
   dates) selects candidates via the normal gallery filter (``query.conditions``). Photos the
   vision model has not described yet can't have the required properties; they stay candidates
   (by what they show) as long as everything else matches. If nothing matches, required
   properties are relaxed into preferences. Without a strict part, every photo is a candidate.
3. Candidates are scored by: the visual similarity of the request to each photo (image-text
   embeddings, see ``visual``), the semantic similarity of the rewrite to each image's document
   embedding, how many preferred properties / persons they have, and keyword hits in captions,
   visible text, property values and file names. A photo is scored by the parts it has: one
   without a description yet is ranked by the others. Photos without any evidence are left out.
4. Deep search (optional): the LLM re-ranks the best described candidates and explains each
   choice.

"Similar photos" ranks the library by the visual similarity to one photo.
"""

from __future__ import annotations

import contextlib
import logging
import math
import string
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import and_, func, or_, select

from ..config import Settings
from ..db.cache import VersionedCache
from ..db.models import (
    Animal,
    Cluster,
    Face,
    Image,
    ImageEmbedding,
    ImageProperty,
    VisualEmbedding,
    VlmResult,
)
from ..db.session import Database
from ..db.util import chunked, like_escape
from ..geo.places import PLACE_KEYS
from ..i18n import Lazy
from ..query import ImageFilter, all_image_ids, property_counts
from ..vlm.backend import VlmBackend, VlmError, make_backend
from ..vlm.schema import Vocabulary, default_synonyms_path
from .index import EmbeddingMatrix
from .interpret import (
    Interpretation,
    KeywordInterpreter,
    PropTerm,
    SearchContext,
    build_context,
    llm_interpret,
)
from .rerank import describe, rerank

log = logging.getLogger(__name__)


#: property keys known right after indexing or from other stages; the other keys come from
#: the vision model, so photos it has not described yet can't have them
_KNOWN_WITHOUT_VLM = {"year", "orientation", "capture_device", "has_gps", "animal", "pet"} | set(
    PLACE_KEYS
)
#: "Similar photos": pHashes this close (of 64 bits) are copies of one photo (re-encoded,
#: resized); the indexer merges only identical pHashes
COPY_PHASH_BITS = 4


@dataclass
class SearchHit:
    image_id: int
    score: float
    semantic: float | None = None
    properties: float | None = None
    keywords: float | None = None
    visual: float | None = None
    rerank: float | None = None
    reason: str | None = None


@dataclass
class SearchResult:
    interpretation: Interpretation
    hits: list[SearchHit]
    candidates: int = 0
    relaxed: bool = False
    semantic_used: bool = False
    semantic_coverage: float = 0.0
    visual_used: bool = False
    visual_coverage: float = 0.0
    #: "Similar photos": the photo the others are similar to
    similar_to: int | None = None
    deep: bool = False
    notes: list[str | Lazy] = field(default_factory=list)  # shown in the UI: Lazy
    seconds: float = 0.0


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _ascii_lower(text: str) -> str:
    """Lower-case like SQLite's ``lower()`` / ``LIKE`` (ASCII letters only)."""
    return text.translate(_ASCII_LOWER)


def visual_score(cos: float, low: float, high: float) -> float:
    """Cosine similarity of a request and a photo -> 0..1.

    0 up to ``low`` (photos that don't show it), 0.75 at ``high`` (a clear match), approaching
    1 above, so clear matches keep their order. Absolute, not relative to the candidates: a
    request nothing matches gets low scores, and the other parts decide.
    """
    z = (cos - low) / max(high - low, 1e-6)
    return 0.0 if z <= 0 else 1.0 - math.exp(-z * math.log(4.0))


class SearchEngine:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        backend: VlmBackend | None = None,
        text_encoder: Any = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.vocab = Vocabulary.load(settings.vlm.vocabulary_file)
        self.backend = backend or make_backend(settings.vlm, self.vocab)
        self.index = EmbeddingMatrix(db)
        self.visual = EmbeddingMatrix(db, VisualEmbedding)
        if text_encoder is None:
            from ..visual.encoder import TextEncoder

            text_encoder = TextEncoder(settings)
        #: turns requests into vectors of the visual space (tests replace it with a fake)
        self.text_encoder = text_encoder
        self.cache = VersionedCache(db)
        self._cache: OrderedDict[tuple[str, str], Interpretation] = OrderedDict()
        self._ctx: tuple[float, SearchContext] | None = None

    @property
    def llm_model(self) -> str:
        return self.settings.search.model or self.settings.vlm.model

    # ------------------------------------------------------------------ interpretation
    def context(self, max_age: float = 30.0) -> SearchContext:
        if self._ctx is None or time.monotonic() - self._ctx[0] > max_age:
            synonyms = self.settings.vlm.synonyms_file or default_synonyms_path()
            with self.db.session() as s:
                self._ctx = (
                    time.monotonic(),
                    build_context(s, self.vocab, synonyms, counts=self.value_counts()),
                )
        return self._ctx[1]

    def interpret(self, query: str) -> Interpretation:
        ctx = self.context()
        key = (query.strip(), ctx.fingerprint)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            it = llm_interpret(self.backend, self.llm_model, query, ctx)
        except (VlmError, ValueError, KeyError, TypeError) as exc:
            log.info("LLM query interpretation unavailable (%s); using keyword mode", exc)
            it = KeywordInterpreter().interpret(query, ctx)
            if self.backend.model != "fake":
                it.notes.append(
                    Lazy("LLM unavailable (%(error)s); basic keyword mode", error=str(exc))
                )
        else:
            self._cache[key] = it
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return it

    # ------------------------------------------------------------------ scoring parts
    def _property_scores(
        self, terms: list[PropTerm], it: Interpretation, candidates: set[int]
    ) -> dict[int, float] | None:
        # "Anna or Bello": photos with more of them rank higher
        any_persons = it.persons_mode == "any" and len(it.persons) + len(it.pets) > 1
        if not terms and not any_persons:
            return None
        total = sum(t.weight for t in terms) + (
            len(it.persons) + len(it.pets) if any_persons else 0
        )
        scores: dict[int, float] = {}
        with self.db.session() as s:
            if terms:
                cond = or_(
                    *(
                        and_(ImageProperty.key == t.key, ImageProperty.value == t.value)
                        for t in terms
                    )
                )
                weights = {(t.key, t.value): t.weight for t in terms}
                for image_id, key, value, conf in s.execute(
                    select(
                        ImageProperty.image_id,
                        ImageProperty.key,
                        ImageProperty.value,
                        ImageProperty.confidence,
                    ).where(cond)
                ):
                    if image_id in candidates:
                        factor = conf if conf is not None and key not in ("year",) else 1.0
                        factor = max(0.4, float(factor))
                        scores[image_id] = (
                            scores.get(image_id, 0.0) + weights[(key, value)] * factor
                        )
            if any_persons:
                pids = [pid for pid, _ in it.persons]
                rows = s.execute(
                    select(Face.image_id, func.count(func.distinct(Cluster.person_id)))
                    .join(Cluster, Cluster.id == Face.cluster_id)
                    .where(Cluster.person_id.in_(pids), Face.ignored.is_(False))
                    .group_by(Face.image_id)
                )
                for image_id, n in rows:
                    if image_id in candidates:
                        scores[image_id] = scores.get(image_id, 0.0) + n
                pet_ids = [pid for pid, _ in it.pets]
                rows = s.execute(
                    select(Animal.image_id, func.count(func.distinct(Animal.pet_id)))
                    .where(Animal.pet_id.in_(pet_ids), Animal.ignored.is_(False))
                    .group_by(Animal.image_id)
                )
                for image_id, n in rows:
                    if image_id in candidates:
                        scores[image_id] = scores.get(image_id, 0.0) + n
        return {i: v / total for i, v in scores.items()} if total else None

    def _keyword_scores(self, keywords: list[str], candidates: set[int]) -> dict[int, float] | None:
        if not keywords:
            return None
        hits: dict[int, int] = {}
        with self.db.session() as s:
            for word in keywords:
                like = f"%{like_escape(word)}%"
                found: set[int] = set()
                for stmt in (
                    select(VlmResult.image_id).where(
                        or_(
                            VlmResult.caption.ilike(like, escape="\\"),
                            func.json_extract(VlmResult.raw_json, "$.text_excerpt").ilike(
                                like, escape="\\"
                            ),
                        )
                    ),
                    select(Image.id).where(Image.rel_path.ilike(like, escape="\\")),
                ):
                    found.update(i for i in s.scalars(stmt) if i in candidates)
                found.update(i for i in self._images_with_value(s, word) if i in candidates)
                for i in found:
                    hits[i] = hits.get(i, 0) + 1
        return {i: n / len(keywords) for i, n in hits.items()}

    def value_counts(self) -> dict[str, list[tuple[str, int]]]:
        """``property_counts`` of the whole library, cached until the database changes."""
        return self.cache.get("property_counts", property_counts)

    def _images_with_value(self, s, word: str) -> set[int]:
        """Images with a property value containing ``word`` (like ``value ILIKE '%word%'``).

        The distinct values (a few thousand) are matched here instead of scanning millions of
        property rows; then only the matching values are looked up in the index.
        """
        needle = _ascii_lower(word)
        matches: dict[str, list[str]] = {}
        for key, values in self.value_counts().items():
            hits = [v for v, _n in values if needle in _ascii_lower(v)]
            if hits:
                matches[key] = hits
        out: set[int] = set()
        for key, values in matches.items():
            for part in chunked(values):
                out.update(
                    s.scalars(
                        select(ImageProperty.image_id).where(
                            ImageProperty.key == key, ImageProperty.value.in_(part)
                        )
                    )
                )
        return out

    def _semantic_scores(
        self, it: Interpretation, candidates: list[int], notes: list[str | Lazy]
    ) -> dict[int, float] | None:
        model = self.settings.search.embed_model
        try:
            self.index.ensure(model)
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(Lazy("semantic index unavailable: %(error)s", error=str(exc)))
            return None
        if self.index.count == 0:
            return None
        try:
            q = self.backend.embed([self.settings.search.query_prefix + it.semantic_query], model)[
                0
            ]
        except (VlmError, ValueError) as exc:
            notes.append(Lazy("semantic search unavailable (%(error)s)", error=str(exc)))
            return None
        sims = self.index.similarities(q, candidates)
        if not sims:
            return None
        lo, hi = min(sims.values()), max(sims.values())
        span = hi - lo
        return {i: (v - lo) / span if span > 1e-6 else 1.0 for i, v in sims.items()}

    @property
    def visual_model(self) -> str:
        """The vector space searched (``visual_embeddings.model``)."""
        return self.text_encoder.model

    def _visual_ready(self) -> bool:
        """Load the visual vectors of the current model; ``False`` if there are none."""
        try:
            self.visual.ensure(self.visual_model)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("visual index unavailable: %s", exc)
            return False
        return self.visual.count > 0

    def _visual_scores(
        self, it: Interpretation, candidates: list[int], notes: list[str | Lazy]
    ) -> dict[int, float] | None:
        text = it.visual_query.strip()
        if not text or not self._visual_ready():
            return None
        try:
            q = self.text_encoder.embed([text])[0]
        except Exception as exc:  # model files missing, onnxruntime errors
            log.warning("visual search unavailable: %s", exc)
            notes.append(Lazy("visual search unavailable (%(error)s)", error=str(exc)))
            return None
        low, high = self.text_encoder.low, self.text_encoder.high
        sims = self.visual.similarities(q, candidates)
        return {i: visual_score(v, low, high) for i, v in sims.items()}

    def _described(self, image_ids: list[int]) -> set[int]:
        """The photos among ``image_ids`` the vision model has described."""
        out: set[int] = set()
        with self.db.session() as s:
            for part in chunked(image_ids):
                out.update(
                    s.scalars(select(VlmResult.image_id).where(VlmResult.image_id.in_(part)))
                )
        return out

    def _undescribed(self, s, it: Interpretation, strict: list[int]) -> list[int]:
        """The candidates plus the photos without a description yet (no answer of the vision
        model) that match everything else: they can't have the required properties yet, but
        may show them. ``strict`` itself if there are none."""
        f = it.hard_filter()
        props = [p for p in f.props if p[0] in _KNOWN_WITHOUT_VLM]
        groups = [g for g in f.prop_groups if all(k in _KNOWN_WITHOUT_VLM for k, _ in g)]
        if len(props) == len(f.props) and len(groups) == len(f.prop_groups):
            return strict
        wider = all_image_ids(s, f.copy(props=props, prop_groups=groups))
        strict_set = set(strict)
        others = [i for i in wider if i not in strict_set and self.visual.has(i)]
        described: set[int] = set()
        for part in chunked(others):
            described.update(
                s.scalars(select(VlmResult.image_id).where(VlmResult.image_id.in_(part)))
            )
        extra = {i for i in others if i not in described}
        if not extra:
            return strict
        return [i for i in wider if i in strict_set or i in extra]  # newest first

    # ------------------------------------------------------------------ public
    def search(self, query: str, *, deep: bool = False, limit: int | None = None) -> SearchResult:
        started = time.monotonic()
        query = " ".join(query.split())
        limit = limit or self.settings.search.result_limit
        it = self.interpret(query)
        result = SearchResult(interpretation=it, hits=[], notes=list(it.notes))
        visual_ready = self._visual_ready()
        widened = False
        hard = it.hard_filter()
        # no strict part: every photo is a candidate, and only photos that match something
        # about the request are results (strict candidates are results anyway)
        open_search = hard.is_empty
        with self.db.session() as s:
            candidates = all_image_ids(s, hard)
            strict = set(candidates)
            if it.required and visual_ready and it.visual_query:
                candidates = self._undescribed(s, it, candidates)
                widened = len(candidates) > len(strict)
                if widened:
                    result.notes.append(
                        Lazy(
                            "%(num)s photos without a description yet are included by what "
                            "they show",
                            num=len(candidates) - len(strict),
                        )
                    )
            if not candidates and it.required:  # the closest matches, all of them ranked
                candidates = all_image_ids(s, it.hard_filter(relaxed=True))
                strict, open_search = set(candidates), False
                result.relaxed = bool(candidates)
            total = s.scalar(
                select(func.count()).select_from(Image).where(Image.missing.is_(False))
            )
            embedded = s.scalar(
                select(func.count())
                .select_from(ImageEmbedding)
                .where(ImageEmbedding.model == self.settings.search.embed_model)
            )
        result.candidates = len(candidates)
        result.semantic_coverage = (embedded or 0) / total if total else 0.0
        result.visual_coverage = min(1.0, self.visual.count / total) if total else 0.0
        if not candidates:
            result.seconds = time.monotonic() - started
            return result

        cand_set = set(candidates)
        w = self.settings.search.weights
        visual = self._visual_scores(it, candidates, result.notes)
        result.visual_used = visual is not None
        if visual is not None and result.visual_coverage < 0.995:
            result.notes.append(
                Lazy(
                    "%(percent)s%% of the photos can be found by what they show so far; the "
                    "“Visual search” stage embeds the others",
                    percent=f"{result.visual_coverage * 100:.0f}",
                )
            )
        semantic = self._semantic_scores(it, candidates, result.notes)
        result.semantic_used = semantic is not None
        terms = it.soft_terms(result.relaxed or widened)
        props = self._property_scores(terms, it, cand_set)
        words = self._keyword_scores(it.keywords, cand_set)
        # properties only the vision model gives count for the photos it has described: the
        # others can't have them yet (they are ranked by what they show)
        props_for: set[int] | None = None
        vlm_only = all(t.key not in _KNOWN_WITHOUT_VLM for t in terms) and not it.who_any
        if props is not None and visual is not None and vlm_only:
            props_for = self._described(candidates)
        # (weight, scores, the photos the part counts for; None: every photo)
        parts: list[tuple[float, dict[int, float], set[int] | dict[int, float] | None]] = [
            (weight, scores, applies)
            for weight, scores, applies in (
                (w.visual, visual, visual),  # photos with a vector
                (w.semantic, semantic, semantic),  # photos with a document embedding
                (w.properties, props, props_for),
                (w.keywords, words, None),
            )
            if scores is not None and weight > 0
        ]

        ranked: list[tuple[float, int, int]] = []  # (-score, newest-first rank, image id)
        for rank, image_id in enumerate(candidates):  # candidates come newest first
            weight_sum, score = 0.0, 0.0
            for weight, scores, applies in parts:
                if applies is None or image_id in applies:
                    score += weight * scores.get(image_id, 0.0)
                    weight_sum += weight
            score = score / weight_sum if weight_sum else 0.0
            if parts and score <= 0.0 and (open_search or image_id not in strict):
                continue  # nothing about this photo matches the request
            ranked.append((-score, rank, image_id))
        ranked.sort()

        def part(scores: dict[int, float] | None, image_id: int) -> float | None:
            return None if scores is None else round(scores.get(image_id, 0.0), 3)

        hits = [
            SearchHit(
                image_id=image_id,
                score=round(-neg_score, 4),
                semantic=part(semantic, image_id),
                properties=part(props, image_id),
                keywords=part(words, image_id),
                visual=part(visual, image_id),
            )
            for neg_score, _rank, image_id in ranked[:limit]
        ]

        if deep and hits:
            self._deep(query, hits, result)
        result.hits = hits
        result.seconds = round(time.monotonic() - started, 2)
        return result

    def similar(self, image_id: int, *, limit: int | None = None) -> SearchResult:
        """The photos that look most like ``image_id`` (visual similarity), without the photo
        itself and copies of a photo already listed.

        A copy (re-encoded, resized) has nearly the same pHash. Vectors can't tell: the same
        photo re-encoded has a cosine of 0.88-0.99, two shots of the same scene up to 0.99.
        """
        started = time.monotonic()
        limit = limit or self.settings.search.result_limit
        it = Interpretation(query="", mode="similar")
        result = SearchResult(interpretation=it, hits=[], similar_to=image_id)
        vector = self.visual.vector(image_id) if self._visual_ready() else None
        if vector is None:
            result.notes.append(
                Lazy(
                    "This photo has no visual vector yet; the “Visual search” stage on the "
                    "Pipeline page embeds it"
                )
            )
            result.seconds = round(time.monotonic() - started, 2)
            return result
        with self.db.session() as s:
            candidates = all_image_ids(s, ImageFilter())
            total = len(candidates)
            sims = self.visual.similarities(vector, candidates)
            ranked = sorted(
                ((sim, i) for i, sim in sims.items() if i != image_id),
                key=lambda t: (-t[0], t[1]),
            )
            # pHashes of the photo and of enough neighbours to fill the list without copies
            wanted = [image_id] + [i for _sim, i in ranked[: 2 * limit + 50]]
            hashes: dict[int, int] = {}
            for part in chunked(wanted):
                for i, value in s.execute(select(Image.id, Image.phash).where(Image.id.in_(part))):
                    with contextlib.suppress(TypeError, ValueError):
                        hashes[i] = int(value, 16)
        result.visual_used = True
        result.visual_coverage = min(1.0, self.visual.count / total) if total else 0.0
        result.candidates = len(ranked)
        shown: list[int] = [hashes[image_id]] if image_id in hashes else []
        for sim, i in ranked:
            if len(result.hits) >= limit:
                break
            h = hashes.get(i)
            if h is not None:
                if any((h ^ other).bit_count() <= COPY_PHASH_BITS for other in shown):
                    continue  # a copy of the photo or of one listed already
                shown.append(h)
            result.hits.append(
                SearchHit(image_id=i, score=round(max(0.0, sim), 4), visual=round(sim, 3))
            )
        result.seconds = round(time.monotonic() - started, 2)
        return result

    def _deep(self, query: str, hits: list[SearchHit], result: SearchResult) -> None:
        """The LLM judges the best matches it can read about (photos with a description);
        photos without one keep their place."""
        top = hits[: self.settings.search.rerank_top]
        with self.db.session() as s:
            ids = [h.image_id for h in top]
            described = set(
                s.scalars(select(VlmResult.image_id).where(VlmResult.image_id.in_(ids)))
            ) | set(
                s.scalars(
                    select(ImageProperty.image_id).where(
                        ImageProperty.image_id.in_(ids), ImageProperty.source == "vlm"
                    )
                )
            )
            descriptions = describe(s, [i for i in ids if i in described])
        if not descriptions:
            result.notes.append(Lazy("deep search needs descriptions from the vision model"))
            return
        try:
            judged = rerank(
                self.backend, self.llm_model, query, descriptions, self.settings.search.rerank_batch
            )
        except (VlmError, ValueError, KeyError, TypeError) as exc:
            result.notes.append(Lazy("deep search unavailable (%(error)s)", error=str(exc)))
            return
        for h in top:
            j = judged.get(h.image_id)
            if j is not None:
                h.rerank, h.reason = j.score, j.reason
        order = {h.image_id: i for i, h in enumerate(top)}
        slots = [n for n, h in enumerate(top) if h.image_id in descriptions]
        judged_hits = sorted(
            (top[n] for n in slots),
            key=lambda h: (-(h.rerank if h.rerank is not None else -1), order[h.image_id]),
        )
        for n, h in zip(slots, judged_hits, strict=True):
            top[n] = h
        hits[: len(top)] = top
        result.deep = bool(judged)
