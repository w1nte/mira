"""Hybrid search: structured filters + property matches + keywords + semantic similarity.

1. The query is interpreted (LLM, or keyword fallback) into persons, pets, dates, required /
   preferred / excluded properties, keywords and an English "semantic" rewrite. Alternatives
   ("Anna or Ben", "beach or mountains", "2019 or 2021") are OR groups; groups are AND-ed.
2. The strict part (persons, pets, required properties and their alternatives, exclusions,
   dates) selects candidates via the normal gallery filter (``query.conditions``). If nothing
   matches, required properties are relaxed into preferences.
3. Candidates are scored by: semantic similarity of the rewrite to each image's document
   embedding, how many preferred properties / persons they have, and keyword hits in captions,
   visible text, property values and file names.
4. Deep search (optional): the LLM re-ranks the best candidates and explains each choice.
"""

from __future__ import annotations

import logging
import string
import time
from collections import OrderedDict
from dataclasses import dataclass, field

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
    VlmResult,
)
from ..db.session import Database
from ..db.util import chunked, like_escape
from ..i18n import Lazy
from ..query import all_image_ids, property_counts
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


@dataclass
class SearchHit:
    image_id: int
    score: float
    semantic: float | None = None
    properties: float | None = None
    keywords: float | None = None
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
    deep: bool = False
    notes: list[str | Lazy] = field(default_factory=list)  # shown in the UI: Lazy
    seconds: float = 0.0


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _ascii_lower(text: str) -> str:
    """Lower-case like SQLite's ``lower()`` / ``LIKE`` (ASCII letters only)."""
    return text.translate(_ASCII_LOWER)


class SearchEngine:
    def __init__(self, db: Database, settings: Settings, backend: VlmBackend | None = None) -> None:
        self.db = db
        self.settings = settings
        self.vocab = Vocabulary.load(settings.vlm.vocabulary_file)
        self.backend = backend or make_backend(settings.vlm, self.vocab)
        self.index = EmbeddingMatrix(db)
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

    # ------------------------------------------------------------------ public
    def search(self, query: str, *, deep: bool = False, limit: int | None = None) -> SearchResult:
        started = time.monotonic()
        query = " ".join(query.split())
        limit = limit or self.settings.search.result_limit
        it = self.interpret(query)
        result = SearchResult(interpretation=it, hits=[], notes=list(it.notes))
        with self.db.session() as s:
            candidates = all_image_ids(s, it.hard_filter())
            if not candidates and it.required:
                candidates = all_image_ids(s, it.hard_filter(relaxed=True))
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
        if not candidates:
            result.seconds = time.monotonic() - started
            return result

        cand_set = set(candidates)
        w = self.settings.search.weights
        parts: list[tuple[float, dict[int, float]]] = []
        semantic = self._semantic_scores(it, candidates, result.notes)
        result.semantic_used = semantic is not None
        props = self._property_scores(it.soft_terms(result.relaxed), it, cand_set)
        words = self._keyword_scores(it.keywords, cand_set)
        for weight, scores in ((w.semantic, semantic), (w.properties, props), (w.keywords, words)):
            if scores is not None and weight > 0:
                parts.append((weight, scores))
        weight_sum = sum(p[0] for p in parts) or 1.0

        ranked: list[tuple[float, int, int]] = []  # (-score, newest-first rank, image id)
        for rank, image_id in enumerate(candidates):  # candidates come newest first
            score = sum(weight * scores.get(image_id, 0.0) for weight, scores in parts) / weight_sum
            ranked.append((-score, rank, image_id))
        ranked.sort()
        hits = [
            SearchHit(
                image_id=image_id,
                score=round(-neg_score, 4),
                semantic=None if semantic is None else round(semantic.get(image_id, 0.0), 3),
                properties=None if props is None else round(props.get(image_id, 0.0), 3),
                keywords=None if words is None else round(words.get(image_id, 0.0), 3),
            )
            for neg_score, _rank, image_id in ranked[:limit]
        ]

        if deep and hits:
            self._deep(query, hits, result)
        result.hits = hits
        result.seconds = round(time.monotonic() - started, 2)
        return result

    def _deep(self, query: str, hits: list[SearchHit], result: SearchResult) -> None:
        top = hits[: self.settings.search.rerank_top]
        with self.db.session() as s:
            descriptions = describe(s, [h.image_id for h in top])
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
        top.sort(key=lambda h: (-(h.rerank if h.rerank is not None else -1), order[h.image_id]))
        hits[: len(top)] = top
        result.deep = bool(judged)
