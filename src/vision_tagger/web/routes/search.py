"""Natural-language search page (LLM interpretation + semantic index + deep re-ranking)."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response

from ...db.models import Image
from ...db.util import chunked
from ...export import METADATA, export_images
from ..common import ctx, ints, redirect, render
from ..context import Job
from .images import export_summary

router = APIRouter()

EXAMPLES = [
    "Anna and Ben at the beach in summer",
    "snowy mountains without people",
    "Weihnachten mit der Familie",
    "screenshots and documents from work",
    "sunsets at the sea",
    "birthday party with cake",
]


@router.get("/search")
def search_page(request: Request, q: str = "", deep: int = 0) -> Response:
    q = " ".join(q.split())
    return render(
        request,
        "search.html",
        q=q,
        deep=bool(deep),
        examples=EXAMPLES,
        results_url="/search/results?" + urlencode({"q": q, "deep": int(bool(deep))}),
    )


@router.get("/search/results")
def search_results(request: Request, q: str = "", deep: int = 0) -> Response:
    q = " ".join(q.split())
    if not q:
        return Response("")
    c = ctx(request)
    result = c.search.search(q, deep=bool(deep))
    images: dict[int, Image] = {}
    ids = [h.image_id for h in result.hits]
    with c.db.session() as s:
        from sqlalchemy import select

        for part in chunked(ids):
            images.update((i.id, i) for i in s.scalars(select(Image).where(Image.id.in_(part))))
    it = result.interpretation
    return render(
        request,
        "partials/search_results.html",
        q=q,
        deep=bool(deep),
        result=result,
        it=it,
        images=images,
        filter_qs=it.hard_filter(result.relaxed).query_string(),
        export_mode=c.settings.export.mode,
    )


@router.post("/search/export")
def export_results(
    request: Request,
    ids: str = Form(""),
    target: str = Form(...),
    mode: str = Form("copy"),
    layout: str = Form("flat"),
    q: str = Form(""),
    metadata: list[str] = Form([]),
) -> Response:
    c = ctx(request)
    if c.settings.remote_mode:  # remote users may not write to folders on the server
        raise HTTPException(403, "use the ZIP download in remote mode")
    image_ids = ints(ids.split(","))
    back_url = "/search?" + urlencode({"q": q})
    if not image_ids or not target.strip():
        return redirect(request, back_url, err="Nothing to export or no target folder.")
    mode = mode if mode in ("copy", "hardlink", "symlink") else "copy"
    layout = layout if layout in ("flat", "date", "folder") else "flat"

    def job(_j: Job) -> str:
        from ...query import ImageFilter

        stats = export_images(
            c.db,
            ImageFilter(),
            target.strip(),
            mode=mode,
            layout=layout,
            image_ids=image_ids,  # type: ignore[arg-type]
            metadata=[m for m in metadata if m in METADATA],
            sidecar_naming=c.settings.export.sidecar_naming,
        )
        return export_summary(stats, "Search export")

    if not c.jobs.start("export", job):
        return redirect(request, back_url, err="Another job is running.")
    return redirect(request, back_url, msg=f"Exporting {len(image_ids)} images – see dashboard.")


@router.get("/search/download.zip")
def download_results(request: Request, ids: str = "") -> Response:
    from fastapi.responses import StreamingResponse

    from ...export import stream_zip

    c = ctx(request)
    return StreamingResponse(
        stream_zip(c.db, ints(ids.split(","))),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="vision-tagger-search.zip"'},
    )
