"""FastAPI application factory."""

from __future__ import annotations

import io
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import Settings, load_settings
from ..db.models import Face, Image, PersonReference
from ..imaging import (
    ensure_face_crop,
    ensure_thumbnail,
    face_crop_path,
    load_image,
    thumbnail_is_fresh,
    thumbnail_path,
)
from .common import STATIC_DIR, ctx
from .context import AppContext
from .routes import (
    dashboard,
    faces,
    generations,
    images,
    overview,
    persons,
    properties,
    quiz,
    search,
)
from .security import allowed_hostnames, guard

log = logging.getLogger(__name__)

BROWSER_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Decoding a photo for a missing thumbnail or face crop is CPU work (and ~100 MB of memory
# for a 12 MP HEIC). A gallery asks for dozens at once; more parallel decodes than CPU cores
# only add memory pressure, so they queue here. Cached files are served without waiting.
_GENERATE = threading.BoundedSemaphore(max(2, min(8, os.cpu_count() or 2)))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = AppContext(settings)
        yield
        context: AppContext = app.state.ctx
        if context.runner.running:
            from ..pipeline import control

            control.request_state(context.db, control.STOPPED)
            context.runner.join(timeout=30)
        context.close()

    app = FastAPI(title="vision-tagger", lifespan=lifespan, docs_url="/api/docs")
    app.middleware("http")(guard(allowed_hostnames(settings.web.host, settings.web.allowed_hosts)))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    for module in (
        dashboard,
        faces,
        persons,
        properties,
        images,
        overview,
        search,
        quiz,
        generations,
    ):
        app.include_router(module.router)

    # ------------------------------------------------------------------ media
    @app.get("/media/thumb/{image_id}")
    def thumb(request: Request, image_id: int, size: int | None = None) -> Response:
        c = ctx(request)
        base_size = c.settings.web.thumb_size
        size = min(max(size or base_size, 64), 1024)
        with c.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise HTTPException(404)
            path = img.path
        target = thumbnail_path(c.settings.thumbs_dir, image_id, size)
        if not thumbnail_is_fresh(target, path):
            try:
                with _GENERATE:  # see _GENERATE: cached thumbnails never wait here
                    ensure_thumbnail(
                        c.settings.thumbs_dir, image_id, path, size, base_size=base_size
                    )
            except (OSError, ValueError) as exc:
                raise HTTPException(404, f"cannot read image: {exc}") from exc
        return FileResponse(target, media_type="image/webp", headers=_cache_headers())

    @app.get("/media/image/{image_id}")
    def original(request: Request, image_id: int) -> Response:
        c = ctx(request)
        with c.db.session() as s:
            img = s.get(Image, image_id)
            if img is None:
                raise HTTPException(404)
            path, ext = Path(img.path), img.ext
        if not path.exists():
            raise HTTPException(404, "file missing")
        if ext in BROWSER_FORMATS and not (img.meta or {}).get("orientation"):
            return FileResponse(path, headers=_cache_headers())
        # HEIC/TIFF or rotated via EXIF: serve an oriented JPEG rendition
        buf = io.BytesIO()
        load_image(path, max_px=3000).save(buf, "JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg", headers=_cache_headers())

    @app.get("/media/face/{face_id}")
    def face_crop(request: Request, face_id: int) -> Response:
        c = ctx(request)
        target = face_crop_path(c.settings.face_crops_dir, face_id)
        if not target.exists():
            with c.db.session() as s:
                face = s.get(Face, face_id)
                if face is None:
                    raise HTTPException(404)
                img = s.get(Image, face.image_id)
                assert img is not None
                path, w = img.path, img.width
                bbox = (face.x1, face.y1, face.x2, face.y2)
            with _GENERATE:
                try:
                    im = load_image(path, max_px=c.settings.faces.max_image_px)
                except OSError as exc:
                    raise HTTPException(404, str(exc)) from exc
                scale = im.width / w if w else 1.0
                ensure_face_crop(
                    c.settings.face_crops_dir,
                    face_id,
                    im,
                    tuple(v * scale for v in bbox),  # type: ignore[arg-type]
                    size=c.settings.faces.crop_size,
                )
        return FileResponse(target, media_type="image/webp", headers=_cache_headers())

    @app.get("/media/reference/{ref_id}")
    def reference(request: Request, ref_id: int) -> Response:
        c = ctx(request)
        with c.db.session() as s:
            ref = s.get(PersonReference, ref_id)
            if ref is None:
                raise HTTPException(404)
            face_id, source = ref.face_id, ref.source_path
        if face_id is not None:
            return face_crop(request, face_id)
        if source and Path(source).exists():
            buf = io.BytesIO()
            load_image(source, max_px=320).save(buf, "WEBP", quality=80)
            return Response(buf.getvalue(), media_type="image/webp")
        raise HTTPException(404)

    return app


def _cache_headers() -> dict[str, str]:
    return {"Cache-Control": "private, max-age=86400"}
