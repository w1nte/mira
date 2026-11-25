"""Command line interface: ``vision-tagger --help``."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from .config import Settings, load_settings

app = typer.Typer(
    help="Index, deduplicate, face-cluster and VLM-tag local image collections.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", help="Path to config.toml (default: ./config.toml or $VT_CONFIG)."
    ),
]


class State:
    settings: Settings | None = None


state = State()


@app.callback()
def main(
    config: ConfigOpt = None, verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    state.settings = load_settings(config)


def _settings() -> Settings:
    assert state.settings is not None
    return state.settings


def _db():
    from .db.migrate import MigrationRequired
    from .db.session import Database

    s = _settings()
    s.ensure_dirs()
    db = Database(s.db_path)
    try:
        db.create_all()
    except MigrationRequired as exc:
        db.dispose()
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    return db


@app.command()
def migrate(
    check: Annotated[
        bool, typer.Option("--check", help="Only report; exit code 1 if an upgrade is needed.")
    ] = False,
    backup: Annotated[
        bool, typer.Option("--backup/--no-backup", help="Copy the database file first.")
    ] = True,
    to: Annotated[
        str | None,
        typer.Option("--to", help="Target revision (default: newest). Older = downgrade."),
    ] = None,
) -> None:
    """Upgrade the database schema to this version of vision-tagger."""
    from .db.migrate import (
        MigrationError,
        downgrade,
        head_revision,
        migration_engine,
        schema_status,
        upgrade,
    )

    settings = _settings()
    settings.ensure_dirs()
    url = f"sqlite:///{settings.db_path.as_posix()}"
    engine = migration_engine(url)
    status = schema_status(engine)
    engine.dispose()
    current = status.current or (
        "none (new database)" if status.state == "new" else "pre-migrations"
    )
    typer.echo(f"database: {settings.db_path}")
    typer.echo(f"schema:   {current}  ->  newest: {status.head}")
    if check:
        if status.needs_upgrade:
            typer.echo("an upgrade is needed: run `vision-tagger migrate`")
            raise typer.Exit(1)
        typer.echo("up to date" if status.state == "current" else "new database")
        return
    try:
        if to is not None and to != "head" and status.current and to != head_revision():
            result = downgrade(url, to, backup=backup)
        else:
            result = upgrade(url, backup=backup)
    except MigrationError as exc:
        typer.echo(f"migration failed, database unchanged: {exc}", err=True)
        raise typer.Exit(1) from None
    if result.backup:
        typer.echo(f"backup:   {result.backup}")
    if result.changed:
        typer.echo(f"migrated: {result.before or 'pre-migrations'} -> {result.after}")
    else:
        typer.echo("already up to date")


@app.command()
def init() -> None:
    """Create the data directory, the database and a config.toml (if missing)."""
    _db()
    target = Path("config.toml")
    example = Path(__file__).parent / "resources" / "config.example.toml"
    if not target.exists():
        shutil.copy(example, target)
        typer.echo(f"wrote {target.resolve()} - edit image_roots and the model settings")
    typer.echo(f"database: {_settings().db_path}")


@app.command()
def index(
    roots: Annotated[
        list[Path] | None, typer.Argument(help="Folders to scan (default: config).")
    ] = None,
) -> None:
    """Step 0: scan folders recursively and update the image index."""
    from .indexer import Indexer

    db = _db()
    last = [0.0]

    def progress(stats) -> None:
        if time.monotonic() - last[0] > 2 or stats.done:
            last[0] = time.monotonic()
            typer.echo(
                f"  {stats.processed}/{stats.candidates} analysed · {stats.new} new · "
                f"{stats.duplicates} duplicates · {stats.errors} errors"
            )

    stats = Indexer(db, _settings(), progress=progress).run(roots or None)
    typer.echo(
        f"done: found {stats.found}, new {stats.new}, duplicates {stats.duplicates}, "
        f"moved {stats.moved}, updated {stats.updated}, unchanged {stats.unchanged}, "
        f"missing {stats.missing}, errors {stats.errors}"
    )
    for sample in stats.error_samples[:10]:
        typer.echo(f"  ! {sample}", err=True)


@app.command()
def run(
    stage: Annotated[
        list[str] | None, typer.Option("--stage", "-s", help="faces / vlm (repeatable)")
    ] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", "-b")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Max images per stage.")
    ] = None,
) -> None:
    """Run the pipeline in the foreground (Ctrl+C stops gracefully; resume with `run` again)."""
    from .pipeline import control
    from .pipeline.runner import BackgroundRunner, PipelineRunner
    from .pipeline.stages import default_stage_factory

    db = _db()
    settings = _settings()
    runner = BackgroundRunner(PipelineRunner(db, settings, default_stage_factory(db, settings)))
    runner.start(stage or None, batch_size=batch_size, limit=limit)
    try:
        while runner.running:
            runner.join(timeout=1.0)
    except KeyboardInterrupt:
        typer.echo("stopping after the current image…")
        control.request_state(db, control.STOPPED)
        runner.join()
    if runner.last_error:
        typer.echo(f"error: {runner.last_error}", err=True)
        raise typer.Exit(1)
    summary = runner.last_summary
    if summary:
        typer.echo(
            f"run {summary.run_id} {summary.status}: "
            f"processed {summary.processed}, failed {summary.failed}"
        )
        for m in summary.messages:
            typer.echo(f"  {m}")


@app.command()
def pause() -> None:
    """Pause a running pipeline (from another terminal or the web UI)."""
    from .pipeline import control

    control.request_state(_db(), control.PAUSED)


@app.command()
def resume() -> None:
    """Resume a paused pipeline."""
    from .pipeline import control

    control.request_state(_db(), control.RUNNING)


@app.command()
def stop() -> None:
    """Stop a running pipeline after the current image."""
    from .pipeline import control

    control.request_state(_db(), control.STOPPED)


@app.command()
def status() -> None:
    """Show per-stage progress."""
    from .pipeline import control
    from .pipeline.stages import stage_versions

    db = _db()
    counts = control.stage_counts(db)
    outdated = control.outdated_counts(db, stage_versions(_settings()))
    ctrl = control.get_control(db)
    typer.echo(
        f"pipeline: {ctrl.desired_state}" + (f" ({ctrl.active_stage})" if ctrl.active_stage else "")
    )
    for st, c in counts.items():
        typer.echo(
            f"  {st:6} done {c['done']:>7} / {c['total']:<7} pending {c['pending']:>7} "
            f"errors {c['error']:>5} skipped {c['skipped']:>5} outdated {outdated.get(st, 0):>5}"
        )


@app.command()
def cluster(
    full: Annotated[bool, typer.Option(help="Dissolve unlocked clusters first.")] = False,
) -> None:
    """(Re-)cluster faces and match the reference DB."""
    from .pipeline.stages.faces import cluster_and_match

    typer.echo(cluster_and_match(_db(), _settings(), full=full))


@app.command()
def refs() -> None:
    """Import the reference DB (references/<Person>/*.jpg) and name matching clusters."""
    from .faces.references import import_references
    from .pipeline.stages.faces import cluster_and_match

    db = _db()
    stats = import_references(db, _settings())
    typer.echo(f"references: {stats.added} added, {stats.skipped} known, persons {stats.persons}")
    for f in stats.no_face:
        typer.echo(f"  no face found: {f}", err=True)
    typer.echo(cluster_and_match(db, _settings(), full=False))


@app.command("retry-errors")
def retry_errors(stage: Annotated[str | None, typer.Option("--stage", "-s")] = None) -> None:
    """Re-queue failed images."""
    from .pipeline import control

    typer.echo(f"{control.retry_errors(_db(), stage)} items re-queued")


@app.command("requeue-outdated")
def requeue_outdated(stage: Annotated[str, typer.Option("--stage", "-s")] = "vlm") -> None:
    """Re-queue results produced with an older model / prompt / vocabulary."""
    from .pipeline import control
    from .pipeline.stages import stage_versions

    version = stage_versions(_settings())[stage]
    typer.echo(f"{control.requeue_outdated(_db(), stage, version)} items re-queued")


@app.command()
def export(
    target: Annotated[Path, typer.Argument(help="Target folder.")],
    query: Annotated[
        str,
        typer.Option(
            "--filter", "-f", help="Filter as URL query, e.g. 'p=1&t=location_type:beach'."
        ),
    ] = "",
    mode: Annotated[str, typer.Option(help="copy / hardlink / symlink")] = "",
    layout: Annotated[str, typer.Option(help="flat / date / folder")] = "flat",
) -> None:
    """Export the images matching a filter (originals are never touched)."""
    from starlette.datastructures import QueryParams

    from .export import export_images
    from .query import ImageFilter

    settings = _settings()
    stats = export_images(
        _db(),
        ImageFilter.from_params(QueryParams(query)),
        target,
        mode=(mode or settings.export.mode),  # type: ignore[arg-type]
        layout=layout,  # type: ignore[arg-type]
    )
    typer.echo(f"exported {stats.exported} files to {stats.target} ({stats.skipped} skipped)")
    for e in stats.errors[:10]:
        typer.echo(f"  ! {e}", err=True)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for, in any language.")],
    deep: Annotated[bool, typer.Option("--deep", help="Let the LLM re-rank and explain.")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Natural-language image search (LLM interpretation + semantic index)."""
    from sqlalchemy import select

    from .db.models import Image
    from .search.engine import SearchEngine

    db = _db()
    result = SearchEngine(db, _settings()).search(query, deep=deep, limit=limit)
    it = result.interpretation
    parts = [f"mode={it.mode}", f"semantic={'on' if result.semantic_used else 'off'}"]
    if it.persons:
        parts.append("persons=" + ",".join(n for _, n in it.persons))
    if it.must or it.should:
        parts.append("props=" + ",".join(f"{t.key}:{t.value}" for t in [*it.must, *it.should]))
    if it.exclude:
        parts.append("not=" + ",".join(f"{t.key}:{t.value}" for t in it.exclude))
    if it.date_from or it.date_to:
        parts.append(f"dates={it.date_from or ''}..{it.date_to or ''}")
    typer.echo(" ".join(parts))
    for note in result.notes:
        typer.echo(f"  note: {note}")
    with db.session() as s:
        paths = {
            image_id: path
            for image_id, path in s.execute(
                select(Image.id, Image.path).where(Image.id.in_([h.image_id for h in result.hits]))
            )
        }
    for rank, h in enumerate(result.hits, 1):
        line = f"{rank:>3}. {h.score:.2f}  {paths.get(h.image_id, h.image_id)}"
        if h.reason:
            line += f"  - {h.reason}"
        typer.echo(line)
    if not result.hits:
        typer.echo("no matches")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = False,
) -> None:
    """Start the local web UI."""
    import uvicorn

    from .web.app import create_app

    settings = _settings()
    _db().dispose()  # fails early with a clear message if `vision-tagger migrate` is needed
    host = host or settings.web.host
    port = port or settings.web.port
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    typer.echo(f"vision-tagger UI on http://{host}:{port}/")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
