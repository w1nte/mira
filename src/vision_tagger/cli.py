"""Command line interface: ``vision-tagger --help``."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

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


#: exit codes of `migrate` (installers rely on them)
MIGRATE_NEEDED, MIGRATE_TOO_NEW, MIGRATE_FAILED = 1, 3, 4


class _Report:
    """What `migrate` says: on the console and, for the desktop library, in its log folder."""

    def __init__(self, log_file: Path | None) -> None:
        self.log_file = log_file
        if log_file is not None:
            from . import __version__

            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} migrate, version {__version__}")

    def __call__(self, message: str, *, err: bool = False) -> None:
        typer.echo(message, err=err)
        self.write(message)

    def write(self, message: str) -> None:
        if self.log_file is not None:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(message + "\n")


def _report_config(say: Callable[[str], None] = typer.echo) -> bool:
    """Say what loading config.toml did; True if it is outdated and could not be updated."""
    from . import config as config_mod
    from .config_migrate import CONFIG_VERSION, load_config

    path = config_mod._resolve_config_file()
    if path is None:
        say("config:   none (defaults and VT_ environment variables)")
        return False
    done = config_mod.last_config_migration
    if done is not None and done.path == path:
        if not done.written:
            say(f"config:   {path}: version {done.before}, could not be updated: {done.error}")
            return True
        say(f"config:   {path}: version {done.before} -> {done.after}")
        say(f"backup:   {done.backup}")
        for note in done.notes:
            say(f"          - {note}")
        return False
    _data, now = load_config(path, write=False)
    state = "up to date" if now.before == CONFIG_VERSION else f"newest: {CONFIG_VERSION}"
    say(f"config:   {path}: version {now.before}, {state}")
    return now.before < CONFIG_VERSION


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
    desktop: Annotated[
        bool,
        typer.Option(
            "--desktop",
            help="The desktop app's library, found as the app finds it (for installers). "
            "Creates nothing when there is none yet; logs to its logs/migrate.log.",
        ),
    ] = False,
) -> None:
    """Upgrade the database schema (and config.toml) to this version of vision-tagger.

    An outdated config.toml is updated whenever it is loaded, also with --check. Exit codes:
    0 done or nothing to do, 1 an upgrade is needed (--check), 3 the database is from a newer
    version, 4 the migration failed (the database is unchanged).
    """
    import traceback

    from .db.migrate import (
        DatabaseTooNew,
        downgrade,
        head_revision,
        migration_engine,
        schema_status,
        upgrade,
    )

    if desktop:
        from . import APP_NAME
        from .desktop import find_desktop_settings, user_data_dir

        found = find_desktop_settings()
        if found is None:
            typer.echo(f"no {APP_NAME} library yet ({user_data_dir()}): nothing to migrate")
            return
        if not found.db_path.is_file():
            typer.echo(f"no database yet ({found.db_path}): nothing to migrate")
            return
        settings = found
        report = _Report(settings.data_dir / "logs" / "migrate.log")
        logging.getLogger("alembic").setLevel(logging.WARNING)  # the report says what matters
    else:
        settings = _settings()
        settings.ensure_dirs()
        report = _Report(None)
    url = f"sqlite:///{settings.db_path.as_posix()}"
    report(f"database: {settings.db_path}")
    config_outdated = _report_config(report)
    try:
        engine = migration_engine(url)
        try:
            status = schema_status(engine)
        finally:
            engine.dispose()
        current = status.current or (
            "none (new database)" if status.state == "new" else "pre-migrations"
        )
        report(f"schema:   {current}  ->  newest: {status.head}")
        if status.state == "newer":
            raise status.too_new(settings.db_path)
        if check:
            if status.needs_upgrade or config_outdated:
                report("an upgrade is needed: run `vision-tagger migrate`")
                raise typer.Exit(MIGRATE_NEEDED)
            report("up to date" if status.state == "current" else "new database")
            return
        if to is not None and to != "head" and status.current and to != head_revision():
            result = downgrade(url, to, backup=backup)
        else:
            result = upgrade(url, backup=backup)
    except DatabaseTooNew as exc:
        report(f"error: {exc}", err=True)
        raise typer.Exit(MIGRATE_TOO_NEW) from None
    except typer.Exit:
        raise
    except Exception as exc:
        report(f"migration failed, database unchanged: {exc}", err=True)
        report.write(traceback.format_exc())
        logging.getLogger(__name__).debug("migration failed", exc_info=True)  # with --verbose
        raise typer.Exit(MIGRATE_FAILED) from None
    if result.backup:
        report(f"backup:   {result.backup}")
    if result.changed:
        report(f"migrated: {result.before or 'pre-migrations'} -> {result.after}")
    else:
        report("already up to date")


@app.command("quit-app")
def quit_app(
    timeout: Annotated[
        float, typer.Option(help="Seconds to wait until the app has exited.")
    ] = 120.0,
) -> None:
    """Quit the running desktop app as if its window was closed (for installers).

    A running pipeline is paused and continues on the next start. Exit codes: 0 it has quit or
    was not running, 1 still running after --timeout, 3 the running app is too old to quit on
    request.
    """
    from .desktop import quit_app as quit_desktop_app

    raise typer.Exit(quit_desktop_app(timeout, echo=typer.echo))


#: exit codes of `update-check`
UPDATE_CURRENT, UPDATE_AVAILABLE, UPDATE_UNKNOWN = 0, 1, 2


@app.command("update-check")
def update_check(
    prereleases: Annotated[
        bool | None,
        typer.Option(
            "--prereleases/--no-prereleases",
            help="Also consider release candidates and betas (default: updates.prereleases).",
        ),
    ] = None,
) -> None:
    """Look for a newer Mira release on GitHub (one request to api.github.com).

    Prints the running and the newest version and the download for this computer. Nothing is
    downloaded or installed. Exit codes: 0 up to date, 1 an update is available, 2 unknown (no
    connection, rate limit, unexpected answer). Works whatever updates.check says.
    """
    from . import APP_NAME, __version__, updates

    settings = _settings()
    include = settings.updates.prereleases if prereleases is None else prereleases
    client = updates.make_client()
    try:
        result = updates.check(
            __version__,
            client=client,
            include_prereleases=include,
            api_url=settings.updates.api_url,
        )
    finally:
        client.close()
    typer.echo(f"current:  {APP_NAME} {result.current}")
    if result.status == updates.UNKNOWN:
        reason = {
            updates.NETWORK: "could not reach GitHub",
            updates.RATE_LIMITED: "GitHub turned the request down for now (rate limit)",
        }.get(result.error or "", "unexpected answer from GitHub")
        if not result.error:
            reason = f"the running version {result.current!r} cannot be compared"
        typer.echo(f"latest:   unknown ({reason})")
        raise typer.Exit(UPDATE_UNKNOWN)
    if result.latest is None:
        typer.echo("latest:   no release published yet")
        raise typer.Exit(UPDATE_CURRENT)
    published = f" (released {result.published})" if result.published else ""
    typer.echo(f"latest:   {APP_NAME} {result.latest_version}{published}")
    typer.echo(f"release:  {result.release_url}")
    if result.status == updates.AVAILABLE:
        if result.download_url:
            typer.echo(f"download: {result.download_url}")
        else:
            typer.echo("download: see the release page for the files for this computer")
        typer.echo("An update is available: run the new setup (your library is kept and migrated).")
        raise typer.Exit(UPDATE_AVAILABLE)
    typer.echo("Up to date.")
    raise typer.Exit(UPDATE_CURRENT)


db_app = typer.Typer(
    help="Export, import or clear the whole library database (backup, another computer).",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


@db_app.command("export")
def db_export(
    target: Annotated[
        Path,
        typer.Argument(help="Target file or folder (default: vision-tagger-<date>.sqlite3 here)."),
    ] = Path(),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a consistent copy of the database (works while Mira and the pipeline run)."""
    from .db import transfer

    if target.is_dir():
        target = target / transfer.export_name()
    if target.exists() and not force:
        typer.echo(f"error: {target} exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    db = _db()
    tmp = target.with_name(f".{target.name}.tmp")
    transfer.remove_file(tmp)
    try:
        transfer.export_database(db, tmp)
        tmp.replace(target)
    except transfer.TransferError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    finally:
        transfer.remove_file(tmp)
        db.dispose()
    typer.echo(f"exported {target} ({target.stat().st_size / 1024**2:.1f} MB)")


@db_app.command("import")
def db_import(
    source: Annotated[Path, typer.Argument(help="Database file exported by Mira.")],
) -> None:
    """Replace the library with an exported database (a backup of the current one is kept).

    Close the web UI first. Afterwards the location of every photo is checked; photos that are
    not found are flagged as missing until a scan of their new folder links them again.
    """
    from .db import transfer
    from .indexer import Indexer

    if not source.is_file():
        typer.echo(f"error: not a file: {source}", err=True)
        raise typer.Exit(1)
    settings = _settings()
    db = _db()
    try:
        reason = transfer.busy_reason(db)
        if reason is not None:
            typer.echo(f"error: {reason}", err=True)
            raise typer.Exit(1)
        tmp = transfer.temp_file(settings, "import")
        shutil.copyfile(source, tmp)  # the given file is never changed

        def before_replace() -> None:
            busy = transfer.busy_reason(db)
            if busy is not None:
                raise transfer.TransferError(busy)

        try:
            result = transfer.import_database(
                db,
                settings,
                tmp,
                before_replace=before_replace,
                progress=lambda message: typer.echo(f"  {message}"),
            )
        except transfer.TransferError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
        finally:
            transfer.remove_file(tmp)
        typer.echo(f"imported {result.images} photos (schema {result.revision})")
        if result.backup:
            typer.echo(f"backup:  {result.backup}")
        typer.echo("checking where the photos are…")
        missing = Indexer(db, settings).check_locations()
        typer.echo(f"missing: {missing} of {result.images} photos not found on this computer")
        if result.roots:
            typer.echo(f"library folders: {', '.join(result.roots)}")
        if missing:
            typer.echo(
                "scan the folders where the photos are now (`vision-tagger index FOLDER`): "
                "they are linked again by perceptual hash, with their faces and properties"
            )
    finally:
        db.dispose()


@db_app.command("clear")
def db_clear(
    yes: Annotated[
        int,
        typer.Option(
            "--yes",
            "-y",
            count=True,
            help="Answer a confirmation; give it twice to answer both (scripts).",
        ),
    ] = 0,
) -> None:
    """Delete everything Mira knows about the photos and start with an empty library.

    Deleted: all photos' data, faces, persons and their names, pets, properties, collections,
    places, the search index and the history. Kept: the photo files (never touched), the
    library folders (`vision-tagger index` builds the library again) and the settings. A backup
    of the database is written first. Close the web UI first. Asks twice: a yes/no question,
    then the number of photos in the library.
    """
    from .db import transfer

    settings = _settings()
    db = _db()
    try:
        reason = transfer.busy_reason(db)
        if reason is not None:
            typer.echo(f"error: {reason}", err=True)
            raise typer.Exit(1)
        n = transfer.library_contents(db)
        typer.echo("This deletes from Mira:")
        typer.echo(
            f"  {n['images']} photos (index, dates, places, analysis), {n['faces']} faces in "
            f"{n['clusters']} clusters, {n['persons']} persons with their names, "
            f"{n['animals']} animals and {n['pets']} named pets, {n['properties']} properties, "
            f"{n['collections']} collections, {n['places']} places, {n['search']} search index "
            f"entries, {n['runs']} pipeline runs"
        )
        backup = settings.db_path.with_name(f"{settings.db_path.name}.bak-clear-<time>")
        typer.echo("Kept: the photo files (never touched), the library folders and the settings.")
        typer.echo(f"A backup of the database is written first: {backup}")
        if yes < 1 and not typer.confirm("Delete all of this from Mira?", default=False):
            typer.echo("Nothing was deleted.")
            raise typer.Exit(1)
        if yes < 2:
            typed = typer.prompt(f"Type the number of photos ({n['images']}) to confirm")
            if transfer.typed_number(typed) != str(n["images"]):
                typer.echo("The number does not match. Nothing was deleted.", err=True)
                raise typer.Exit(1)

        def before_replace() -> None:
            busy = transfer.busy_reason(db)
            if busy is not None:
                raise transfer.TransferError(busy)

        try:
            result = transfer.clear_library(
                db,
                settings,
                before_replace=before_replace,
                progress=lambda message: typer.echo(f"  {message}"),
            )
        except transfer.TransferError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
        typer.echo(f"cleared: {result.images} photos removed from Mira, no file was touched")
        if result.backup:
            typer.echo(f"backup:  {result.backup}")
        roots = result.roots if result.roots is not None else settings.paths.image_roots
        if roots:
            typer.echo(f"library folders: {', '.join(str(r) for r in roots)}")
            typer.echo("`vision-tagger index` builds the library again")
    finally:
        db.dispose()


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
                + (
                    f" · thumbnails {stats.thumbnails_warmed}/{stats.thumbnails_todo}"
                    if stats.thumbnails_todo
                    else ""
                )
            )

    stats = Indexer(db, _settings(), progress=progress).run(roots or None)
    typer.echo(
        f"done: found {stats.found}, new {stats.new}, duplicates {stats.duplicates}, "
        f"moved {stats.moved}, updated {stats.updated}, unchanged {stats.unchanged}, "
        f"missing {stats.missing}, errors {stats.errors}, thumbnails {stats.thumbnails}"
    )
    if stats.places:
        typer.echo(stats.places)
    for sample in stats.error_samples[:10]:
        typer.echo(f"  ! {sample}", err=True)


@app.command()
def run(
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", "-s", help="faces / ages / animals / vlm / embed (repeatable)"),
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


models_app = typer.Typer(
    help="The Ollama models Mira needs: list them (exit code 1 if some are missing) or "
    "`models pull` the missing ones.",
    invoke_without_command=True,
)
app.add_typer(models_app, name="models")


@models_app.callback()
def models(ctx: typer.Context) -> None:
    """List the Ollama models this configuration needs and whether Ollama has them."""
    if ctx.invoked_subcommand is not None:
        return
    from .i18n import _
    from .vlm.models import ollama_status, unreachable_text

    settings = _settings()
    status = ollama_status(settings)
    if not status.enabled:
        typer.echo(f"vlm.backend = {settings.vlm.backend!r}: no Ollama models needed")
        return
    if not status.reachable:
        typer.echo(f"{unreachable_text(status.url)}\n({status.error})", err=True)
        raise typer.Exit(1)
    typer.echo(f"Ollama {status.version or '?'} at {status.url}")
    width = max(len(m.name) for m in status.required)
    for m in status.required:
        state = "installed" if status.is_installed(m) else "missing"
        purposes = ", ".join(_(p).lower() for p in m.purposes)
        typer.echo(f"  {m.name:<{width}}  {state:<9}  {purposes}")
    missing = status.missing
    if missing:
        typer.echo(
            f"{len(missing)} missing: run `vision-tagger models pull` (a pipeline run that needs "
            "them downloads them too, unless vlm.auto_pull = false)"
        )
        raise typer.Exit(1)
    typer.echo("all models are there")


@models_app.command("pull")
def models_pull() -> None:
    """Let Ollama download the missing models, one after another (Ctrl+C cancels)."""
    from .vlm.models import PullError, PullProgress, phase_text, pull_missing

    settings = _settings()
    tty = sys.stdout.isatty()
    shown: dict[str, Any] = {"phase": None, "step": -1}

    def on_start(index: int, todo: list[str]) -> None:
        typer.echo(f"downloading {todo[index]} ({index + 1}/{len(todo)})")

    def on_progress(index: int, todo: list[str], p: PullProgress) -> None:
        text = f"  {p.model}: {phase_text(p.as_dict())}"
        if tty:  # one line, updated in place
            typer.echo(f"\r{text:<72}", nl=p.phase == "done")
            return
        # a log or a pipe: each phase and every 10 % (a new high: the next layer lowers it)
        step = (p.percent or 0) // 10
        if p.phase != shown["phase"] or step > shown["step"]:
            shown.update(phase=p.phase, step=step)
            typer.echo(text)

    try:
        pulled = pull_missing(settings, on_start=on_start, on_progress=on_progress)
    except KeyboardInterrupt:
        typer.echo("\ncancelled; Ollama keeps what it has downloaded, `models pull` continues")
        raise typer.Exit(130) from None
    except PullError as exc:
        typer.echo(f"\nerror: {exc}", err=True)
        raise typer.Exit(1) from None
    if settings.vlm.backend != "ollama":
        typer.echo(f"vlm.backend = {settings.vlm.backend!r}: no Ollama models needed")
    elif pulled:
        typer.echo(f"downloaded: {', '.join(pulled)}")
    else:
        typer.echo("all models are there")


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
def pets(
    full: Annotated[
        bool, typer.Option(help="Dissolve the automatic groups first (named pets stay).")
    ] = False,
) -> None:
    """(Re-)group the detected animals into pets and update their properties."""
    from .pipeline.stages.animals import group_animals

    typer.echo(group_animals(_db(), _settings(), full=full))


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


@app.command()
def places(
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh", help="Regroup all photos and rename every place you did not name."
        ),
    ] = False,
) -> None:
    """Group GPS-tagged photos into named places (also runs after every index scan)."""
    from .geo.places import find_places

    result = find_places(
        _db(), _settings(), refresh=refresh, progress=lambda m: typer.echo(f"  {m}")
    )
    typer.echo(result.message)


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
    xmp: Annotated[bool, typer.Option(help="Write XMP sidecar files.")] = True,
    embed: Annotated[bool, typer.Option(help="Write XMP into copied JPEG/PNG files.")] = True,
    index: Annotated[
        bool, typer.Option(help="Write vision-tagger.csv / .json into the target.")
    ] = True,
) -> None:
    """Export the images matching a filter (originals are never touched).

    Keywords, caption, people (with face areas) and dates go along as XMP, so other photo
    applications can use them.
    """
    from starlette.datastructures import QueryParams

    from .export import export_images
    from .query import ImageFilter

    settings = _settings()
    metadata = [name for name, on in (("xmp", xmp), ("embed", embed), ("index", index)) if on]
    stats = export_images(
        _db(),
        ImageFilter.from_params(QueryParams(query)),
        target,
        mode=(mode or settings.export.mode),  # type: ignore[arg-type]
        layout=layout,  # type: ignore[arg-type]
        metadata=metadata,
        sidecar_naming=settings.export.sidecar_naming,
    )
    typer.echo(
        f"exported {stats.exported} files to {stats.target} ({stats.skipped} skipped, "
        f"{stats.sidecars} sidecars, metadata in {stats.embedded} copies)"
    )
    for e in stats.errors[:10]:
        typer.echo(f"  ! {e}", err=True)


@app.command()
def sidecars(
    query: Annotated[
        str,
        typer.Option("--filter", "-f", help="Filter as URL query, e.g. 'p=1'. Default: all."),
    ] = "",
) -> None:
    """Write XMP sidecars next to the original photos (the photos are not changed).

    For libraries managed in place by digiKam, darktable, Immich or PhotoPrism. Sidecars
    written by other applications are left alone.
    """
    from starlette.datastructures import QueryParams

    from .export import write_sidecars_in_place
    from .query import ImageFilter

    stats = write_sidecars_in_place(
        _db(),
        ImageFilter.from_params(QueryParams(query)),
        naming=_settings().export.sidecar_naming,
    )
    typer.echo(
        f"{stats.sidecars} sidecars written, {stats.kept} of other apps kept, "
        f"{stats.skipped} photos missing"
    )
    for e in stats.errors[:10]:
        typer.echo(f"  ! {e}", err=True)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for, in any language.")],
    deep: Annotated[bool, typer.Option("--deep", help="Let the LLM re-rank and explain.")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Natural-language image search (what photos show + LLM interpretation + semantic index)."""
    from .search.engine import SearchEngine

    db = _db()
    result = SearchEngine(db, _settings()).search(query, deep=deep, limit=limit)
    it = result.interpretation
    parts = [
        f"mode={it.mode}",
        f"visual={'on' if result.visual_used else 'off'}",
        f"semantic={'on' if result.semantic_used else 'off'}",
    ]
    who = [n for _, n in [*it.persons, *it.pets]]
    if who:
        parts.append("persons=" + ("|" if it.who_any else ",").join(who))
    if it.must or it.should:
        parts.append("props=" + ",".join(f"{t.key}:{t.value}" for t in [*it.must, *it.should]))
    for group in it.any_of:
        parts.append("one of=" + "|".join(f"{t.key}:{t.value}" for t in group))
    if it.exclude:
        parts.append("not=" + ",".join(f"{t.key}:{t.value}" for t in it.exclude))
    if it.date_from or it.date_to:
        parts.append(f"dates={it.date_from or ''}..{it.date_to or ''}")
    if it.dates:
        parts.append("dates=" + "|".join(f"{a or ''}..{b or ''}" for a, b in it.dates))
    if it.visual_query and result.visual_used:
        parts.append(f"looking for={it.visual_query!r}")
    typer.echo(" ".join(parts))
    _print_hits(db, result)


def _print_hits(db, result) -> None:
    from sqlalchemy import select

    from .db.models import Image
    from .db.util import chunked

    for note in result.notes:
        typer.echo(f"  note: {note}")
    paths: dict[int, str] = {}
    with db.session() as s:
        for part in chunked([h.image_id for h in result.hits]):
            paths.update(s.execute(select(Image.id, Image.path).where(Image.id.in_(part))).all())
    for rank, h in enumerate(result.hits, 1):
        line = f"{rank:>3}. {h.score:.2f}  {paths.get(h.image_id, h.image_id)}"
        if h.reason:
            line += f"  - {h.reason}"
        typer.echo(line)
    if not result.hits:
        typer.echo("no matches")


@app.command()
def similar(
    image: Annotated[str, typer.Argument(help="Image id, or the path of an indexed photo.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Photos that look like the given one (visual search; run the 'visual' stage first)."""
    from sqlalchemy import select

    from .db.models import Image
    from .indexer import path_key
    from .search.engine import SearchEngine

    db = _db()
    with db.session() as s:
        image_id = int(image) if image.isdigit() else None
        if image_id is None:
            image_id = s.scalar(select(Image.id).where(Image.path_key == path_key(image)))
        if image_id is None or s.get(Image, image_id) is None:
            typer.echo(f"no indexed photo {image!r}", err=True)
            raise typer.Exit(1)
    result = SearchEngine(db, _settings()).similar(image_id, limit=limit)
    _print_hits(db, result)


@app.command()
def growup(
    person: Annotated[str, typer.Argument(help="Person name.")],
    output: Annotated[Path, typer.Argument(help="Target file (.webp).")] = Path("growing-up.webp"),
) -> None:
    """Render an animation of a person growing up (eye-aligned faces, ordered by age)."""
    from .access import checker
    from .faces.clustering import find_person
    from .faces.generations import growup_animation

    settings = _settings()
    db = _db()
    with db.session() as s:
        p = find_person(s, person)
        if p is None:
            typer.echo(f"unknown person: {person}", err=True)
            raise typer.Exit(1)
        result = growup_animation(
            s, settings.cache_dir / "aligned", p.id, allowed=checker(settings)
        )
    if result is None:
        typer.echo(f"need photos of {p.name} at two or more different ages", err=True)
        raise typer.Exit(1)
    data, frames = result
    output.write_bytes(data)
    typer.echo(f"wrote {output} ({len(frames)} ages: {', '.join(f.label for f in frames)})")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = False,
    remote: Annotated[
        bool, typer.Option("--remote", help="Remote mode: photos are uploaded via the web UI.")
    ] = False,
    upload_dir: Annotated[
        Path | None, typer.Option(help="Remote mode: folder where uploaded photos are stored.")
    ] = None,
    migrate_db: Annotated[
        bool,
        typer.Option("--migrate", help="Upgrade the database first if needed (with a backup)."),
    ] = False,
) -> None:
    """Start the web UI (local mode, or remote mode with --remote --upload-dir PATH)."""
    import uvicorn

    from .web.app import create_app
    from .web.security import exposure_problem

    settings = _settings()
    if remote:
        settings.mode = "remote"
    if upload_dir is not None:
        settings.remote.upload_dir = upload_dir
    if settings.remote_mode:
        try:
            settings.upload_dir  # noqa: B018 - validates that a folder is configured
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(2) from exc
    host = host or settings.web.host
    port = port or settings.web.port
    settings.web.host = host  # the app accepts the address it listens on as a Host name
    problem = exposure_problem(settings.remote_mode, host, settings.web.password)
    if problem:
        typer.echo(
            f"Error: {problem} Set a password with the environment variable VT_WEB__PASSWORD.",
            err=True,
        )
        raise typer.Exit(2)
    if migrate_db:
        from .db.migrate import upgrade_if_needed

        settings.ensure_dirs()
        if upgrade_if_needed(settings.db_path):
            typer.echo("upgraded the database (a backup copy was made first)")
    _db().dispose()  # fails early with a clear message if `vision-tagger migrate` is needed
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    typer.echo(f"vision-tagger UI on http://{host}:{port}/")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


@app.command()
def desktop(
    port: Annotated[
        int | None, typer.Option(help="Port (default: web.port, or any free one).")
    ] = None,
) -> None:
    """Open the app in a window of its own (local mode). Closing it stops the server."""
    from . import config as config_mod
    from .desktop import load_desktop_settings, run

    # without a config file: the per-user config, as in the installed app
    settings = _settings() if config_mod._resolve_config_file() else load_desktop_settings()
    raise typer.Exit(run(settings, port=port))


@app.command("self-test", hidden=True)
def self_test() -> None:
    """Check an installation or a frozen build (imports, resources, migrations, pages)."""
    from .desktop import self_test as run_self_test

    raise typer.Exit(run_self_test())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
