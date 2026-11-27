# CLAUDE.md

Guidance for Claude Code (and humans) working on this repository. Read
[`MEMORY.md`](MEMORY.md) too: decisions and lessons from earlier work. Keep it short and
current: add what will matter again, delete what no longer holds.

## Project

Mira (the package, CLI and repository are still called vision-tagger) is a local photo
pipeline:

1. index (pHash/dHash dedup, EXIF dates),
2. faces (InsightFace + clustering),
3. properties (Qwen-VL via Ollama),
4. search index (text embeddings).

It has a FastAPI + Jinja2/HTMX web UI and a typer CLI. Data lives in SQLite through SQLAlchemy 2.0. The main target platform is **Windows + NVIDIA GPU**; CI runs on Ubuntu and Windows.

Key places:
- `src/vision_tagger/db/models.py`: ORM models (the schema)
- `src/vision_tagger/migrations/`: Alembic migrations (see below)
- `src/vision_tagger/db/migrate.py`: migration runner used by `vision-tagger migrate`
- `src/vision_tagger/pipeline/`: batched, resumable runner and the stages
- `src/vision_tagger/web/`: routes, templates, static assets (HTMX and Alpine are vendored)
- `tests/`: pytest suite (fake VLM backend, real InsightFace in `integration` tests)
- `docs/`: the user documentation (the README stays short: description, overview, setup)

## Commands

```bash
uv sync --extra cpu --group dev        # or --extra gpu on a CUDA machine
uv run ruff check . && uv run ruff format --check .
uv run pytest -v
uv run vision-tagger serve             # web UI on http://127.0.0.1:8765
uv run vision-tagger migrate           # upgrade a database to the current schema
```

Before every commit, `ruff check`, `ruff format --check` and the full `pytest` run must be green.

## Database migrations (mandatory)

**Every change to the database schema must ship with a migration script in the same commit.**

A schema change is any change in `db/models.py` that affects tables: new, renamed or removed
tables or columns, type or nullability changes, defaults, indexes, unique, foreign-key or
primary-key constraints. Users keep their databases across versions. A model change without a
migration makes the app refuse to start, or corrupts existing data.

### How to add a migration

1. Create a scratch database at the current head, *before* changing the models:
   ```bash
   uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 upgrade head
   ```
2. Change `db/models.py`.
3. Autogenerate a revision with the next sequential id (the newest existing revision + 1):
   ```bash
   uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 revision --autogenerate --rev-id 0003 -m "short description"
   ```
   This creates `src/vision_tagger/migrations/versions/0003_short_description.py`.
4. Review and edit the generated file. Autogenerate is a draft, not the answer:
   - Write a meaningful docstring: what changed and why.
   - Implement `downgrade()` where it is possible.
   - Data transformations (backfills, splitting columns) go in with `op.execute(...)` or
     `sa.table(...)` constructs. **Never import application models in a migration**,
     because migrations must stay frozen in time.
5. Verify, then delete the scratch database:
   ```bash
   uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 upgrade head
   uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 check   # must report no changes
   uv run pytest tests/test_migrations.py -v
   ```
6. For non-trivial migrations (data changes, table rebuilds) add a test to
   `tests/test_migrations.py`. Build a database at the previous revision with
   `build_at(...)`, insert data, upgrade, and assert the data survived.

### Rules

- **Never edit or delete a committed revision.** Fix mistakes with a new revision. `0001` is the
  frozen schema of the first release (commit `c1e357b`).
- Revision ids are sequential (`0001`, `0002`, …), and each revision's `down_revision` is the
  previous head. Keep exactly one head.
- SQLite specifics:
  - Use `op.batch_alter_table(...)` for altering or dropping columns and constraints. The
    environment sets `render_as_batch=True`, so autogenerate already does this.
  - A new `NOT NULL` column needs a `server_default` (in the migration *and* the model), or
    existing rows can't be migrated.
  - Foreign keys are switched off while migrations run, so table rebuilds don't
    cascade-delete child rows. `PRAGMA foreign_key_check` runs afterwards. Don't turn them back
    on inside a migration.
- All pending revisions run in **one transaction**, so a failing migration leaves the database
  unchanged. `vision-tagger migrate` also writes a backup copy (`*.sqlite3.bak-<rev>-<time>`)
  first.
- Databases created before migrations existed (no `alembic_version` table) are treated as
  revision `0001`. That's why `0002` checks the existing state before each step. New revisions
  normally don't need such checks.
- `tests/test_migrations.py::test_migrations_match_models` fails whenever models and migrations
  disagree. Never skip or weaken it; add the missing migration instead.
- New databases are created from the models (`create_all`) and stamped with the head revision.
  Existing outdated databases are refused with a message to run `vision-tagger migrate`.

## Config migrations (mandatory)

Users keep their `config.toml` across versions too. `vision-tagger init` and the desktop app
write it from `resources/config.example.toml`, so whatever the example sets is frozen in their
files.

- A change that makes existing config files wrong or stale ships with a config migration in the
  same commit. That covers a renamed, moved or removed key, a changed meaning or unit, or a
  default users have pinned because the example config set it.
- Steps live in `src/vision_tagger/config_migrate.py`:
  1. Write `_vN(doc)`, which edits the tomlkit document in place and returns a note or `None`.
  2. Register it in `STEPS`.
  3. Bump `CONFIG_VERSION`.
  4. Set `config_version = N` in the example config.
  5. Add a test to `tests/test_config_migrate.py`.
- Files without `config_version` are version 1. Loading runs the pending steps and rewrites the
  file atomically after a backup (`config.toml.bak-v<old>-<time>`). A file that can't be written
  is migrated in memory.
- Never edit a released step, and don't import settings classes in a step: steps are frozen
  in time, like database migrations.
- Prefer leaving settings out of the example config (commented out) when their default may
  change; then nothing needs migrating.

## Changelog (mandatory)

The changelog is [`src/vision_tagger/resources/CHANGELOG.md`](src/vision_tagger/resources/CHANGELOG.md)
(the root `CHANGELOG.md` only points there: the file ships with the app). Mira shows it on its
"What's new" page (`/changelog`), and the desktop workflow uses it as GitHub release notes.

**Every user-visible change adds one short English line under `## Unreleased` in the same
commit**: a new feature under `### New`, a change in behaviour under `### Changed`, a bug fix
under `### Fixed`. Internal refactors, tests, CI-only and documentation-only changes don't get an
entry. A change that spans several commits gets one entry, not one per commit.

Format (`changelog.validate()` enforces it):
- `# Changelog`, an introduction, then `## Unreleased` (always first, possibly empty), then the
  releases newest first as `## <tag> - <YYYY-MM-DD>`, e.g. `## 1.2.1 - 2026-11-26`. The tag is a
  release tag as `packaging/version.py` accepts it (`1.3`, `v1.3`, `1.4-rc.1`).
- Only the sections `### New`, `### Changed` and `### Fixed`, in that order, each optional, never
  empty.
- One line per item, `- text`, at most 160 characters. Inline code and `[text](https://…)` links
  are the only markup; no HTML, no nested lists, no line continuations.

Style: write for users, not developers. Say what they can do or notice now, in the present
tense, and name the place in the UI or the command. Leave out internals (modules, tables, PR
numbers).
- Good: ``- Mira downloads the AI models it needs itself (Home → Download now); no `ollama pull` needed.``
- Good: `- The sidebar no longer scrolls sideways when a job shows a long file path.`
- Bad: `- Add vlm/models.py with pull_missing() and a models job` (internal, not what users see).
- Bad: `- Fixed bug` (which one?), `- Refactor clustering` (not user-visible: no entry).

Releasing:
1. Just before publishing a GitHub release, rename `## Unreleased` to `## <tag> - <date>` (the
   tag you are about to create, today's date) and add a fresh, empty `## Unreleased` above it.
   Commit that to `main`.
   `python packaging/changelog.py release <tag>` does exactly this.
2. Publish the release with that tag on that commit. When its notes are left empty, the publish
   job of `.github/workflows/desktop.yml` fills them with the tag's section
   (`python packaging/changelog.py notes <tag>`). Notes written on GitHub are kept as they are.
3. Once the files are attached, `.github/workflows/virustotal.yml` appends a "Downloads" section
   to the notes: each file's SHA-256 and VirusTotal link (`packaging/virustotal.py`, secret
   `VIRUSTOTAL_API_KEY`). Run it by hand with the tag to scan again; it replaces its section.

If step 1 was forgotten, the workflow catches it: every build files `Unreleased` under the tag
before packaging (so the app's "What's new" shows the version), the notes come from that section,
and the publish job opens a pull request "Changelog: release <tag>" against `main` that moves only
the entries of the released commit (newer ones stay under `Unreleased`). Merge it. Creating that
pull request needs "Allow GitHub Actions to create and approve pull requests" (Settings → Actions →
General); without it the job pushes the branch `changelog/release-<tag>` and warns, so merge that
branch instead.

`tests/test_changelog.py` validates the file (a broken format fails CI) and tests the parser,
the notes script, the page and the workflow step. Check the file locally with
`python packaging/changelog.py check`.

## Code conventions

- Names: users see "Mira" (`vision_tagger.APP_NAME`, `app_name` in templates). Technical names
  stay `vision-tagger`: package, command, `VT_` settings, cookies, data folder, XMP keyword root
  and export file names, so existing data, scripts and exports keep working.

- Windows first:
  - use `pathlib`; uniqueness keys use `os.path.normcase`;
  - store relative folders with `/`;
  - don't rely on symlinks;
  - tolerate locked files (antivirus, viewers).
- SQLite:
  - keep write transactions short (`db.write()` uses `BEGIN IMMEDIATE`) and never hold one
    during model inference, DBSCAN or disk scans;
  - chunk `IN (...)` lists built from Python with `db.util.chunked` (parameter limit);
  - escape user input in `LIKE` with `db.util.like_escape`.
- Logging (`logs.py`): every module logs through `logging.getLogger(__name__)` into the log file
  users attach to bug reports. Log what explains an error later: actions and their results at
  INFO, problems at WARNING/ERROR with the traceback (`exc_info`) once, not per photo. In
  English, never passwords or keys; nothing per request or per photo at INFO (the log is
  capped at 20 MB).
- After bulk `update()`/`delete()` statements in a session that also holds ORM objects, flush
  and expire them. See the `_synced` decorator in `faces/clustering.py`.
- Never discard user work automatically: locked clusters, manual face assignments and manually
  set dates must survive re-clustering, re-detection and re-indexing.
- Web UI:
  - state-changing endpoints are POST, protected by the Host/Origin middleware
    (`web/security.py`);
  - never put user data into inline JavaScript; use `data-*` attributes;
  - redirect with `common.redirect(request, back(request, …))`.
- Translations (`i18n.py`, catalogs in `resources/i18n/<lang>.json`):
  - every user-visible text goes through `_()` / `ngettext()` (templates and Python); in
    templates a literal `%` is written `%%`;
  - text made outside a request (background jobs) or stored in shared caches stays
    language-neutral: `Lazy(...)` or `N_()` and translate at render time;
  - after adding or changing UI text run `python -m vision_tagger.i18n_extract update` and fill in
    every catalog; `tests/test_i18n.py` checks completeness and placeholders.
- Tests:
  - every bug fix or feature gets a test;
  - tests must not need network access or a GPU: use `FakeBackend` / `FakeAnalyzer`;
  - real InsightFace runs only in `integration` tests;
  - real Ollama only in `ollama` tests, which are skipped unless `VT_OLLAMA_URL` is set.
