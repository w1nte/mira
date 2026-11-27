# Development

```bash
uv sync --extra cpu --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

- The suite includes integration tests that run the real InsightFace model. It is downloaded to
  `~/.insightface` on first use.
- Tests marked `ollama` run against a real server when `VT_OLLAMA_URL` (and optionally
  `VT_OLLAMA_MODEL`) is set.
- Set `vlm.backend = "fake"` to try the whole UI without a GPU or Ollama.

GitHub Actions ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs ruff and the full
test suite on Ubuntu and Windows.

**Translations.** UI text is English in the code: `_("…")` / `ngettext(…)` in templates and
Python (see [`i18n.py`](../src/vision_tagger/i18n.py)); the catalogs are
`src/vision_tagger/resources/i18n/<lang>.json`. After adding or changing UI text:

```bash
uv run python -m vision_tagger.i18n_extract update       # adds new strings to every catalog
uv run python -m vision_tagger.i18n_extract check de -v  # what is missing or broken in one language
```

`tests/test_i18n.py` fails while a catalog misses a string or a translation breaks a placeholder.

Schema changes always need an Alembic migration in `src/vision_tagger/migrations/versions/`.
See [CLAUDE.md](../CLAUDE.md#database-migrations-mandatory) for the workflow.
`tests/test_migrations.py` fails when the models and the migrations disagree.

## Releases

Publishing a GitHub release builds the desktop apps and attaches them to it, in about an hour:

1. In the [changelog](../src/vision_tagger/resources/CHANGELOG.md), rename `## Unreleased` to `## X.Y.Z - <today>` (the
   version of the tag in step 2), add an empty `## Unreleased` above it and commit that to
   `main` (see [CLAUDE.md](../CLAUDE.md#changelog-mandatory)).
2. On GitHub, *Releases → Draft a new release*, create a tag `vX.Y.Z` (pre-releases:
   `vX.Y.Z-rc.1`, `-beta.2`, `-alpha.1`) on `main`, and publish it (a pre-release counts too;
   a draft does not). Leave the notes empty to get the changelog's section as notes.
3. [`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml) builds Windows (cpu, gpu),
   Linux (cpu, gpu) and macOS (Apple silicon) with the tag as the version
   ([`packaging/version.py`](../packaging/version.py): `v1.2.3-rc.1` → `1.2.3rc1`), runs each
   app's self-test, packs the installers and uploads them to the release. A release without
   notes gets the changelog's section for the tag ([`packaging/changelog.py`](../packaging/changelog.py)).
4. Then [`.github/workflows/virustotal.yml`](../.github/workflows/virustotal.yml) downloads the
   attached files, has [VirusTotal](https://www.virustotal.com) scan them and appends a
   *Downloads* section to the notes: each file's SHA-256 checksum and a link to its scan results
   ([`packaging/virustotal.py`](../packaging/virustotal.py); files over 650 MB are only hashed).
   It needs the repository secret `VIRUSTOTAL_API_KEY` (*Settings → Secrets and variables →
   Actions*), the API key of a VirusTotal account; a free one will do. Without it the job only
   warns. To scan a release again, run *Actions → VirusTotal → Run workflow* with its tag.
5. If a build fails, re-run the failed jobs; the upload replaces files of the same name, and the
   scan replaces its section in the notes.

The version in `pyproject.toml` is only used for builds without a release (manual runs of the
workflow, and the cpu builds it makes as a check on pull requests and pushes to `main` that touch
the packaging). Build locally with:

```bash
uv sync --extra cpu --extra desktop --group packaging
uv run pyinstaller packaging/vision-tagger.spec --noconfirm --clean
dist/vision-tagger/vision-tagger-cli self-test
packaging/linux/build_tarball.sh 0.0.0 cpu     # Linux; macOS: packaging/macos/build_pkg.sh 0.0.0
```
