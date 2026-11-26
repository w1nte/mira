# Changelog

Short, user-facing notes in English: what's new, what changed, which bugs were fixed.
Mira shows them on its "What's new" page, and each GitHub release uses its section as notes.

## Unreleased

### New
- A "What's new" page lists the changes of every version; it is linked in the sidebar and next to the version on the Pipeline page.

## 1.3 - 2026-11-26

### New
- Mira downloads the AI models it needs itself (Home → Download now); no `ollama pull` needed.
- The Environment card on the Pipeline page lists every AI model with its state, a Download button and the progress.
- Pipeline runs download a missing model before they start (setting `vlm.auto_pull`, on by default).
- Home tells you when Ollama is not installed or not running, with a link to [ollama.com](https://ollama.com/download).
- `vision-tagger models` lists the AI models, and `vision-tagger models pull` downloads the missing ones.

## 1.2.1 - 2026-11-26

### Fixed
- The Windows setup programs are built again; release 1.2 came without them.

## 1.2 - 2026-11-26

### New
- Desktop apps for Linux and macOS (Apple silicon) next to Windows, attached to every GitHub release.
- The installers update an installed Mira: they quit it (a running pipeline pauses), keep the library and migrate it with a backup.
- Database page: export the library database and import it on another computer (also `vision-tagger db export` and `db import`).
- Missing files page: photos Mira can't find any more; scan their new folder and they are linked again, with faces, names and properties.

### Changed
- `config.toml` is updated automatically when a new version changes its settings, after a backup.
- The Pipeline page lists every stage; stages switched off in `config.toml` are unticked and can still be run.
- A database from a newer Mira version is left untouched: Mira refuses to open it instead of migrating it.
- Missing photos show a placeholder instead of a broken image.

### Fixed
- Config files from 1.0 no longer keep newer stages (animals, search index) switched off.

## 1.1 - 2026-11-26

### New
- Animals and pets: Mira finds animals in your photos, groups pictures of the same animal and lets you name them ("Bello at the beach").
- A thin loading bar at the top shows when a page takes a moment to load.

### Fixed
- The sidebar no longer scrolls sideways when a job shows a long file path or a menu label is long.
- Docker on Windows: set the password by copying `.env.example` to `.env`; the old `echo > .env` wrote a file Docker couldn't read.

## 1.0 - 2026-11-26

### New
- First release of Mira: it indexes your photo folders with recording dates and duplicate detection, and never changes the originals.
- Finds and groups faces, so you can name the people in your photos, also as a "Who's that?" quiz.
- Describes every photo with a local vision model (via Ollama) as filterable properties, and searches them in plain language.
- Estimates ages, flags doubtful dates and shows people growing up (Generations).
- Groups geotagged photos into places and shows them on a map.
- Exports photos with their metadata (XMP) for other apps.
- A web UI in 11 languages that installs as an app, a Windows desktop app, and a remote mode with Docker.
