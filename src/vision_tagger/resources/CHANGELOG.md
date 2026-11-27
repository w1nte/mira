# Changelog

Short, user-facing notes in English: what's new, what changed, which bugs were fixed.
Mira shows them on its "What's new" page, and each GitHub release uses its section as notes.

## Unreleased

### New
- `web.trusted_proxies`: behind a reverse proxy in another container, the login lockout counts each visitor, not the proxy.

### Changed
- `vision-tagger serve` refuses to start without a password in remote mode or on an address other computers can reach (`--host 0.0.0.0`).

### Fixed
- Photos only open image files, and in remote mode only files in the upload folder: an imported database can no longer reveal other files.
- Changing the password now ends every login; Database → Log out everywhere ends them on all devices. Everyone logs in once more after the update.
- Exports and ZIP downloads stay inside their folder, and an imported database can no longer make Mira contact network shares you don't use.
- Login: guessing from many addresses is limited, wrong passwords no longer slow down other pages, and the example password from `.env.example` is refused.
- Text visible in a photo (a sign, a screenshot) can no longer steer how Deep search ranks other photos.
- Uploads and database imports larger than allowed (`remote.chunk_mb`, `remote.max_file_gb`) are refused before they fill the server's memory or disk.
- Other websites can no longer show Mira in a hidden frame to trick you into clicking its buttons.
- Face models are checked (SHA-256) before use, Docker uses fixed versions of Python and Ollama, and releases list the SHA-256 of their files.
- Other websites can link to Mira but no longer load its photos or data in the background (which revealed whether a photo exists).

## 1.7 - 2026-11-26

### New
- Database → Clear the library starts over with an empty library after two confirmations (also `vision-tagger db clear`); photo files stay, a backup is kept.
- Database → Missing files removes photos whose file is gone, one by one or all at once; names of persons and pets stay.
- On your own computer, "Show in folder" opens a photo, a library folder or the database in Explorer, Finder or the file manager.
- Images: a click on a photo shows it large over the gallery; arrow keys or a swipe for the next one, Esc to close, "Open details" for its page.

## 1.6 - 2026-11-26

### New
- NSFW photos show as black tiles until you open them; a local check finds them, and you can mark photos yourself (photo page, gallery → Select).
- Search finds photos by what they show, in any language and right after the import, without Ollama: the new "Visual search" stage (SigLIP 2).
- "Similar photos" on a photo's page lists the photos that look like it; also `vision-tagger similar <id>`.

### Changed
- Search ranks by what photos show first, then their descriptions, properties and words; results that match nothing are left out.

## 1.5 - 2026-11-26

### New
- Search understands "or": "Anna or Ben at the beach", "dogs or cats", "photos from 2019 or 2021", in every UI language, also without an LLM.
- Gallery filters take alternatives: a filter section's "or" switch (or Ctrl+click on a value) gives "beach or mountains"; collections keep them.

### Changed
- Properties page: "Images with any selected" keeps the other filters, e.g. summer and (beach or mountains).

## 1.4 - 2026-11-26

### New
- The installed app tells you when a new Mira is out (Home, sidebar and Pipeline page), with a link to the download; `updates.check = "off"` turns this off.
- A "What's new" page lists the changes of every version; it is linked in the sidebar and next to the version on the Pipeline page.

### Changed
- The Windows gpu setup is a single `.exe` again; it no longer needs a `.bin` file next to it.

### Fixed
- Cancelling a model download on Windows stops it at once instead of waiting until Ollama reports again.

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
