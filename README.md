# Mira

*Formerly vision-tagger. The Python package, the `vision-tagger` command and the repository
keep that name.*

A local pipeline that indexes a large photo collection, finds and clusters faces, and extracts
stable, filterable properties with a local vision-language model (Qwen2.5-VL via Ollama). You
start and control it from a web UI that runs on your computer, or on a server where you upload
your photos (**remote mode**, with a Docker image). The UI can be installed as an app. Originals
are never moved or modified.

```
            ┌──────────────┐   ┌─────────────────────────────┐   ┌──────────────────────────┐
 folders ──▶│ 0 · Index    │──▶│ 1 · Faces (InsightFace)     │──▶│ 2 · Properties (Qwen-VL) │
            │ path, date,  │   │ detect → embed → cluster →  │   │ closed vocabularies,     │
            │ EXIF, pHash  │   │ name via reference DB       │   │ assumptions, work images │
            └──────┬───────┘   └──────────────┬──────────────┘   └────────────┬─────────────┘
                   └──────────── SQLite (state per image & stage) ────────────┘
                                              │
               Web UI: home · import · faces · persons · properties · images · pipeline
```

## Features

- **Index (step 0).** Recursive scan that records path, file name, recording date (EXIF →
  file-name pattern → mtime), a dynamic JSON metadata column (camera, lens, exposure, GPS, …)
  and a perceptual hash (`imagehash.phash`).
  - A file whose pHash is already indexed **and** whose dHash nearly matches (re-encodes,
    resized copies) is **not** added again; it is recorded as a duplicate. Distinct images
    that only collide on the pHash (blank pages, same-template forms) stay separate.
  - Moved files are re-linked, and changed files are re-queued. If an original disappears,
    one of its duplicates takes its place.
  - System folders (`$RECYCLE.BIN`, `System Volume Information`) are skipped, and link loops
    can't trap the scan.
- **Resumable batches.** Every image has a state per stage (`pending / running / done / error /
  skipped`) in the database. Runs claim work in batches and commit after every image.
  - You can **pause, resume, stop** or kill a run at any time; interrupted images are picked up
    on the next start.
  - Errors are retried automatically up to `max_attempts`.
  - Results produced with an older model, prompt or vocabulary show up as *outdated* and can be
    re-queued.
- **Stage 1: faces.** InsightFace (`buffalo_l`) detection plus 512-d embeddings. Clustering uses
  DBSCAN (cosine distance) and **keeps your manual work**:
  - Clusters you named, merged or split are locked and never dissolved by re-clustering.
  - Re-running the stage (e.g. after changing the detector settings) carries names, manual
    moves and removed faces over to the new detections.
  - A **reference DB** (`references/<Person Name>/*.jpg`) names clusters automatically or
    suggests a name.
- **Stage 2: properties.** Qwen2.5-VL (or any Ollama vision model) answers a fixed JSON schema
  built from [`vocabulary.yaml`](src/vision_tagger/resources/vocabulary.yaml).
  - Closed vocabularies, temperature 0, a fixed seed and normalisation (singular forms,
    synonyms, your merges) keep the values stable enough to cluster on later.
  - **Assumed** properties come with a confidence: work related, work category, personal
    memory, likely purpose, sensitive info.
  - A derived `work_image = yes / maybe / no` combines these with file metadata to filter out
    work photos.
- **Natural-language search.** Ask in any language, for example "Anna and Ben at the beach in
  summer 2021, no work photos".
  - The local LLM turns the request into persons, dates and required, preferred or excluded
    properties, validated against your real vocabulary and person list.
  - A text-embedding index (Stage 3) adds semantic similarity over everything extracted per
    image: caption, visible text, all properties, folder and date.
  - **Deep search** lets the LLM double-check the best ~40 results and explain why each fits.
  - Without an LLM, a keyword fallback (English and German) still works.
- **Ages and doubtful dates.** Give a person a birth year, or let Mira estimate one
  from photo dates and face ages.
  - **Face ages from two models.** InsightFace's age model turns babies and children into young
    adults (a 12-year-old easily comes out as 40). Mira adds AgeNet, a small network
    from [face-api](https://github.com/vladmandic/face-api) (MIT license) trained on many
    children and old people, and combines both. AgeNet is used alone for children and teenagers,
    and both count equally from about 35 years on.
  - **Your birth years make it better.** Every birth year you set also shows how far off the
    face model is on *your* photos. Mira learns a correction per age range from those
    faces and applies it to everyone else.
  - **Robust estimate.** The birth year is a weighted median of "photo year − face age". It
    trusts EXIF dates more than file times, faces of children more than those of adults (their
    age is easier to see), and sharp faces more than tiny ones, and it drops outliers.
  - Every face shows the person's age in that photo, and each person gets a "through the years"
    timeline.
  - Photos where the date can't be right (camera clock reset to 2000, scans that only carry the
    file date, "taken" before birth) are flagged with a suggested year and a one-click fix. A
    date is only doubted when the faces disagree far beyond the model's typical error at that
    age; dates you set by hand are never doubted. Fixed dates survive re-scans.
- **"Who's that?" quiz** 🎲. Name unnamed clusters as a game, or test yourself on the people you
  already named.
  - After each answer you see how sure the face recognition was.
  - "That's not X!" pulls misfiled faces out of a cluster, and the quiz offers to move a face
    when the AI agrees with you.
- **Generations** ✨. Watch the people in your photos grow up.
  - **Grow-up timelapse**: one face per age (the sharpest, most frontal one), aligned on the
    eyes and cross-faded from the youngest to the oldest. Scrub through the years with a slider,
    or download it as an animated WebP to share.
  - Ages come from the birth year and the photo date. When a date can't be right, the face's
    estimated age is used instead (shown as "~7 years").
  - **Same age, side by side**: compare two people at matching ages ("Dad at 6 and his son at
    6"), with how similar the faces are.
  - **Looks most like…**: who in the family resembles a person most, compared at matching ages
    where possible.
- **Web UI.**
  - Face cluster overview with inline naming, merge, split, move, auto-split and remove.
  - Persons, including merging persons.
  - Property explorer: combine values with AND/OR, merge/rename/delete values.
  - Gallery with a filter builder (persons AND/OR, properties AND/OR, NOT, date, folder,
    camera, …) and export (copy / hardlink / symlink) **with the metadata for other apps**
    (see [Using the results in other applications](#using-the-results-in-other-applications)).
  - Image detail with face boxes.
  - Timeline / data overview and a metadata table.

## Installation

### Windows app (setup program)

The simplest way on Windows: download the setup program from the
[releases](https://github.com/w1nte/vision-tagger/releases) (or from the artifacts of the
*Windows installer* workflow) and run it. No administrator rights are needed.

- **gpu** for an NVIDIA graphics card (CUDA and cuDNN are included; needs driver R580 or newer).
  It comes as a `.exe` plus `.bin` files: keep them in one folder and start the `.exe`.
- **cpu** for every other PC (slower face detection, much smaller download).

Mira then starts from the Start menu like any other application, in a window of its
own. Closing the window stops it (a running pipeline is paused and continues on the next start);
starting it again while it runs opens another window. Install **Ollama** as well (step 4 below)
for properties and search.

- The library (settings, database, thumbnails) is kept in `%LOCALAPPDATA%\vision-tagger`. It
  stays when you update or uninstall the app; settings are in `config.toml` there.
- `vision-tagger-cli.exe` in the install folder is the command line (`vision-tagger-cli index
  D:\Photos`, `vision-tagger-cli run`, …).
- Logs: `%LOCALAPPDATA%\vision-tagger\data\logs\desktop.log`.

From a source checkout the same window opens with `uv run vision-tagger desktop` (add
`--extra desktop` to `uv sync` for a native window; without it, Edge or Chrome opens in app mode).
The setup program is built by [`.github/workflows/installer.yml`](.github/workflows/installer.yml)
from [`packaging/windows`](packaging/windows): PyInstaller makes the app folder, a self-test
checks it, and Inno Setup packs it.

### From source (Windows + NVIDIA GPU)

1. **uv** (package manager):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Microsoft C++ Build Tools.** `insightface` is built from source. In the
   [Build Tools installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/), select
   *Desktop development with C++*.
3. **A recent NVIDIA driver.** The `gpu` extra installs `onnxruntime-gpu[cuda,cudnn]`, which
   brings the CUDA and cuDNN runtime as pip packages, so a separate CUDA install is not needed.
   Mira calls `onnxruntime.preload_dlls()` so these DLLs are found. Current
   onnxruntime-gpu builds target CUDA 13, which needs driver R580 or newer. If CUDA can't be
   loaded, InsightFace falls back to the CPU and the dashboard shows a warning. Check
   **Environment** on the dashboard.
4. **Ollama** from <https://ollama.com/download>, then pull the model:
   ```powershell
   ollama pull qwen2.5vl:7b      # ~6 GB VRAM; qwen2.5vl:3b for smaller GPUs, qwen3-vl:8b also works
   ollama pull nomic-embed-text  # small text-embedding model for the search index
   ```
5. **Mira:**
   ```powershell
   git clone https://github.com/w1nte/vision-tagger.git
   cd vision-tagger
   uv sync --extra gpu           # Python 3.12 is installed automatically by uv
   uv run vision-tagger init     # creates config.toml and the database
   ```

For CPU-only machines (or Linux/macOS) use `uv sync --extra cpu` and set
`providers = ["CPUExecutionProvider"]` in `config.toml`.

## Upgrading

```powershell
git pull
uv sync --extra gpu
uv run vision-tagger migrate          # upgrades the database schema; writes a backup copy first
```

- If the database is older than the code, Mira refuses to start and tells you to run
  `vision-tagger migrate`.
- All pending migrations run in a single transaction, so a failed migration leaves the database
  as it was.
- The backup is written next to the database as `vision_tagger.sqlite3.bak-<version>-<time>`.
- Databases from the very first version are upgraded too.
- `vision-tagger migrate --check` only reports whether an upgrade is needed.

## Moving a library to another computer (export / import)

Everything Mira knows about your photos (index, faces, clusters, names, properties, collections,
settings) is in one database file. The **Database** page (sidebar, *System*) exports and imports
it; the photos themselves are not part of it, copy them separately.

- **Export** downloads `vision-tagger-YYYY-MM-DD.sqlite3`: a consistent snapshot, made while
  Mira keeps running, also while the pipeline works (`VACUUM INTO`, so it is compact and one
  self-contained file).
- **Import** replaces the current library with such a file.
  - It is refused while the pipeline runs or is paused, or while another job runs: stop the
    pipeline first.
  - The file is checked first (SQLite, not damaged, a Mira database, a schema this version
    knows). Files from older versions are migrated, files from newer versions are refused.
  - A backup of the current database stays next to it: `vision_tagger.sqlite3.bak-import-<time>`.
  - Thumbnails and face crops are cleared (image ids now mean other photos) and made again on
    demand. The pipeline state of the other computer (a run that was active there) is reset.
  - The library folders become the imported folders that exist on this computer plus the ones
    configured before; remote mode keeps its upload folder.
- **Missing files.** After an import Mira checks where every photo is, not only those below
  the current folders. Photos that are not where the other computer had them are flagged as
  missing and listed under **Database → Missing files** (with their faces and names), with a
  **Check again** button. The gallery, search and statistics leave missing photos out, as they
  do for files that vanished; their faces stay in *Faces* and *Persons*.
- **Re-linking.** Add or scan the folder where the photos are now (in remote mode: upload them
  again). A new file whose perceptual hash matches a missing photo (same pHash, dHash within
  `index.dhash_max_distance`) is linked to that photo instead of being added again: same id,
  all faces, names, properties and manual dates kept. The scan result counts these as *moved*.

From the command line (close the web UI first for an import):

```powershell
uv run vision-tagger db export D:\Backup                 # writes vision-tagger-<date>.sqlite3
uv run vision-tagger db import D:\Backup\vision-tagger-2026-11-26.sqlite3
uv run vision-tagger index "E:\Photos"                  # re-links the photos found there
```

## Quick start

```powershell
uv run vision-tagger serve --open          # or: .\start.ps1
```

1. Open **Import** (sidebar, *System*). Add your photo folders with **Add a folder…** (a folder
   browser; you can also paste a path) and click **Scan now**. With *Analyse new photos right
   away* ticked, the pipeline runs afterwards: faces, ages, properties and the search index.
2. Optionally put reference portraits into `references\<Person Name>\` and click
   **Import references** on the **Pipeline** page.
3. **Pipeline** shows the progress per stage, throughput, ETA and errors. There you can also
   choose the stages, a batch size and an optional limit, and **Pause**, **Resume** or **Stop**
   at any time.

Then name and clean up clusters under **Faces**, explore **Properties**, and filter or export in
**Images**. **Home** shows what is in your library and your collections.

**Closing the app pauses a running pipeline** after the current image. The next start continues
it where it stopped, with the same stages and settings. Turn this off with
`pipeline.resume_on_start = false`.

Everything also works headless:

```powershell
uv run vision-tagger index "D:\Photos"
uv run vision-tagger run --stage faces --stage vlm --batch-size 16 --limit 500
uv run vision-tagger status
uv run vision-tagger pause | resume | stop       # from a second terminal
uv run vision-tagger refs                        # import reference DB + name clusters
uv run vision-tagger cluster [--full]
uv run vision-tagger retry-errors --stage vlm
uv run vision-tagger requeue-outdated --stage vlm
uv run vision-tagger migrate [--check]           # database schema upgrade (with backup)
uv run vision-tagger export "D:\Export\Anna at the beach" --filter "p=1&t=location_type:beach" --layout date
uv run vision-tagger sidecars                    # XMP next to the originals (digiKam, Immich, …)
uv run vision-tagger search "Anna am Strand im Sommer, ohne Arbeitsfotos" --deep
uv run vision-tagger growup "Anna" anna-growing-up.webp  # grow-up animation
uv run vision-tagger db export | import FILE     # whole database (see "Moving a library")
```

`Ctrl+C` during `run` stops gracefully. Running `run` again continues where it stopped.

## The web UI

| Page | What you can do |
|---|---|
| **Home** | Your library at a glance: number of photos, people, places and years, *On this day* memories, recently added photos, and collections. Nudges you when face groups still need a name. |
| **Collections** | A collection is a saved filter: it stays up to date as new photos arrive. Suggested ones (people, places, events, this year) appear by themselves; **Save as collection** in *Images* adds your own. Rename, reorder and delete them here. |
| **Import** | Local mode: your photo folders, with a folder browser, and **Scan now**. Remote mode: upload photos, folders or ZIP files (see [Remote mode](#remote-mode-and-docker)). Shows what is being indexed and analysed right now. |
| **Pipeline** | The technical part: start/pause/resume/stop runs (stages, batch size, limit), per-stage progress, ETA, errors (retry/skip), outdated results, jobs, recent runs, environment check (CUDA, Ollama, model). |
| **Database** | Export the whole library database (backup, another computer) and import one (see [Moving a library](#moving-a-library-to-another-computer-export--import)). **Missing files** lists photos that are not where Mira last saw them; scanning their new folder re-links them. |
| **Faces** | All clusters with cover face and size. Name inline (autocomplete), accept reference suggestions, select several to **merge**. Views for unclustered and ignored faces. |
| **Cluster** | All faces of a cluster. Select faces to **split** into a new cluster, **assign to a person**, **move** to another cluster, or **remove**. **Auto-split** mixed clusters. Similar clusters with one-click merge. |
| **Persons** | Named persons. Rename (renaming to an existing name merges), merge, delete, manage references. Images with *all* / *any* of the selected persons. Per person: birth year (set or estimated), age today, a "through the years" face timeline, and photos with a doubtful date with suggested fixes. |
| **Properties** | Every extracted property grouped by topic, with counts. Tick values and show images with **all** / **any** of them. Per property: sample images per value, **merge / rename / delete** values (remembered for future results). |
| **Images** | **Timeline slider** above the gallery: photos per month; click the first month, then the last, drag the band or its edges, or sweep across the bars (arrow keys move the handles). Gallery with infinite scroll and a filter builder: persons (AND/OR, NOT), properties (AND/OR, NOT), date range, folder, camera, faces, text search, sort. Facets refine the current result. **Export** the result, with XMP metadata for other apps. Filters are part of the URL, so they can be bookmarked. |
| **Image** | Full image with face boxes and names, all properties with source and confidence, metadata, duplicates, raw VLM JSON, per-stage status and re-run buttons. `←` / `→` open the previous / next photo of the current filter. |
| **Quiz** | "Who's that?": name the largest unnamed clusters with one click (suggestions from references and look-alikes), or play the face quiz with score, streaks and "you vs. the AI"; wrong answers can fix misfiled faces. Keys 1–4 and Enter. |
| **Generations** | Every person with photos from different ages. Per person: a grow-up timelapse (play, scrub, download as animated WebP), "looks most like…" ranking, and a same-age comparison with any other person. |
| **Search** | Search box in the top bar on every page (`/` focuses it). Shows how the request was understood (persons, required / preferred / excluded properties, dates, keywords), ranks the results and, with **Deep search**, gives the LLM's reason for each. "Open in filter builder" turns the strict part into a gallery filter; results can be exported. |
| **Overview** | Timeline (year → month → day), breakdowns by folder, camera, file type, date reliability, work/personal, orientation, and a sortable metadata table. |

The sidebar groups the pages into *Library*, *People*, *Organize* and *System* and shows what
the pipeline is doing right now. On small screens it folds away behind the menu button.

**Languages.** The UI speaks English, German, French, Spanish, Italian, Portuguese, Dutch,
Polish, Russian, Chinese and Japanese. It starts in your browser's language; the language menu at
the bottom of the sidebar changes it (kept in the `vt_lang` cookie). Property names and values
are translated too, while filters, exports and the database keep the English values, so links
and exported keywords stay the same in every language. Searches can be typed in any language.

**Light and dark mode.** The switch at the bottom of the sidebar picks the theme: *System*
(follows your operating system), *Light* or *Dark*. The choice is kept in a cookie (`vt_theme`),
so every page is rendered in the right theme from the start, without flashing.

**Install as an app.** Mira is a Progressive Web App. In Chrome or Edge, click
**Install app** in the sidebar (or the install icon in the address bar); on Android use *Add to
home screen*, on iOS *Share → Add to Home Screen*. It then opens in its own window, with its own
icon. Browsers allow this on `localhost` and over HTTPS, so a server needs HTTPS (see below).
Without a connection the app shows an offline page instead of a browser error.

## Remote mode and Docker

In **remote mode** Mira runs on a server (a NAS, a home server, a cloud machine) and
you upload your photos through the web UI. Uploaded photos are indexed and analysed
automatically.

- **Uploads** on the *Import* page: drop photos, whole folders or ZIP files, or pick them. Big
  files are sent in 8 MB pieces, and an interrupted upload continues where it stopped.
- **Nothing gets lost when you close the page.** The queue is kept in the browser (IndexedDB).
  Open the page again, in the same browser, and the remaining uploads continue.
- Folder structure is kept: uploaded folders and ZIP contents land under the upload folder with
  their own paths, single photos in a folder named after the day. Files with the same name are
  never overwritten.
- ZIP files are unpacked on the server. Entries can't escape the upload folder (`../` paths are
  placed inside it), "ZIP bombs" are rejected, and non-photo files are skipped.
- *Images → Download ZIP* replaces the folder export: you get the photos with XMP sidecars and
  the index files as one ZIP, streamed while it is created.
- In remote mode the UI never shows or browses the server's own folders.

### Docker

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Open `.env` in a text editor and choose a password (`VT_PASSWORD=…`), then:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5vl:7b
docker compose exec ollama ollama pull nomic-embed-text
```

Then open `http://<server>:8765`, log in and upload photos on the *Import* page.

On Windows, don't create `.env` with `echo … > .env` in Windows PowerShell 5.1: it writes UTF-16,
and Docker Compose stops with `unexpected character "�" in variable name`. Copy the example and
edit it, or write it as plain text:
`Set-Content -Path .env -Value "VT_PASSWORD=…" -Encoding ascii`.

- [`docker-compose.yml`](docker-compose.yml) starts Mira and Ollama. The volumes
  `vt-data` (database, thumbnails, models) and `vt-photos` (the uploaded photos) keep your data.
  To use a host folder, replace `vt-photos:/photos` with e.g. `./photos:/photos`.
- NVIDIA GPU: build with `EXTRA: gpu` and uncomment the two `deploy` sections (needs the NVIDIA
  Container Toolkit).
- The container upgrades its database on start (`serve --migrate`, with a backup) and pauses a
  running pipeline when it is stopped; it continues after the next start.
- Without Docker: `vision-tagger serve --remote --upload-dir /srv/photos --host 0.0.0.0`, or set
  `mode = "remote"` and `remote.upload_dir` in `config.toml`.

### Password and HTTPS

Set a password with the environment variable `VT_WEB__PASSWORD` (the compose file requires one).
Every page then asks for it; a login lasts 30 days. After 10 wrong passwords an address is
blocked for 5 minutes.

Put the server behind a reverse proxy with HTTPS (Caddy, Traefik, nginx), for example with Caddy:

```
photos.example.com {
    reverse_proxy localhost:8765
}
```

HTTPS protects the password on the way and lets browsers install the app. Allow uploads of any
size in the proxy (`client_max_body_size 0;` in nginx); each request is at most one 8 MB piece.

## Using the results in other applications

Exports carry what Mira found as **XMP**, the metadata standard photo applications
share. The originals are never modified.

| What | Where it ends up |
|---|---|
| Keywords: objects, place, event, season, … and the people's names | `dc:subject` |
| Every property as a hierarchical keyword (`vision-tagger\|Location type\|beach`, `People\|Anna`) | `lr:hierarchicalSubject` |
| Caption of the vision model | `dc:description` |
| People, with the area of each face | `Iptc4xmpExt:PersonInImage`, MWG face regions (`mwg-rs:Regions`) |
| Capture date, including dates you corrected (not file-time guesses) | `exif:DateTimeOriginal`, `xmp:CreateDate`, `photoshop:DateCreated` |
| All raw values with confidence | `vt:Properties` (Mira namespace) |

Choose in the export dialog (or with `--xmp/--no-xmp`, `--embed/--no-embed`,
`--index/--no-index`):

- **XMP sidecar files** (`photo.jpg.xmp`) for digiKam, darktable, Immich, PhotoPrism, ExifTool,
  and Lightroom for raw files. `export.sidecar_naming = "replace"` writes `photo.xmp` instead.
- **Into the copies:** the XMP packet is added to exported JPEG and PNG copies without
  re-encoding the image. Lightroom, Apple Photos, Windows Explorer and Bridge read it. Hard and
  symbolic links are never changed (they *are* the originals), and an XMP packet that is already
  in a photo (e.g. a motion photo) is never replaced; the sidecar carries the data then.
- **Index files:** `vision-tagger.csv` (opens in Excel) and `vision-tagger.json` (for scripts)
  list every exported photo with its date, people, keywords, caption, face areas and all
  properties. Exporting again into the same folder updates them.

If your library is managed in place by digiKam, Immich or PhotoPrism, **Write XMP next to the
originals** (in the export dialog, or `vision-tagger sidecars`) puts a sidecar beside each
photo. A sidecar that another application wrote is never overwritten.

Checked with Exiv2, the metadata library of digiKam and darktable: keywords, hierarchy, caption,
people, date and named face regions are read back from sidecars and embedded copies.

## How the property extraction stays stable

1. **Closed vocabularies.** Almost all fields are enums in the JSON schema. Ollama's structured
   output enforces them during decoding.
2. **Deterministic decoding.** `temperature = 0`, a fixed `seed`, a fixed prompt, and the same
   resize for every image.
3. **Normalisation of free-text lists** (objects, tags, …). Values are lower-cased, articles are
   dropped, words are singularised, then [`synonyms.yaml`](src/vision_tagger/resources/synonyms.yaml)
   and the merges you make in the UI are applied.
4. **Versioning.** The model, prompt version and raw answer are stored. Changing the model or
   the vocabulary marks results as outdated, and the dashboard can re-queue them.

All properties end up in one key/value table (`image_properties`) with `source`
(`vlm`, `inferred`, `derived`, `exif`) and `confidence`. That table is the basis for filtering
and for clustering images by their properties later (one-hot vectors per image).

To customise the fields, copy `vocabulary.yaml` and set `vlm.vocabulary_file` in `config.toml`.

## How search works

1. **Interpretation.** The request goes to the LLM (`search.model`, by default the VLM model)
   together with the vocabulary, the most frequent free-text values, the person names and the
   date range. The JSON answer is checked against the database: unknown persons or values
   become free-text keywords instead of failing.
2. **Candidates.** Persons, required and excluded properties, and dates filter strictly. If
   nothing is left, required properties are relaxed into preferences.
3. **Ranking** combines three signals:
   - semantic similarity of an English rewrite of the request to each image's document
     embedding;
   - matches with the preferred properties, weighted by the LLM and by confidence;
   - keyword hits in captions, visible text, values and paths.

   Person names are not part of the embedded documents, so renaming people never requires
   re-indexing.
4. **Deep search.** The LLM scores the top results from short descriptions (including person
   names) and gives a one-line reason in the language of your request.

The search index is kept up to date automatically: images whose VLM results or property values
change are queued for Stage 3 again.

## Configuration

`vision-tagger init` writes `config.toml` (see
[`config.example.toml`](src/vision_tagger/resources/config.example.toml)). Any value can also be
set with an environment variable, for example `VT_VLM__MODEL=qwen3-vl:8b` or
`VT_PIPELINE__BATCH_SIZE=32`.

Important knobs:

- `mode`: `local` (scan folders on this computer) or `remote` (uploads, with `remote.upload_dir`).
- `web.password` (better: `VT_WEB__PASSWORD`): login password, meant for remote mode.
- `pipeline.resume_on_start`: continue a pipeline that was paused when the app was closed.
- `faces.cluster_eps` / `cluster_min_samples`: how strict face clustering is.
- `faces.reference_threshold` / `auto_accept_threshold`: when reference matches are suggested
  or assigned.
- `vlm.model`, `vlm.concurrency`: set `concurrency` to match `OLLAMA_NUM_PARALLEL` for
  parallel requests.
- `derived.work_path_keywords`: folder or file-name hints for work images.

## Data

Everything lives in `data/`:

- `vision_tagger.sqlite3`: index, job state, faces, clusters, persons, properties.
- `cache/thumbs`, `cache/faces`: thumbnails and face crops. These can be deleted at any time.
  - Gallery thumbnails are written while indexing, from the image that is decoded for duplicate
    detection anyway (about 5 % extra indexing time).
  - **Scan & index** (or `vision-tagger index`) also creates the thumbnails that are missing,
    e.g. for photos indexed with an older version or after deleting the cache.
  - HEIC photos use their embedded preview when it is large enough (a few milliseconds instead
    of decoding 12 megapixels).
  - Turn this off with `index.thumbnails = false`.

- `uploads/`: unfinished uploads (remote mode); finished ones are moved into the upload folder.
- `transfer/`: temporary files of database exports and imports (removed afterwards).
- `vision_tagger.sqlite3.bak-*`: backups made before a migration or an import. Delete old ones
  when you no longer need them.
- `secret.key`: the key that signs login cookies. Delete it to log everyone out.

Main tables: `images` (+ `metadata` JSON), `duplicate_files`, `stage_status`, `runs`, `faces`,
`clusters`, `persons`, `person_references`, `image_properties`, `property_aliases`,
`vlm_results`, `collections`.

## Development

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

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff and the full
test suite on Ubuntu and Windows.

**Translations.** UI text is English in the code: `_("…")` / `ngettext(…)` in templates and
Python (see [`i18n.py`](src/vision_tagger/i18n.py)); the catalogs are
`src/vision_tagger/resources/i18n/<lang>.json`. After adding or changing UI text:

```bash
uv run python -m vision_tagger.i18n_extract update       # adds new strings to every catalog
uv run python -m vision_tagger.i18n_extract check de -v  # what is missing or broken in one language
```

`tests/test_i18n.py` fails while a catalog misses a string or a translation breaks a placeholder.

Schema changes always need an Alembic migration in `src/vision_tagger/migrations/versions/`.
See [CLAUDE.md](CLAUDE.md#database-migrations-mandatory) for the workflow.
`tests/test_migrations.py` fails when the models and the migrations disagree.

## Security

In local mode the UI is meant for your own machine. It listens on `127.0.0.1`, only answers
requests whose `Host` is this machine (blocks DNS rebinding) and rejects form submissions coming
from other websites (CSRF). If you expose it on your network (`web.host = "0.0.0.0"`), add the
machine's name to `web.allowed_hosts` and set a password.

In remote mode any host name is accepted (it runs behind your own domain or proxy), while
cross-site form submissions are still rejected. Set a password and use HTTPS (see
[Password and HTTPS](#password-and-https)). Login cookies are `HttpOnly`, `SameSite=Lax` and
signed with a key stored in the data folder.

## Notes and limitations

- Deduplication: a file is a duplicate if its pHash matches and its dHash is within
  `index.dhash_max_distance` bits. Every collapsed file stays listed under *duplicates*, so
  nothing is lost.
- Symlink export needs Developer Mode on Windows. Otherwise Mira falls back to
  copying.
- InsightFace's pretrained `buffalo_l` models are licensed for **non-commercial research
  purposes** only.
- Face ages: the first faces run downloads AgeNet (430 KB, MIT license) from GitHub into
  `~/.insightface/models/agegender_faceapi`. The download is checked against a SHA-256 checksum
  and turned into an ONNX model locally. The **ages** stage re-estimates faces that were found
  before (no new face detection). It is added automatically whenever **faces** runs. Set
  `faces.age_model = "insightface"` to keep the old behaviour.
- Face ages are estimates. Expect about ±2 years for children and ±5–10 years for adults. Set
  the birth years of a few people for exact ages; Mira then also corrects the others.
