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
 folders ──▶│ 0 · Index    │──▶│ 1 · Faces & animals         │──▶│ 2 · Properties (Qwen-VL) │
            │ path, date,  │   │ detect → embed → cluster →  │   │ closed vocabularies,     │
            │ EXIF, pHash  │   │ name people and pets        │   │ assumptions, work images │
            └──────┬───────┘   └──────────────┬──────────────┘   └────────────┬─────────────┘
                   └──────────── SQLite (state per image & stage) ────────────┘
                                              │
        Web UI: home · import · faces · persons · animals · properties · images · pipeline
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
- **Stage 1c: animals and pets.** YOLOX-s finds animals (dogs, cats, birds, horses, …),
  DINOv2-small tells them apart, and photos of the same animal are grouped like faces. Name a
  group ("Bello") and search for it: "Bello at the beach", "Anna with Bello". Names, merges and
  moves survive re-grouping. See [Animals and pets](#animals-and-pets).
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
  - Alternatives work too: "Anna or Ben at the beach", "dogs or cats", "Paris or Rome",
    "photos from 2019 or 2021" (see [Alternatives](#alternatives-or)).
  - A text-embedding index (Stage 3) adds semantic similarity over everything extracted per
    image: caption, visible text, all properties, folder and date.
  - **Deep search** lets the LLM double-check the best ~40 results and explain why each fits.
  - Without an LLM, a keyword fallback still works: English and German words, the vocabulary
    in the language of the UI, names of persons, pets and places, dates, and "or" in every UI
    language.
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
- **Places and map.** Photos with a GPS position are grouped into places, each with a name to
  filter by ("Berlin, Germany"; with the optional OpenStreetMap lookup "Prenzlauer Berg,
  Berlin"). The map shows a heatmap of all geotagged photos, honouring the gallery filters, with
  the places on top. See [Places and map](#places-and-map).
- **Web UI.**
  - Face cluster overview with inline naming, merge, split, move, auto-split and remove.
  - Persons, including merging persons.
  - Property explorer: combine values with AND/OR, merge/rename/delete values.
  - Gallery with a filter builder (persons AND/OR, properties AND/OR, NOT, date, folder,
    camera, …) and export (copy / hardlink / symlink) **with the metadata for other apps**
    (see [Using the results in other applications](#using-the-results-in-other-applications)).
  - Animals page: pet groups with inline naming, merge, move and remove.
  - Image detail with face and animal boxes.
  - Timeline / data overview and a metadata table.

## Installation

### Desktop app (Windows, macOS, Linux)

The simplest way: download the app for your system from the
[releases](https://github.com/w1nte/vision-tagger/releases) and install it. No administrator
rights are needed on Windows and Linux.

| System | File | Flavours |
| --- | --- | --- |
| Windows 10/11 (x64) | `Mira-setup-<version>-<flavour>.exe` | cpu, gpu |
| macOS 15 or newer, Apple silicon | `Mira-<version>-macos-arm64.pkg` | cpu |
| Linux x86_64 (glibc 2.35+, e.g. Ubuntu 22.04, Debian 12, Fedora 36) | `Mira-<version>-linux-x86_64-<flavour>.tar.gz` | cpu, gpu |

- **gpu**: for an NVIDIA graphics card (CUDA and cuDNN are included; needs driver R580 or newer).
- **cpu**: for every other computer (slower face detection, much smaller download).

Mira then starts like any other application, in a window of its own. Closing the window stops
it (a running pipeline is paused and continues on the next start); starting it again while it
runs opens another window. Install **Ollama** as well (step 4 below) for properties and search;
Mira then downloads the models it needs by itself.

**Windows.** Run the setup program; it installs into `%LOCALAPPDATA%\Programs\Mira` and adds
Mira to the Start menu. Each setup is a single `.exe`; only the gpu setup of 1.2.1 also needs
its `.bin` file in the same folder. (Should a future setup ever be split into `.bin` files, it
downloads missing ones from the release by itself.) The setup is not signed, so SmartScreen may
warn about an "unrecognized app": click **More info → Run anyway**.

**macOS.** Open the `.pkg`; it installs `Mira.app` into the Applications folder (macOS asks for
an administrator password). The package is not signed by Apple, so macOS first refuses to open
it: open **System Settings → Privacy & Security** and click **Open Anyway** next to the message
about the package (the same applies if macOS blocks Mira.app at its first start).
Alternatively remove the quarantine flag in the Terminal: `xattr -dr com.apple.quarantine
~/Downloads/Mira-*.pkg`. Intel Macs are not supported (there is no ONNX Runtime for them any
more).

**Linux.** Unpack the archive and run its installer as your own user (not root):

```bash
tar -xzf Mira-<version>-linux-x86_64-cpu.tar.gz
cd Mira-<version>-linux-x86_64-cpu
./install.sh                  # into ~/.local/share/mira; --prefix DIR for another folder
```

It adds `mira` and `vision-tagger-cli` to `~/.local/bin` and Mira to the applications menu.
`./install.sh --uninstall` removes the app again (the library is kept). The gpu archive is
bigger than GitHub allows for one file, so it comes in parts: download all
`….tar.gz.partNN` files and unpack them with `cat Mira-<version>-linux-x86_64-gpu.tar.gz.part* |
tar -xz`. On Linux, Mira's window is Chrome, Chromium or Edge in app mode (the app does not
bundle GTK or Qt for a native window); without one of them it opens in your default browser.

**Updating.** Mira tells you when a new version is out (see [Updates](#updates)). Run the new
setup program, the new `.pkg`, or the new `install.sh`. They detect the
installed version and update it: a running Mira is asked to quit first (a running pipeline is
paused and continues on the next start), the program files are replaced, and your library is
kept and its database migrated to the new version (a backup copy is made first,
`vision_tagger.sqlite3.bak-<version>-<time>`). If the migration fails, the installation is kept
and Mira tries again when it starts; the details are in `logs/migrate.log` in the data folder.
What changed in each version is in the [changelog](CHANGELOG.md), and in the app under
*What's new* (sidebar).

Where things are:

| | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Library (`config.toml`, `data/`) | `%LOCALAPPDATA%\vision-tagger` | `~/Library/Application Support/vision-tagger` | `~/.local/share/vision-tagger` |
| Command line | `vision-tagger-cli.exe` in the install folder | `/Applications/Mira.app/Contents/MacOS/vision-tagger-cli` | `vision-tagger-cli` |
| Logs | `data\logs\desktop.log` in the library | `data/logs/desktop.log` in the library | `data/logs/desktop.log` in the library |

The library (settings, database, thumbnails) stays when you update or uninstall the app;
settings are in its `config.toml`. The command line works like `vision-tagger` from a source
checkout (`vision-tagger-cli index D:\Photos`, `vision-tagger-cli run`, …); `vision-tagger-cli
quit-app` closes a running app gracefully and `vision-tagger-cli migrate --desktop` migrates the
app's library (the installers use both).

From a source checkout the same window opens with `uv run vision-tagger desktop` (add
`--extra desktop` to `uv sync` for a native window; without it, Edge or Chrome opens in app mode).
The apps are built by [`.github/workflows/desktop.yml`](.github/workflows/desktop.yml) from
[`packaging`](packaging): PyInstaller makes the app ([`vision-tagger.spec`](packaging/vision-tagger.spec)),
its self-test checks it, and Inno Setup ([`windows`](packaging/windows)), `pkgbuild`
([`macos`](packaging/macos)) or a tar archive with an install script ([`linux`](packaging/linux))
packs it. See [Releases](#releases).

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
4. **Ollama** from <https://ollama.com/download>. Install and start it; no `ollama pull` is
   needed. Mira lets Ollama download the models it needs (see [AI models](#ai-models)):
   `qwen2.5vl:7b` for properties (~6 GB VRAM; `qwen2.5vl:3b` for smaller GPUs, `qwen3-vl:8b`
   also works) and `nomic-embed-text` for the search index.
5. **Mira:**
   ```powershell
   git clone https://github.com/w1nte/vision-tagger.git
   cd vision-tagger
   uv sync --extra gpu           # Python 3.12 is installed automatically by uv
   uv run vision-tagger init     # creates config.toml and the database
   ```

For CPU-only machines (or Linux/macOS) use `uv sync --extra cpu` and set
`providers = ["CPUExecutionProvider"]` in `config.toml`.

## Updates

The installed desktop app looks for a new release once a day and tells you when there is one:

- **What is checked.** About 15 seconds after the start, and then every 24 hours while it runs,
  Mira asks GitHub for the newest release of this repository
  (`https://api.github.com/repos/w1nte/mira/releases/latest`) and compares its version
  with the one you run. The answer is kept in `update-check.json` in the data folder, so
  starting the app often doesn't ask GitHub more than about once a day.
- **What you see.** When a newer version exists, the sidebar shows *Update available: Mira
  1.3*, and Home shows a card with *What's new* (the release page), *Download* (the installer
  for your computer, when it is a single file; otherwise the release page has it) and *Not now*,
  which hides the card for that version (the sidebar keeps the notice). The Environment card on
  the Pipeline page shows your version, the last check and its result, and a *Check now*
  button.
- **Privacy.** One request to `api.github.com`, without an account, cookies or anything about
  your photos or library. GitHub sees what any web request shows: your IP address, the time and
  a client name with the version (`Mira-update-check/1.2`).
- **Nothing is installed by itself.** Mira never downloads or installs anything on its own. To
  update, run the new setup (see [Updating](#desktop-app-windows-macos-linux)): your library is
  kept and its database migrated, with a backup copy first.
- **Turning it off.** Set `check = "off"` in the `[updates]` section of `config.toml` (or the
  environment variable `VT_UPDATES__CHECK=off`). The default, `"auto"`, checks only in the
  installed app; `"on"` also checks from a source checkout, on a server or in Docker (there the
  card links to the release page, since the server's system is not the viewer's).
  `prereleases = true` also tells you about release candidates and betas.
- **Command line.** `vision-tagger update-check` (`vision-tagger-cli update-check` in the
  installed app) prints your version, the newest one and the download link. Exit code 0: up to
  date, 1: an update is available, 2: unknown (no connection, GitHub's rate limit).

## Upgrading

The desktop app: run the installer of the new version (see [Updating](#desktop-app-windows-macos-linux)).
From source:

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
- A database from a *newer* version (after going back to an older one) is never touched: Mira
  refuses to open it, and `migrate` exits with code 3. A failed migration exits with code 4.

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
   away* ticked, the pipeline runs afterwards: faces, ages, animals, properties and the search
   index.
2. Optionally put reference portraits into `references\<Person Name>\` and click
   **Import references** on the **Pipeline** page.
3. **Pipeline** shows the progress per stage, throughput, ETA and errors. There you can also
   choose the stages, a batch size and an optional limit, and **Pause**, **Resume** or **Stop**
   at any time.

Then name and clean up clusters under **Faces**, name your pets under **Animals**, explore
**Properties**, and filter or export in **Images**. **Home** shows what is in your library and your collections.

**Closing the app pauses a running pipeline** after the current image. The next start continues
it where it stopped, with the same stages and settings. Turn this off with
`pipeline.resume_on_start = false`.

Everything also works headless:

```powershell
uv run vision-tagger index "D:\Photos"
uv run vision-tagger run --stage faces --stage vlm --batch-size 16 --limit 500
uv run vision-tagger status
uv run vision-tagger models [pull]               # Ollama models: list / download missing ones
uv run vision-tagger pause | resume | stop       # from a second terminal
uv run vision-tagger refs                        # import reference DB + name clusters
uv run vision-tagger cluster [--full]
uv run vision-tagger pets [--full]               # group the detected animals again
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

### AI models

Properties and search use models that Ollama keeps (`vlm.model`, `search.embed_model` and
`search.model` if you set one). You don't have to pull them in a terminal; Mira asks Ollama to
download them:

- **Home** shows a card while models are missing, with **Download now**. If Ollama isn't
  installed or not running, the card links to the Ollama download instead.
- **Pipeline → Environment** lists every model with its state (*ready*, *missing*,
  *downloading 42 %*) and a **Download** button, and **Download missing models** when several are
  missing. The download runs in the background, one model after another, and the sidebar shows
  its progress (model, GB done of the total, percent). **Cancel download** stops it; Ollama keeps
  what it has downloaded, so the next download continues there.
- **A pipeline run** that needs a missing model downloads it first (the Pipeline page shows the
  progress), then carries on. Pause and Stop work during the download. Turn this off with
  `vlm.auto_pull = false`: the run then stops and asks you to download the model.
- **Command line:** `vision-tagger models` lists the models and whether Ollama has them (exit
  code 1 if some are missing); `vision-tagger models pull` downloads the missing ones with a
  progress line. The desktop app's `vision-tagger-cli.exe` has the same commands.

The models need several GB of disk space (qwen2.5vl:7b about 6 GB). Mira never starts a download
on its own when it starts, only on a button or for a run that needs the model. In remote mode
and with Docker, the server's Ollama downloads them. `ollama pull <model>` still works if you
prefer it.

## The web UI

| Page | What you can do |
|---|---|
| **Home** | Your library at a glance: number of photos, people, places and years, *On this day* memories, recently added photos, and collections. Nudges you when face or animal groups still need a name, and while AI models are missing (with **Download now**). |
| **Collections** | A collection is a saved filter: it stays up to date as new photos arrive. Suggested ones (people, places, events, this year) appear by themselves; **Save as collection** in *Images* adds your own. Rename, reorder and delete them here. |
| **Import** | Local mode: your photo folders, with a folder browser, and **Scan now**. Remote mode: upload photos, folders or ZIP files (see [Remote mode](#remote-mode-and-docker)). Shows what is being indexed and analysed right now. |
| **Pipeline** | The technical part: start/pause/resume/stop runs (stages, batch size, limit), per-stage progress, ETA, errors (retry/skip), outdated results, jobs, recent runs, environment check (CUDA, Ollama, and the AI models with their state and **Download** buttons). |
| **Database** | Export the whole library database (backup, another computer) and import one (see [Moving a library](#moving-a-library-to-another-computer-export--import)). **Missing files** lists photos that are not where Mira last saw them; scanning their new folder re-links them. |
| **Faces** | All clusters with cover face and size. Name inline (autocomplete), accept reference suggestions, select several to **merge**. Views for unclustered and ignored faces. |
| **Cluster** | All faces of a cluster. Select faces to **split** into a new cluster, **assign to a person**, **move** to another cluster, or **remove**. **Auto-split** mixed clusters. Similar clusters with one-click merge. |
| **Animals** | Pet groups with picture and number of photos, by species. Name a group inline (typing another group's name merges the two), tick several to **merge**. Per group: all its photos to **move** to a pet (by name) or a new group, **take out** ("Not this one") or **remove**, similar groups with one-click merge, and the species (correct it when the detector took a dog for a bear). Views for ungrouped and removed animals. |
| **Persons** | Named persons. Rename (renaming to an existing name merges), merge, delete, manage references. Images with *all* / *any* of the selected persons. Per person: birth year (set or estimated), age today, a "through the years" face timeline, and photos with a doubtful date with suggested fixes. |
| **Properties** | Every extracted property grouped by topic, with counts. Tick values and show images with **all** / **any** of them (*any* keeps the other filters: "summer and (beach or mountains)"). Per property: sample images per value, **merge / rename / delete** values (remembered for future results). |
| **Images** | **Timeline slider** above the gallery: photos per month; click the first month, then the last, drag the band or its edges, or sweep across the bars (arrow keys move the handles). Gallery with infinite scroll and a filter builder: persons and pets (AND/OR, NOT), properties (AND/OR, NOT), date range, folder, camera, faces, text search, sort. Facets refine the current result; their **or** switch (or Ctrl+click / ⌘+click on a value) adds a value as an alternative, e.g. "beach or mountains" (see [Alternatives](#alternatives-or)). **Export** the result, with XMP metadata for other apps. Filters are part of the URL, so they can be bookmarked. |
| **Image** | Full image with face and animal boxes and names, all properties with source and confidence, metadata, duplicates, raw VLM JSON, per-stage status and re-run buttons. `←` / `→` open the previous / next photo of the current filter. |
| **Quiz** | "Who's that?": name the largest unnamed clusters with one click (suggestions from references and look-alikes), or play the face quiz with score, streaks and "you vs. the AI"; wrong answers can fix misfiled faces. Keys 1–4 and Enter. |
| **Generations** | Every person with photos from different ages. Per person: a grow-up timelapse (play, scrub, download as animated WebP), "looks most like…" ranking, and a same-age comparison with any other person. |
| **Search** | Search box in the top bar on every page (`/` focuses it). Shows how the request was understood (persons, pets, required / preferred / excluded properties, alternatives such as "Anna or Ben", dates, keywords), ranks the results and, with **Deep search**, gives the LLM's reason for each. "Open in filter builder" turns the strict part into a gallery filter; results can be exported. |
| **Map** | Heatmap of the geotagged photos of the current filter (e.g. one person's photos), the places with their photo counts (click one for thumbnails, *Show photos* and renaming), **Photos in this area** (the map bounds as a gallery filter) and *Find places*. The view is kept in the URL (`#zoom/lat/lon`). |
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
```

Then open `http://<server>:8765`, log in and upload photos on the *Import* page. The `ollama`
service downloads the AI models the first time they are needed: click **Download now** on
*Home*, or let the first run do it (see [AI models](#ai-models)). By hand it is
`docker compose exec ollama ollama pull qwen2.5vl:7b` (and `nomic-embed-text`).

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

## Places and map

Phones and many cameras store where a photo was taken (EXIF GPS). Mira keeps that
position in `images.lat` / `images.lon` and groups the photos into **places**:

- Photos closer than `geo.cluster_km` (750 m) to each other form a place (DBSCAN with
  haversine distance). Groups wider than `geo.max_place_km` (5 km) are split.
- **One town, one place.** Groups with the same automatic name within `geo.merge_km` (25 km) of
  each other are joined, so a town and its outskirts never show up twice. Same-named towns
  farther apart stay separate, and if their names would still be identical, they are told apart
  by the region or the nearest larger town ("Neustadt (Coburg), Germany" and "Neustadt
  (Mannheim), Germany"; "Springfield, IL" and "Springfield, MO" differ already).
- Every photo gets the place as `images.place_id` and as the properties **place**, **city** and
  **country**. They work everywhere other properties work: gallery facets and filters, the
  properties page, XMP export and the search index. `/images?place=12` shows one place
  (`place=12&place=14` either of two), `/images?bbox=minlon,minlat,maxlon,maxlat` an area of
  the map.
- Places are found at the end of every scan, with **Find places** on the map page, or with
  `vision-tagger places` (`--refresh` groups everything again and renames every place you did
  not name).
- **Your names are kept.** Rename a place in its popup on the map: the name, the place and its
  photos survive regrouping (like locked face clusters), and new photos nearby join it. An empty
  name returns to the automatic one. Values you merge or rename on the properties page stay
  merged too.

**Offline names (default).** A place is named after the town it lies in: the nearest one, where
big towns reach farther, so a photo in a suburb gets the city's name and a photo in a village
the village's name. The towns come from [GeoNames](https://www.geonames.org/) ("cities with a
population of at least 1000", 130,000 towns, CC BY 4.0). The first time places are named,
Mira downloads a copy of that data (3 MB) from the npm registry (the
[`all-the-cities`](https://www.npmjs.com/package/all-the-cities) package, version 3.1.0, MIT),
checks its size and SHA-256 and converts it into `data/models/geo`. GeoNames' own server
publishes a new dump every day, which can't be checked against a fixed checksum. To use an
official dump instead, download `cities500.zip` (or `cities1000.zip`, ...) from
<https://download.geonames.org/export/dump/> and set `geo.cities_file`; with
`geo.download_gazetteer = false` nothing is downloaded at all (places are then named by their
coordinates, e.g. `52.516° N, 13.378° E`).

**OpenStreetMap names (optional).** With `geo.nominatim = true`, places are named from
OpenStreetMap through [Nominatim](https://nominatim.org/) reverse geocoding, down to the district
("Prenzlauer Berg, Berlin"), in `geo.language`. This **sends the position of every new place to
the Nominatim server** (rounded to about 100 m), so it is off by default. Mira follows
the [usage policy](https://operations.osmfoundation.org/policies/nominatim/) of the public
server: at most one request per second, an identifying User-Agent without personal data, every
answer cached in the database (`geocode_cache`) and never asked again, and no further requests
in a run after the server refuses one. A self-hosted server can be set with `geo.nominatim_url`.

**Map tiles.** The map background comes from `geo.tile_url`, by default the OpenStreetMap tile
server (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`, with its required attribution). The
heatmap data stays on your machine, but **every tile request tells the tile server which area
you are looking at.** Point `geo.tile_url` (and `geo.tile_attribution`) at your own tile
server, or set it to `""`: the map then shows a coarse offline world outline (Natural Earth,
public domain) instead. The same outline appears when the tiles can't be loaded.

**Licenses.** Map data and OpenStreetMap names: © OpenStreetMap contributors, available under
the [Open Database License](https://www.openstreetmap.org/copyright). Offline place names:
GeoNames, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). World outline: Natural
Earth, public domain. Vendored libraries (in `web/static/vendor`, with their license files):
[Leaflet](https://leafletjs.com/) 1.9.4 and [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat)
0.2.0 (both BSD 2-Clause).

## Animals and pets

The **animals** stage (1c, after the faces) finds the animals in every photo, groups the photos
of the same animal, and lets you name them. A named pet works like a person: a filter
(`/images?pet=5`, chips and a facet in the gallery, AND/OR together with persons), a collection
on the home page, and the search: "Bello am Strand", "photos of Minka in winter", "Anna with
Bello".

- **Detection: [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, Apache-2.0),
  trained on COCO. It finds birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras and
  giraffes, each with a box, the species and a score. Detections below `animals.min_score`
  (0.5) or smaller than `animals.min_box_px` (32 px) are dropped; uncertain or small ones (below
  `group_min_score` 0.6 or `group_min_px` 64 px) are kept but not grouped automatically.
- **Telling animals apart: [DINOv2-small](https://github.com/facebookresearch/dinov2)** (Meta,
  Apache-2.0, as ONNX from `onnx-community/dinov2-small`). Each animal is cut out with a small
  margin, padded to a square (never stretched) and described by DINOv2's class token, averaged
  with the mirrored crop.
- **Grouping** works like face clustering: DBSCAN with the cosine distance `animals.cluster_eps`
  (0.3) and at least `cluster_min_samples` (2) photos; new photos join a pet when they are at
  least `animals.assign_threshold` (0.7) similar to its average. Animals are grouped per
  species, but the species the detector mixes up on pets (a chow chow detected as a bear, a
  poodle as a sheep) count as one (`animals.species_pools`: cat, dog, bear, sheep). A pet's
  species is what most of its photos show; you can correct it on its page.
- **Naming** happens on the **Animals** page (sidebar, *People*): type a name under a group.
  Typing the name of another group merges the two. Open a group to move photos to another pet
  (by name) or into a new group, to take them out ("Not this one") or to remove them (not an
  animal). Like locked face clusters, **named and edited groups, your moves and removals are
  kept** when the animals are grouped again (*Group all again* on the Pipeline page,
  `vision-tagger pets --full`) and when photos are analysed again. A named pet stays even
  without photos, so its animal finds its way back.
- **Properties.** Every photo gets `animal = <species>` and, for each named pet in it,
  `pet = <name>` (source `animal`), so the facets, filters, the properties page (group *Pets*),
  XMP keywords and the search work as for places. The vision model's own free-text **Animals**
  property stays as it is (it also names animals the detector doesn't know, like ducks or
  fish); the detector's property is shown as **Species** to tell them apart. The species is part
  of the search document; pet names are not (like person names, they filter strictly), so
  renaming a pet needs no re-indexing.
- **How the thresholds were chosen.** On 107 real photos: the 9 dogs and cats of the
  [DreamBooth dataset](https://github.com/google/dreambooth) (5–6 photos of each animal,
  Unsplash license) and, as other animals, 60 photos of 10 breeds from the Oxford-IIIT Pet test
  set (CC BY-SA 4.0). Every photo had an animal at a score of 0.5 or more, and 27 photos without
  animals had none; the species was wrong for 9 of the 107 photos (mostly fluffy dogs as bears
  or sheep, hence the pools). Photos of the same animal had a cosine similarity of 0.80 on
  average (10th percentile 0.70), other animals of the same species 0.14 (99th percentile 0.49).
  With the defaults the 9 animals form 9 groups (pairwise precision 1.00, recall 0.95: one photo
  stays ungrouped). Animals of one breed and colour are hard to tell apart: the grey cat was
  joined by Russian Blues, and different pugs or black cats form one group. Split them with
  *New group*.
- **Privacy and downloads.** Both models (125 MB) are downloaded once into
  `<faces.model_root>/models/animals` (default `~/.insightface/models/animals`; or
  `animals.model_dir`) from GitHub and Hugging Face, pinned to fixed files and checked against
  their SHA-256 checksums. Without internet, the error message names the URL and the place for
  the file. Everything else runs locally on onnxruntime (CUDA or CPU, `faces.providers`), about
  0.25 s per photo on a 4-core CPU. No photo leaves the computer.

The animals stage runs by default. A `config.toml` written by an earlier version is updated
on the next start (see [Configuration](#configuration)); only a stage list you changed
yourself stays as it is, so add `"animals"` there if you want it.

## Using the results in other applications

Exports carry what Mira found as **XMP**, the metadata standard photo applications
share. The originals are never modified.

| What | Where it ends up |
|---|---|
| Keywords: objects, place, event, season, …, pets and their species, and the people's names | `dc:subject` |
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
   together with the vocabulary, the most frequent free-text values, the person and pet names
   and the date range. The JSON answer is checked against the database: unknown persons or values
   become free-text keywords instead of failing.
2. **Candidates.** Persons, pets, required and excluded properties, alternatives and dates filter
   strictly. If nothing is left, required properties are relaxed into preferences.
3. **Ranking** combines three signals:
   - semantic similarity of an English rewrite of the request to each image's document
     embedding;
   - matches with the preferred properties, weighted by the LLM and by confidence;
   - keyword hits in captions, visible text, values and paths.

   Person and pet names are not part of the embedded documents, so renaming people or pets
   never requires re-indexing.
4. **Deep search.** The LLM scores the top results from short descriptions (including person
   names) and gives a one-line reason in the language of your request.

The search index is kept up to date automatically: images whose VLM results or property values
change are queued for Stage 3 again.

### Alternatives ("or")

Alternatives inside a condition are OR-ed, conditions are AND-ed, and exclusions always apply:

| Request | Understood as |
|---|---|
| Anna or Ben at the beach | (Anna or Ben) and beach |
| Anna oder Ben am Strand | the same, in German |
| dogs or cats | animals: dog or cat |
| beach or mountains in summer 2021 | (beach or mountains) and summer, 2021 |
| photos from 2019 or 2021 | taken in 2019 or in 2021 |
| Paris or Rome, not at work | city Paris or Rome, without work photos |
| without Anna or Ben | neither Anna nor Ben |

"Anna and Ben" still means photos with both. The LLM understands "or" in any language; the
keyword fallback knows *or, oder, ou, o, of, lub, или, 或, または* and *か* (between two
recognised names, values, places or dates, so names that contain such a word stay whole). The
search page shows each group as one chip ("Anna *or* Ben"), and **Open in filter builder** keeps
the groups.

In the gallery, a facet's **or** switch lists the values you could add as alternatives (counted
without the condition on that property); Ctrl+click (⌘+click) on a value does the same. The
filter chips show alternatives in one chip, each with its own remove button. In the URL, `|`
separates the alternatives of one `t` condition; a photo has one date and one place, so repeated
`d` and `place` parameters are alternatives as well:

```
/images?t=location_type:beach|location_type:mountain&t=season:summer   (beach or mountains) and summer
/images?p=3&p=4&pm=any                                                   Anna or Ben (pm=all: both)
/images?d=2019&d=2021-06..2021-08                                        2019 or summer 2021
/images?place=12&place=14                                                either of two places
```

Older URLs and saved collections keep their meaning: repeated `t` parameters are all required,
and `tm=any` (the former "any property" switch) makes them one "or" condition.

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
- `animals.cluster_eps` / `assign_threshold` / `min_score`: how strict pet grouping and animal
  detection are (see [Animals and pets](#animals-and-pets)).
- `vlm.model`, `vlm.concurrency`: set `concurrency` to match `OLLAMA_NUM_PARALLEL` for
  parallel requests.
- `vlm.auto_pull` (default `true`): a pipeline run downloads a missing Ollama model first. With
  `false` it stops and asks for the download (the buttons and `vision-tagger models pull` still
  work).
- `derived.work_path_keywords`: folder or file-name hints for work images.
- `geo.cluster_km`, `geo.nominatim`, `geo.tile_url`: places and the map (see
  [Places and map](#places-and-map)).
- `pipeline.stages`: leave it out to run every stage, including stages that later versions
  add. List stages only to turn some off. The Pipeline page shows the stages your config turns
  off, so you can still run them once.
- `updates.check` (`"auto"`, `"on"`, `"off"`) and `updates.prereleases`: looking for new
  releases (see [Updates](#updates)).

**Updates.** `config.toml` carries a `config_version`. When a new version of Mira changes what
a setting means or where it lives, it updates older files itself when it loads them, keeping
your values and comments. It writes a backup copy next to the file first
(`config.toml.bak-v1-<time>`). `vision-tagger migrate` shows what was changed. If the file can't
be written (read-only), Mira uses the updated settings anyway and logs a warning.

Version 2 removes the `pipeline.stages` list that `vision-tagger init` and the desktop app
copied from the example config. That copy kept every stage added later switched off: ages, the
search index and animals. A list you changed yourself is kept.

## Data

Everything lives in `data/`:

- `vision_tagger.sqlite3`: index, job state, faces, clusters, persons, animals, pets,
  properties.
- `cache/thumbs`, `cache/faces`, `cache/animals`: thumbnails, face and animal crops. These can be deleted at any time.
  - Gallery thumbnails are written while indexing, from the image that is decoded for duplicate
    detection anyway (about 5 % extra indexing time).
  - **Scan & index** (or `vision-tagger index`) also creates the thumbnails that are missing,
    e.g. for photos indexed with an older version or after deleting the cache.
  - HEIC photos use their embedded preview when it is large enough (a few milliseconds instead
    of decoding 12 megapixels).
  - Turn this off with `index.thumbnails = false`.
- `models/geo`: the converted GeoNames towns for place names (downloaded once).

- `uploads/`: unfinished uploads (remote mode); finished ones are moved into the upload folder.
- `transfer/`: temporary files of database exports and imports (removed afterwards).
- `vision_tagger.sqlite3.bak-*`: backups made before a migration or an import. Delete old ones
  when you no longer need them.
- `secret.key`: the key that signs login cookies. Delete it to log everyone out.
- `update-check.json`: the last update check and the version you dismissed on Home.

Main tables: `images` (+ `metadata` JSON), `duplicate_files`, `stage_status`, `runs`, `faces`,
`clusters`, `persons`, `person_references`, `image_properties`, `property_aliases`,
`vlm_results`, `collections`, `places`, `geocode_cache`, `animals`, `pets`.

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

## Releases

Publishing a GitHub release builds the desktop apps and attaches them to it, in about an hour:

1. In the [changelog](CHANGELOG.md), rename `## Unreleased` to `## X.Y.Z - <today>` (the
   version of the tag in step 2), add an empty `## Unreleased` above it and commit that to
   `main` (see [CLAUDE.md](CLAUDE.md#changelog-mandatory)).
2. On GitHub, *Releases → Draft a new release*, create a tag `vX.Y.Z` (pre-releases:
   `vX.Y.Z-rc.1`, `-beta.2`, `-alpha.1`) on `main`, and publish it (a pre-release counts too;
   a draft does not). Leave the notes empty to get the changelog's section as notes.
3. [`.github/workflows/desktop.yml`](.github/workflows/desktop.yml) builds Windows (cpu, gpu),
   Linux (cpu, gpu) and macOS (Apple silicon) with the tag as the version
   ([`packaging/version.py`](packaging/version.py): `v1.2.3-rc.1` → `1.2.3rc1`), runs each
   app's self-test, packs the installers and uploads them to the release. A release without
   notes gets the changelog's section for the tag ([`packaging/changelog.py`](packaging/changelog.py)).
4. If a build fails, re-run the failed jobs; the upload replaces files of the same name.

The version in `pyproject.toml` is only used for builds without a release (manual runs of the
workflow, and the cpu builds it makes as a check on pull requests and pushes to `main` that touch
the packaging). Build locally with:

```bash
uv sync --extra cpu --extra desktop --group packaging
uv run pyinstaller packaging/vision-tagger.spec --noconfirm --clean
dist/vision-tagger/vision-tagger-cli self-test
packaging/linux/build_tarball.sh 0.0.0 cpu     # Linux; macOS: packaging/macos/build_pkg.sh 0.0.0
```

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
  purposes** only. The animal models (YOLOX-s, DINOv2-small) are Apache-2.0.
- Face ages: the first faces run downloads AgeNet (430 KB, MIT license) from GitHub into
  `~/.insightface/models/agegender_faceapi`. The download is checked against a SHA-256 checksum
  and turned into an ONNX model locally. The **ages** stage re-estimates faces that were found
  before (no new face detection). It is added automatically whenever **faces** runs. Set
  `faces.age_model = "insightface"` to keep the old behaviour.
- Face ages are estimates. Expect about ±2 years for children and ±5–10 years for adults. Set
  the birth years of a few people for exact ages; Mira then also corrects the others.
