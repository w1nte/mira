# vision-tagger

A local pipeline that indexes a large photo collection, finds and clusters faces, and extracts
stable, filterable properties with a local vision-language model (Qwen2.5-VL via Ollama). You
start and control it from a local web UI. Originals are never moved or modified.

```
            ┌──────────────┐   ┌─────────────────────────────┐   ┌──────────────────────────┐
 folders ──▶│ 0 · Index    │──▶│ 1 · Faces (InsightFace)     │──▶│ 2 · Properties (Qwen-VL) │
            │ path, date,  │   │ detect → embed → cluster →  │   │ closed vocabularies,     │
            │ EXIF, pHash  │   │ name via reference DB       │   │ assumptions, work images │
            └──────┬───────┘   └──────────────┬──────────────┘   └────────────┬─────────────┘
                   └──────────── SQLite (state per image & stage) ────────────┘
                                              │
                         Web UI: dashboard · faces · persons · properties · images · overview
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
- **Ages and doubtful dates.** Give a person a birth year, or let vision-tagger estimate one
  from photo dates and face ages.
  - **Face ages from two models.** InsightFace's age model turns babies and children into young
    adults (a 12-year-old easily comes out as 40). vision-tagger adds AgeNet, a small network
    from [face-api](https://github.com/vladmandic/face-api) (MIT license) trained on many
    children and old people, and combines both. AgeNet is used alone for children and teenagers,
    and both count equally from about 35 years on.
  - **Your birth years make it better.** Every birth year you set also shows how far off the
    face model is on *your* photos. vision-tagger learns a correction per age range from those
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
    camera, …) and export (copy / hardlink / symlink).
  - Image detail with face boxes.
  - Timeline / data overview and a metadata table.

## Installation (Windows + NVIDIA GPU)

1. **uv** (package manager):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Microsoft C++ Build Tools.** `insightface` is built from source. In the
   [Build Tools installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/), select
   *Desktop development with C++*.
3. **A recent NVIDIA driver.** The `gpu` extra installs `onnxruntime-gpu[cuda,cudnn]`, which
   brings the CUDA and cuDNN runtime as pip packages, so a separate CUDA install is not needed.
   vision-tagger calls `onnxruntime.preload_dlls()` so these DLLs are found. Current
   onnxruntime-gpu builds target CUDA 13, which needs driver R580 or newer. If CUDA can't be
   loaded, InsightFace falls back to the CPU and the dashboard shows a warning. Check
   **Environment** on the dashboard.
4. **Ollama** from <https://ollama.com/download>, then pull the model:
   ```powershell
   ollama pull qwen2.5vl:7b      # ~6 GB VRAM; qwen2.5vl:3b for smaller GPUs, qwen3-vl:8b also works
   ollama pull nomic-embed-text  # small text-embedding model for the search index
   ```
5. **vision-tagger:**
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

- If the database is older than the code, vision-tagger refuses to start and tells you to run
  `vision-tagger migrate`.
- All pending migrations run in a single transaction, so a failed migration leaves the database
  as it was.
- The backup is written next to the database as `vision_tagger.sqlite3.bak-<version>-<time>`.
- Databases from the very first version are upgraded too.
- `vision-tagger migrate --check` only reports whether an upgrade is needed.

## Quick start

```powershell
uv run vision-tagger serve --open          # or: .\start.ps1
```

On the **Dashboard**:

1. Enter your photo folder(s) and click **Scan & index**.
2. Optionally put reference portraits into `references\<Person Name>\` and click
   **Import references**.
3. Choose the stages, a batch size and an optional limit, then click **Start**. **Pause**,
   **Resume** and **Stop** work at any time. Progress, throughput, ETA and errors update live.

Then name and clean up clusters under **Faces**, explore **Properties**, and filter or export in
**Images**.

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
uv run vision-tagger search "Anna am Strand im Sommer, ohne Arbeitsfotos" --deep
uv run vision-tagger growup "Anna" anna-growing-up.webp  # grow-up animation
```

`Ctrl+C` during `run` stops gracefully. Running `run` again continues where it stopped.

## The web UI

| Page | What you can do |
|---|---|
| **Dashboard** | Index folders, start/pause/resume/stop runs (stages, batch size, limit), per-stage progress, ETA, errors (retry/skip), outdated results, jobs, environment check (CUDA, Ollama, model). |
| **Faces** | All clusters with cover face and size. Name inline (autocomplete), accept reference suggestions, select several to **merge**. Views for unclustered and ignored faces. |
| **Cluster** | All faces of a cluster. Select faces to **split** into a new cluster, **assign to a person**, **move** to another cluster, or **remove**. **Auto-split** mixed clusters. Similar clusters with one-click merge. |
| **Persons** | Named persons. Rename (renaming to an existing name merges), merge, delete, manage references. Images with *all* / *any* of the selected persons. Per person: birth year (set or estimated), age today, a "through the years" face timeline, and photos with a doubtful date with suggested fixes. |
| **Properties** | Every extracted property grouped by topic, with counts. Tick values and show images with **all** / **any** of them. Per property: sample images per value, **merge / rename / delete** values (remembered for future results). |
| **Images** | Gallery with infinite scroll and a filter builder: persons (AND/OR, NOT), properties (AND/OR, NOT), date range, folder, camera, faces, text search, sort. Facets refine the current result. **Export** the result. Filters are part of the URL, so they can be bookmarked. |
| **Image** | Full image with face boxes and names, all properties with source and confidence, metadata, duplicates, raw VLM JSON, per-stage status and re-run buttons. `←` / `→` open the previous / next photo of the current filter. |
| **Quiz** | "Who's that?": name the largest unnamed clusters with one click (suggestions from references and look-alikes), or play the face quiz with score, streaks and "you vs. the AI"; wrong answers can fix misfiled faces. Keys 1–4 and Enter. |
| **Generations** | Every person with photos from different ages. Per person: a grow-up timelapse (play, scrub, download as animated WebP), "looks most like…" ranking, and a same-age comparison with any other person. |
| **Search** | Search box in the top bar on every page (`/` focuses it). Shows how the request was understood (persons, required / preferred / excluded properties, dates, keywords), ranks the results and, with **Deep search**, gives the LLM's reason for each. "Open in filter builder" turns the strict part into a gallery filter; results can be exported. |
| **Overview** | Timeline (year → month → day), breakdowns by folder, camera, file type, date reliability, work/personal, orientation, and a sortable metadata table. |

The sidebar groups the pages into *Library*, *People* and *Organize* and shows what the pipeline
is doing right now. On small screens it folds away behind the menu button.

**Light and dark mode.** The switch at the bottom of the sidebar picks the theme: *System*
(follows your operating system), *Light* or *Dark*. The choice is kept in a cookie (`vt_theme`),
so every page is rendered in the right theme from the start, without flashing.

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

Main tables: `images` (+ `metadata` JSON), `duplicate_files`, `stage_status`, `runs`, `faces`,
`clusters`, `persons`, `person_references`, `image_properties`, `property_aliases`,
`vlm_results`.

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

Schema changes always need an Alembic migration in `src/vision_tagger/migrations/versions/`.
See [CLAUDE.md](CLAUDE.md#database-migrations-mandatory) for the workflow.
`tests/test_migrations.py` fails when the models and the migrations disagree.

## Security

The UI is meant for your own machine. It listens on `127.0.0.1`, only answers requests whose
`Host` is this machine (blocks DNS rebinding) and rejects form submissions coming from other
websites (CSRF). If you expose it on your network (`web.host = "0.0.0.0"`), add the machine's
name to `web.allowed_hosts`. There is no login, so only do that on a trusted network.

## Notes and limitations

- Deduplication: a file is a duplicate if its pHash matches and its dHash is within
  `index.dhash_max_distance` bits. Every collapsed file stays listed under *duplicates*, so
  nothing is lost.
- Symlink export needs Developer Mode on Windows. Otherwise vision-tagger falls back to
  copying.
- InsightFace's pretrained `buffalo_l` models are licensed for **non-commercial research
  purposes** only.
- Face ages: the first faces run downloads AgeNet (430 KB, MIT license) from GitHub into
  `~/.insightface/models/agegender_faceapi`. The download is checked against a SHA-256 checksum
  and turned into an ONNX model locally. The **ages** stage re-estimates faces that were found
  before (no new face detection). It is added automatically whenever **faces** runs. Set
  `faces.age_model = "insightface"` to keep the old behaviour.
- Face ages are estimates. Expect about ±2 years for children and ±5–10 years for adults. Set
  the birth years of a few people for exact ages; vision-tagger then also corrects the others.
