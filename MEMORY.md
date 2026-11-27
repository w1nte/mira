# MEMORY.md

Decisions and lessons from earlier work that the code doesn't explain. Short on purpose: add
only what will matter again, and delete what no longer holds. Rules live in `CLAUDE.md`.

## Project
- The repository is `w1nte/mira` (renamed from `w1nte/vision-tagger`; GitHub redirects the old
  name). Users see "Mira". Technical names stay `vision-tagger`. Never change the XMP namespace
  URL in `xmp.py`: other programs have stored data under it.
- Work happens on feature branches `claude/<topic>`. The owner merges the PRs. To catch up,
  merge `main` into the branch; no rebase, no force-push.
- The README stays short (description, overview, setup). Everything else is in `docs/`;
  document new features there.
- The website is the orphan branch `gh-pages` (no project files) at `w1nte.github.io/mira/`.
  It is built from the owner's claude.ai design artifact, in 11 languages (`site.js`). Its facts
  must match the README and `docs/`; copy from there, not from memory.

## Releases and installers
- Publishing a GitHub release triggers `.github/workflows/desktop.yml`. It builds the Windows
  setups (cpu, gpu), the Linux tar.gz with `install.sh`, and the macOS `.pkg` (Apple silicon,
  macOS 15+). The version comes from the tag (`packaging/version.py`). Nothing is signed.
- If the changelog release step was forgotten, the workflow does it
  (`packaging/changelog.py release`) and opens a PR for `main`.
- Every installer updates in place: it quits Mira with `vision-tagger-cli quit-app` (the
  pipeline pauses), replaces the files, and runs `migrate --desktop`.
- Inno Setup (`installer.iss`, UTF-8 with BOM):
  - a line in `[Code]` that starts with `[` is read as a section;
  - `{ }` comments can't contain braces;
  - `test_packaging` guards both. ISCC only runs in CI.
- The gpu setup is a single `.exe` (about 285 MB). The workflow splits it into `.bin` files only
  above the 2 GB limit, and then the setup downloads the missing ones.
- The Linux build runs on ubuntu-22.04 (older glibc). The uv cache key ignores the OS version,
  so desktop builds use `cache-suffix: desktop-<runner>`. Release builds use no cache: an
  insightface wheel compiled on 24.04 broke the 22.04 bundle.

## App decisions
- Desktop data lives in `%LOCALAPPDATA%\vision-tagger` (macOS: `~/Library/Application Support/`,
  Linux: XDG). Closing the app pauses a running pipeline; the next start resumes it.
- Mira downloads its Ollama models itself (`vlm.auto_pull`). Never make `ollama pull` a required
  step in docs.
- The update check uses the GitHub releases API of `w1nte/mira`, once a day, and is on by
  default only in the frozen app. Release notes from GitHub are linked, never rendered.
- Search is a hybrid of LLM interpretation, visual similarity (SigLIP 2, stage `visual`),
  text-document embeddings, properties and keywords.
  - "or": alternatives inside a group are OR-ed, groups are AND-ed, exclusions stay strict.
  - URL `t=k:v|k:v` is one OR group; a repeated `t` is still AND, for old links.
  - Visual model choice (table in `docs/search.md`): SigLIP 2 B/16 with the fp32 image and fp16 text file.
    Its int8 files lose too much (int8 image vectors: cosine 0.8 to fp32), fp16 image is 2x
    slower on the CPU, nomic-embed-vision is English-only. Re-measure before switching.
  - The visual score is calibrated (`VisualModel.low/high`), not min-max: a request nothing
    matches must not push unrelated photos up. Vector cosine can't tell copies from similar
    shots; "Similar photos" uses the pHash for copies.
  - numpy converts float16 slowly: the visual matrix is float32 in memory (154 MB at 50k).
    onnxruntime keeps fp16 weights as fp32 on the CPU (text encoder ~0.8 GB; idle unload).
- Docker: the instructions say `cp .env.example .env`, because PowerShell 5.1's `echo > .env`
  writes UTF-16.
- Library tools (Database page): clearing imports an empty database through `db.transfer`
  (backup `bak-clear-*`, library folders kept); its second confirmation is the number of
  photos, typed. Removing missing photos and clearing are refused while a pipeline runs or is
  paused, like an import: clustering works on a snapshot and would bring back deleted faces.
- Photo paths in the database are untrusted (a database can be imported): every read of a
  photo's file goes through `access.photo_file` (image extension of the path itself; remote
  mode: below the upload folder). The location check treats refused paths as missing.
- "Show in folder" (`reveal.py`) is offered only in local mode to a request from this computer
  (loopback client, local Host, no proxy headers) and never takes a path from the request.

## Pitfalls we hit
- Windows: `socket.shutdown()` from another thread doesn't wake a blocked `recv`. Read in a
  thread and poll a queue (see `vlm/models.py` `pull`).
- Merging branches:
  - the i18n catalogs conflict: merge them key by key against the merge base, then run
    `i18n_extract update`;
  - look for silent duplicates (two rows or two msgids for the same thing);
  - check that changelog lines are still under `Unreleased` after a merge.
- Translations: reuse an existing msgid for the same text, e.g. "What’s new" with a typographic
  apostrophe. Don't add a second spelling of it.
- Playwright `page.route` pauses the page's timers during a navigation. Test loading states
  through a real delaying proxy.
- Parallel agents share this machine: kill servers by PID, never `pkill -f` by name.
- Importing OpenCV prepends its bundled libraries to `LD_LIBRARY_PATH`, also in a source
  install, and child processes inherit it. Programs Mira starts get a cleaned environment
  (`desktop.child_environment`, `reveal.environment`).
- SQLite reuses the highest id after a delete: caches keyed by ids (thumbnails, browser cache)
  must be cleared for deleted rows (`removal.py` renews the library id when that happens).
