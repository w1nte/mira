# Installation

Mira runs as a desktop app (Windows, macOS, Linux), from a source checkout, or on a server with Docker (see [Remote mode and Docker](remote.md)).

## Desktop app (Windows, macOS, Linux)

The simplest way: download the app for your system from the
[releases](https://github.com/w1nte/mira/releases) and install it. No administrator
rights are needed on Windows and Linux.

| System | File | Flavours |
| --- | --- | --- |
| Windows 10/11 (x64) | `Mira-setup-<version>-<flavour>.exe` | cpu, gpu |
| macOS 15 or newer, Apple silicon | `Mira-<version>-macos-arm64.pkg` | cpu |
| Linux x86_64 (glibc 2.35+, e.g. Ubuntu 22.04, Debian 12, Fedora 36) | `Mira-<version>-linux-x86_64-<flavour>.tar.gz` | cpu, gpu |

- **gpu**: for an NVIDIA graphics card (CUDA and cuDNN are included; needs driver R580 or newer).
- **cpu**: for every other computer (slower face detection, much smaller download).

The notes of each release end with a *Downloads* section: the SHA-256 checksum of every file and
a link to its [VirusTotal](https://www.virustotal.com) scan. To check a download, compare its
checksum with `certutil -hashfile <file> SHA256` (Windows), `shasum -a 256 <file>` (macOS) or
`sha256sum <file>` (Linux).

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
unpacked the same way. (Should an archive ever exceed GitHub's limit of 2 GB per file, the release
splits it into `….tar.gz.partNN` files; unpack those with
`cat Mira-<version>-linux-x86_64-gpu.tar.gz.part* | tar -xz`.) On Linux, Mira's window is Chrome, Chromium or Edge in app mode (the app does not
bundle GTK or Qt for a native window); without one of them it opens in your default browser.

**Updating.** Mira tells you when a new version is out (see [Updates](#update-check)). Run the new
setup program, the new `.pkg`, or the new `install.sh`. They detect the
installed version and update it: a running Mira is asked to quit first (a running pipeline is
paused and continues on the next start), the program files are replaced, and your library is
kept and its database migrated to the new version (a backup copy is made first,
`vision_tagger.sqlite3.bak-<version>-<time>`). If the migration fails, the installation is kept
and Mira tries again when it starts; the details are in `logs/migrate.log` in the data folder.
What changed in each version is in the [changelog](../src/vision_tagger/resources/CHANGELOG.md), and in the app under
*What's new* (sidebar).

Where things are:

| | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Library (`config.toml`, `data/`) | `%LOCALAPPDATA%\vision-tagger` | `~/Library/Application Support/vision-tagger` | `~/.local/share/vision-tagger` |
| Command line | `vision-tagger-cli.exe` in the install folder | `/Applications/Mira.app/Contents/MacOS/vision-tagger-cli` | `vision-tagger-cli` |
| Logs (also under *System → Log* in the app) | `data\logs\vision-tagger.log` in the library | `data/logs/vision-tagger.log` in the library | `data/logs/vision-tagger.log` in the library |

The library (settings, database, thumbnails) stays when you update or uninstall the app;
settings are in its `config.toml`. The command line works like `vision-tagger` from a source
checkout (`vision-tagger-cli index D:\Photos`, `vision-tagger-cli run`, …); `vision-tagger-cli
quit-app` closes a running app gracefully and `vision-tagger-cli migrate --desktop` migrates the
app's library (the installers use both).

From a source checkout the same window opens with `uv run vision-tagger desktop` (add
`--extra desktop` to `uv sync` for a native window; without it, Edge or Chrome opens in app mode).
The apps are built by [`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml) from
[`packaging`](../packaging): PyInstaller makes the app ([`vision-tagger.spec`](../packaging/vision-tagger.spec)),
its self-test checks it, and Inno Setup ([`windows`](../packaging/windows)), `pkgbuild`
([`macos`](../packaging/macos)) or a tar archive with an install script ([`linux`](../packaging/linux))
packs it. See [Releases](development.md#releases).

## From source (Windows + NVIDIA GPU)

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
   also works) and `nomic-embed-text` for the search index. The visual search model
   (SigLIP 2, 970 MB) is downloaded by Mira itself on the first run (see
   [Visual search](search.md#visual-search)).
5. **Mira:**
   ```powershell
   git clone https://github.com/w1nte/mira.git
   cd mira
   uv sync --extra gpu           # Python 3.12 is installed automatically by uv
   uv run vision-tagger init     # creates config.toml and the database
   ```

For CPU-only machines (or Linux/macOS) use `uv sync --extra cpu` and set
`providers = ["CPUExecutionProvider"]` in `config.toml`.

## AI models

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

## Update check

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
