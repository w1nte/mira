# Remote mode and Docker

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

## Docker

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Open `.env` in a text editor and choose a password (`VT_PASSWORD=…`), then:

```bash
docker compose up -d
```

Then open `http://<server>:8765`, log in and upload photos on the *Import* page. The `ollama`
service downloads the AI models the first time they are needed: click **Download now** on
*Home*, or let the first run do it (see [AI models](installation.md#ai-models)). By hand it is
`docker compose exec ollama ollama pull qwen2.5vl:7b` (and `nomic-embed-text`).

On Windows, don't create `.env` with `echo … > .env` in Windows PowerShell 5.1: it writes UTF-16,
and Docker Compose stops with `unexpected character "�" in variable name`. Copy the example and
edit it, or write it as plain text:
`Set-Content -Path .env -Value "VT_PASSWORD=…" -Encoding ascii`.

- [`docker-compose.yml`](../docker-compose.yml) starts Mira and Ollama. The volumes
  `vt-data` (database, thumbnails, models) and `vt-photos` (the uploaded photos) keep your data.
  To use a host folder, replace `vt-photos:/photos` with e.g. `./photos:/photos`.
- NVIDIA GPU: build with `EXTRA: gpu` and uncomment the two `deploy` sections (needs the NVIDIA
  Container Toolkit).
- The container upgrades its database on start (`serve --migrate`, with a backup) and pauses a
  running pipeline when it is stopped; it continues after the next start.
- Without Docker: `vision-tagger serve --remote --upload-dir /srv/photos --host 0.0.0.0`, or set
  `mode = "remote"` and `remote.upload_dir` in `config.toml`.

## Password and HTTPS

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
