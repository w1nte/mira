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
- The images are fixed versions (tag and digest). To update Ollama, put a newer tag and its digest
  from Docker Hub into `docker-compose.yml`. Behind a reverse proxy on the same machine, publish
  the port on that machine only: `"127.0.0.1:8765:8765"`.
- The container upgrades its database on start (`serve --migrate`, with a backup) and pauses a
  running pipeline when it is stopped; it continues after the next start.
- Without Docker: set a password (`VT_WEB__PASSWORD`, see below), then
  `vision-tagger serve --remote --upload-dir /srv/photos --host 0.0.0.0`, or set
  `mode = "remote"` and `remote.upload_dir` in `config.toml`.

## Password and HTTPS

Set a password with the environment variable `VT_WEB__PASSWORD`. Remote mode doesn't start without
one, and neither does a server on an address other computers can reach (`--host 0.0.0.0`).
Every page then asks for it; a login lasts 30 days. Changing the password ends every login, and
*Database → Log out everywhere* ends them on all devices. After 10 wrong passwords an address
(IPv6: its /64 network) is blocked for 5 minutes; after 100 wrong passwords from any addresses,
new logins wait 5 minutes (logins made before keep working). Choose a long password: the
example from `.env.example` is refused, and one shorter than 12 characters gets a warning.

Mira counts wrong passwords per visitor. Behind a reverse proxy it only knows the visitor's
address if it trusts the proxy's `X-Forwarded-For` header: a proxy on the same computer
(`127.0.0.1`) is trusted. For a proxy in another container, set
`VT_WEB__TRUSTED_PROXIES='["172.18.0.0/16"]'` (its address or network); otherwise every visitor
shares the proxy's address and a few wrong passwords lock everyone out.

Put the server behind a reverse proxy with HTTPS (Caddy, Traefik, nginx), for example with Caddy:

```
photos.example.com {
    reverse_proxy localhost:8765
}
```

HTTPS protects the password on the way and lets browsers install the app. Allow uploads of any
size in the proxy (`client_max_body_size 0;` in nginx); each request is at most one 8 MB piece.
