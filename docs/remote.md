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

Put the server behind a reverse proxy with HTTPS. The quickest way is the Traefik setup that
comes with Mira (see [HTTPS with Traefik](#https-with-traefik) below). Any other proxy works too
(Caddy, nginx, …), for example with Caddy:

```
photos.example.com {
    reverse_proxy localhost:8765
}
```

HTTPS protects the password on the way and lets browsers install the app. Allow uploads of any
size in the proxy (`client_max_body_size 0;` in nginx); each request is at most one 8 MB piece.

Tell Mira to trust the proxy, so it sees each visitor's own address in the proxy's
`X-Forwarded-For` header: set the environment variable `FORWARDED_ALLOW_IPS` to the proxy's
address as Mira sees it, or to `*` if nothing but the proxy can reach Mira's port (with Docker,
publish it as `127.0.0.1:8765:8765`). Otherwise every visitor seems to come from the proxy, and 10
wrong passwords from anyone lock everyone out for 5 minutes. A proxy on the same computer
(`127.0.0.1`) is trusted without it. `web.trusted_proxies` (`VT_WEB__TRUSTED_PROXIES`, a list such
as `'["172.18.0.0/16"]'`) does the same and takes precedence.

## HTTPS with Traefik

[`docker-compose.traefik.yml`](../docker-compose.traefik.yml) puts
[Traefik](https://doc.traefik.io/traefik/) in front of Mira. Traefik answers on ports 80 and 443,
gets a free certificate for your domain from [Let's Encrypt](https://letsencrypt.org/), renews it
in time, and redirects `http://` to `https://`. Mira no longer opens a port of its own on the
server: Traefik is the only way in. The file adds to `docker-compose.yml`, which stays as it is.

### Before you start

- A domain or subdomain, e.g. `photos.example.com`, whose DNS record (A, and AAAA for IPv6)
  points at the server.
- Ports 80 and 443 reachable from the internet: open them in the firewall, and at home forward
  them on the router to the server. Let's Encrypt connects to port 443 to check that the domain
  is yours.
- No other program on the server using ports 80 and 443 (another web server or proxy); otherwise
  see [Other ports](#other-ports).
- Docker Compose 2.24 or newer (`docker compose version`).

### Setup

1. In `.env` (your copy of `.env.example`, next to `VT_PASSWORD`), remove the `# ` in front of
   these lines and fill in your domain and e-mail address:
   ```bash
   COMPOSE_FILE=docker-compose.yml:docker-compose.traefik.yml
   MIRA_DOMAIN=photos.example.com
   ACME_EMAIL=you@example.com
   ```
2. Start it (again):
   ```bash
   docker compose up -d
   ```
3. Open `https://photos.example.com` and log in. The first certificate takes up to a minute;
   until then the browser warns about Traefik's own certificate. `docker compose logs traefik`
   shows the certificate request and what went wrong, if anything.

`COMPOSE_FILE` makes every `docker compose` command use both files (`up`, `down`, `logs`,
`pull`, …). Without it, name both files each time:
`docker compose -f docker-compose.yml -f docker-compose.traefik.yml up -d`. On Windows, Docker
Compose separates the files with `;` instead of `:`.

### Settings in `.env`

| Variable | Default | What it does |
| --- | --- | --- |
| `MIRA_DOMAIN` | (required) | The domain Mira answers on. The certificate is issued for it. |
| `ACME_EMAIL` | (required) | Your e-mail address for Let's Encrypt, used only for notices about the certificate. |
| `COMPOSE_FILE` | not set | `docker-compose.yml:docker-compose.traefik.yml` uses the Traefik file with every command. |
| `ACME_CA_SERVER` | Let's Encrypt | Where certificates come from; Let's Encrypt's staging server for a first try (below). |
| `TRAEFIK_HTTP_PORT` | `80` | The server's port for `http://` (redirected to `https://`). |
| `TRAEFIK_HTTPS_PORT` | `443` | The server's port for `https://`. |

Change them in `.env`, then run `docker compose up -d` again.

### Trying it out first

Let's Encrypt allows only a few failed attempts per domain and hour. If the DNS record or the
port forwarding may not be ready yet, start with its staging server, whose certificates browsers
don't trust (they warn) but which has generous limits:

```bash
ACME_CA_SERVER=https://acme-staging-v02.api.letsencrypt.org/directory
```

Once Mira opens with the staging certificate, remove that line, delete the stored certificate and
start again:

```bash
docker compose down
docker volume rm mira_traefik-certs      # "docker volume ls" shows the name
docker compose up -d
```

### Other ports

If ports 80 and 443 are taken on the server, set for example `TRAEFIK_HTTP_PORT=8080` and
`TRAEFIK_HTTPS_PORT=8443`, and on the router forward the internet's ports 80 and 443 to them.
Let's Encrypt always connects to port 443, so the certificate can't be issued without that
forwarding.

### What the file sets up

- **Traefik** watches Docker (through `/var/run/docker.sock`, read-only) and publishes only
  containers labelled `traefik.enable=true`: Mira, not Ollama.
- **Mira's port 8765** is not published on the server (`ports: !reset []`), so nobody can bypass
  Traefik and its HTTPS.
- **Uploads:** a request may take up to 10 minutes. Traefik's default of 60 seconds would cut off
  an 8 MB upload piece on a slow connection. Traefik sets no size limit for requests.
- **Visitors' addresses:** `FORWARDED_ALLOW_IPS: "*"` lets Mira read the visitor's address from
  Traefik's `X-Forwarded-For` header. The login lock (10 wrong passwords block an address for 5
  minutes) then applies to whoever typed them; without it, everyone would come from Traefik's
  address and be locked out together. This is safe because only Traefik and Ollama can reach
  Mira's port.
- **The certificate** is kept in the volume `traefik-certs` and renewed by Traefik about 30 days
  before it expires. Mira's login cookie is only sent over HTTPS then.

### Already running Traefik?

Then use only Mira's part with your own Traefik. Leave `COMPOSE_FILE` out of `.env` and put this
into `docker-compose.override.yml` next to `docker-compose.yml` (Docker Compose reads that file
by itself), with your Traefik's network, entrypoint and certificate resolver names:

```yaml
services:
  mira:
    ports: !reset []
    environment:
      FORWARDED_ALLOW_IPS: "*"
    networks: [default, proxy]
    labels:
      - traefik.enable=true
      - traefik.docker.network=proxy
      - traefik.http.routers.mira.rule=Host(`photos.example.com`)
      - traefik.http.routers.mira.entrypoints=websecure
      - traefik.http.routers.mira.tls.certresolver=letsencrypt
      - traefik.http.services.mira.loadbalancer.server.port=8765

networks:
  proxy:
    external: true
```

For uploads over slow connections, raise the read timeout of that entrypoint in your Traefik
(`--entrypoints.websecure.transport.respondingTimeouts.readTimeout=10m`).
