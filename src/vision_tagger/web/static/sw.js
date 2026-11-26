// Mira service worker: makes the web UI an installable app.
// - the app shell (styles, scripts, icons) is cached and refreshed in the background,
// - pages come from the network; without a connection an offline page is shown,
// - thumbnails that were already seen are cached (at most MAX_THUMBS),
// - API calls, uploads and anything that is not GET are never cached.
const VERSION = "__VERSION__"; // replaced by the server: changes whenever the static files change
const SHELL = "vt-shell-" + VERSION;
const THUMBS = "vt-thumbs";
const MAX_THUMBS = 3000;
const SHELL_FILES = [
  "/static/app.css", "/static/app.js", "/static/htmx.min.js", "/static/alpine.min.js",
  "/static/favicon.svg", "/static/offline.html", "/static/icons/icon-192.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("vt-shell-") && k !== SHELL).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function trimThumbs() {
  const cache = await caches.open(THUMBS);
  const keys = await cache.keys();
  for (let i = 0; i < keys.length - MAX_THUMBS; i++) await cache.delete(keys[i]);
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/static/")) { // stale-while-revalidate
    event.respondWith(caches.open(SHELL).then(async (cache) => {
      const cached = await cache.match(req);
      const fresh = fetch(req).then((res) => { if (res.ok) cache.put(req, res.clone()); return res; }).catch(() => cached);
      return cached || fresh;
    }));
    return;
  }
  if (url.pathname.startsWith("/media/thumb/")) { // cache-first, bounded
    event.respondWith(caches.open(THUMBS).then(async (cache) => {
      const cached = await cache.match(req);
      if (cached) return cached;
      const res = await fetch(req);
      if (res.ok) { cache.put(req, res.clone()); trimThumbs(); }
      return res;
    }));
    return;
  }
  if (req.mode === "navigate") { // network first, offline page as the fallback
    event.respondWith(fetch(req).catch(() => caches.match("/static/offline.html")));
  }
});
