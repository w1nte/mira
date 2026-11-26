// Translations for scripts: the page carries the current language's strings (i18n.JS_KEYS).
window.vtT = function (key, vars) {
  if (!window.__vtI18n) {
    try { window.__vtI18n = JSON.parse(document.getElementById("vt-i18n").textContent); } catch (e) { window.__vtI18n = {}; }
  }
  const text = window.__vtI18n[key] || key;
  return text.replace(/%\((\w+)\)s/g, (m, name) => (vars && name in vars ? vars[name] : m));
};
window.vtNum = (n) => Number(n).toLocaleString(document.documentElement.lang || undefined);

// True while the user types into a text field (keyboard shortcuts must not fire then).
// Checkboxes, radios and buttons don't count: focus often lands there after a click.
window.vtIsTyping = function (el) {
  if (!el || !el.tagName) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag !== "INPUT") return false;
  return !["checkbox", "radio", "button", "submit", "reset", "file", "color", "image"].includes(
    (el.type || "").toLowerCase()
  );
};

// Loading indicator: a thin bar at the top of the window while a page, or an HTMX request the
// user started, is loading (with a running pipeline a page can take a few seconds). It shows
// only after a short delay, so fast loads don't flash. Not counted: polling and lazy-loading
// requests (hx-trigger with every / load / revealed / intersect), downloads (<a download>),
// new tabs, and anything inside [data-no-loading].
(function () {
  const DELAY_MS = 150;
  const GIVE_UP_MS = 60000; // a navigation that turned into a download or was cancelled
  const BACKGROUND = /\b(every|load|revealed|intersect)\b/;
  const root = document.documentElement;
  const pending = new Set();
  let navigating = false;
  let showTimer = null;
  let giveUpTimer = null;

  function update() {
    const busy = navigating || pending.size > 0;
    const main = document.getElementById("main");
    if (busy) {
      if (!showTimer && !root.classList.contains("is-loading")) {
        showTimer = setTimeout(function () {
          showTimer = null;
          root.classList.add("is-loading");
          root.classList.toggle("is-navigating", navigating);
        }, DELAY_MS);
      } else if (navigating) {
        root.classList.toggle("is-navigating", root.classList.contains("is-loading"));
      }
      if (main) main.setAttribute("aria-busy", "true");
    } else {
      clearTimeout(showTimer);
      showTimer = null;
      root.classList.remove("is-loading", "is-navigating");
      if (main) main.removeAttribute("aria-busy");
    }
  }

  function startNavigation() {
    navigating = true;
    clearTimeout(giveUpTimer);
    giveUpTimer = setTimeout(stop, GIVE_UP_MS);
    update();
  }

  function stop() {
    navigating = false;
    pending.clear();
    clearTimeout(giveUpTimer);
    update();
  }

  window.vtLoading = { start: startNavigation, stop: stop };

  function leavesPage(a, ev) {
    if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return false;
    if (a.hasAttribute("download") || a.closest("[data-no-loading]")) return false;
    const target = a.getAttribute("target");
    if (target && target !== "_self") return false;
    const href = a.getAttribute("href") || "";
    if (!href || href.charAt(0) === "#" || /^(javascript|mailto|tel):/i.test(href)) return false;
    const url = new URL(a.href, window.location.href);
    if (url.origin !== window.location.origin) return false;
    const here = window.location;
    return !(url.hash && url.pathname === here.pathname && url.search === here.search);
  }

  // checked after the event has run: handlers that call preventDefault() (HTMX links, Alpine
  // @click.prevent, cancelled confirmations) keep the page, so there is nothing to wait for
  document.addEventListener("click", function (ev) {
    const a = ev.target.closest && ev.target.closest("a[href]");
    if (!a || !leavesPage(a, ev)) return;
    setTimeout(function () {
      if (!ev.defaultPrevented) startNavigation();
    }, 0);
  });
  document.addEventListener("submit", function (ev) {
    const form = ev.target;
    if (form.closest("[data-no-loading]")) return;
    const target =
      (ev.submitter && ev.submitter.getAttribute("formtarget")) || form.getAttribute("target");
    if (target && target !== "_self") return;
    setTimeout(function () {
      if (!ev.defaultPrevented) startNavigation();
    }, 0);
  });

  document.addEventListener("htmx:beforeSend", function (ev) {
    const elt = ev.detail.elt;
    const xhr = ev.detail.xhr;
    if (!elt || !xhr || elt.closest("[data-no-loading]")) return;
    if (BACKGROUND.test(elt.getAttribute("hx-trigger") || "")) return;
    pending.add(xhr);
    // on the request itself: the element may be swapped out before HTMX's own events fire
    xhr.addEventListener("loadend", function () {
      if (pending.delete(xhr)) update();
    });
    update();
  });

  // back / forward cache: the page comes back as it was left, bar included
  window.addEventListener("pageshow", stop);
})();

// Theme switch (System / Light / Dark). The server renders <html data-theme> from the
// vt_theme cookie, so pages never flash; a click only stores the new choice.
(function () {
  const THEMES = ["system", "light", "dark"];
  function select(btn) {
    const theme = btn.dataset.themeSet;
    if (!THEMES.includes(theme)) return;
    document.documentElement.dataset.theme = theme;
    document.cookie = "vt_theme=" + theme + "; path=/; max-age=31536000; SameSite=Lax";
    document.querySelectorAll("[data-theme-set]").forEach(function (b) {
      const on = b.dataset.themeSet === theme;
      b.setAttribute("aria-checked", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
  }
  document.addEventListener("click", function (ev) {
    const btn = ev.target.closest && ev.target.closest("[data-theme-set]");
    if (btn) select(btn);
  });
  // radio group keyboard behaviour: arrow keys move the choice
  document.addEventListener("keydown", function (ev) {
    const btn = ev.target.closest && ev.target.closest("[data-theme-set]");
    if (!btn) return;
    const step = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[ev.key];
    if (!step) return;
    ev.preventDefault();
    const all = Array.from(btn.parentElement.querySelectorAll("[data-theme-set]"));
    const next = all[(all.indexOf(btn) + step + all.length) % all.length];
    select(next);
    next.focus();
  });
})();

// "/" focuses the search field in the top bar, Escape leaves it.
document.addEventListener("keydown", function (ev) {
  if (ev.defaultPrevented) return;
  if (ev.key === "/" && !ev.ctrlKey && !ev.metaKey && !ev.altKey && !vtIsTyping(ev.target)) {
    const input = document.querySelector(".navsearch input");
    if (input) {
      ev.preventDefault();
      input.focus();
      input.select();
    }
  } else if (ev.key === "Escape" && ev.target.matches && ev.target.matches(".navsearch input")) {
    ev.target.blur();
  }
});

// <select data-autosubmit> / <input data-autosubmit>: submit the form when the value changes.
document.addEventListener("change", function (ev) {
  const el = ev.target;
  if (el.matches && el.matches("select[data-autosubmit], input[data-autosubmit]") && el.form) {
    el.form.requestSubmit();
  }
});

// <a data-hotkey="ArrowLeft|ArrowRight">: previous / next photo with the arrow keys.
document.addEventListener("keydown", function (ev) {
  if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
  if (ev.defaultPrevented || ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
  if (vtIsTyping(ev.target)) return;
  const link = document.querySelector('a[data-hotkey="' + ev.key + '"]');
  if (link) {
    ev.preventDefault();
    link.click();
  }
});

// Dropdowns (<details class="dropdown">): close on a click outside and on Escape.
document.addEventListener("click", function (ev) {
  document.querySelectorAll("details.dropdown[open]").forEach(function (d) {
    if (!d.contains(ev.target)) d.removeAttribute("open");
  });
});
document.addEventListener("keydown", function (ev) {
  if (ev.key !== "Escape") return;
  document.querySelectorAll("details.dropdown[open]").forEach(function (d) {
    d.removeAttribute("open");
    if (d.contains(document.activeElement)) d.querySelector("summary").focus();
  });
});

// Shift-click range selection for checkbox grids (faces, clusters, property values).
(function () {
  let last = null;
  document.addEventListener("click", function (ev) {
    const box = ev.target.closest("input[type=checkbox].pick");
    if (!box) return;
    const scope = box.closest("[data-select-scope]") || document;
    const boxes = Array.from(scope.querySelectorAll("input[type=checkbox].pick"));
    if (ev.shiftKey && last && last !== box && boxes.includes(last)) {
      const [a, b] = [boxes.indexOf(last), boxes.indexOf(box)].sort((x, y) => x - y);
      for (let i = a; i <= b; i++) boxes[i].checked = box.checked;
      scope.dispatchEvent(new Event("change", { bubbles: true }));
    }
    last = box;
  });
})();

// Gallery facets: Ctrl+click (⌘+click) on a value adds it as an alternative ("or") to the
// condition on the same property, instead of requiring it as well (link in data-or-href).
document.addEventListener("click", function (ev) {
  const link = ev.target.closest("a[data-or-href]");
  if (!link || ev.button !== 0 || !(ev.ctrlKey || ev.metaKey) || ev.shiftKey || ev.altKey) return;
  ev.preventDefault();
  window.location.href = link.dataset.orHref;
});

// Confirmation dialogs: <form data-confirm="..."> or <button data-confirm="...">.
// The text lives in a data attribute so user data never ends up inside JavaScript code.
document.addEventListener(
  "submit",
  function (ev) {
    const msg =
      (ev.submitter && ev.submitter.dataset.confirm) || ev.target.dataset.confirm;
    if (msg && !window.confirm(msg)) {
      ev.preventDefault();
      ev.stopImmediatePropagation();
    }
  },
  true
);

// NSFW covers (web/nsfw.py): an NSFW photo is a black tile (<img src="/static/nsfw.svg"
// data-nsfw="<photo id>" data-nsfw-src="<its URL>">). The photo is only asked for when its cover
// is clicked: the first click on a covered thumbnail reveals it (every picture of that photo on
// the page) instead of opening it. Revealed photos stay revealed for the browser session.
window.vtNsfw = (function () {
  const KEY = "vt-nsfw-revealed";
  const MAX = 5000;
  function revealed() {
    try { return new Set(JSON.parse(sessionStorage.getItem(KEY) || "[]").map(String)); } catch (e) { return new Set(); }
  }
  function remember(id) {
    try {
      const ids = revealed();
      ids.delete(id);
      ids.add(id);
      sessionStorage.setItem(KEY, JSON.stringify(Array.from(ids).slice(-MAX)));
    } catch (e) { /* private mode, storage full: revealed for this page only */ }
  }
  function show(img) {
    const src = img.getAttribute("data-nsfw-src");
    if (!src) return;
    img.removeAttribute("data-nsfw-src");
    img.removeAttribute("title");
    img.alt = "";
    img.src = src;
  }
  function reveal(id) {
    remember(id);
    document.querySelectorAll("img[data-nsfw-src]").forEach(function (img) {
      if (img.dataset.nsfw === id) show(img);
    });
  }
  function restore(root) {
    const ids = revealed();
    if (!ids.size || !root || !root.querySelectorAll) return;
    root.querySelectorAll("img[data-nsfw-src]").forEach(function (img) {
      if (ids.has(img.dataset.nsfw)) show(img);
    });
  }
  // the covered picture a click is meant for: the cover itself, or the only picture of the
  // link / label around it (a thumbnail's caption covers part of the picture)
  function target(ev) {
    if (!ev.target.closest) return null;
    const img = ev.target.closest("img[data-nsfw-src]");
    if (img) return img;
    const host = ev.target.closest("a, label");
    if (!host) return null;
    const imgs = host.querySelectorAll("img");
    return imgs.length === 1 && imgs[0].hasAttribute("data-nsfw-src") ? imgs[0] : null;
  }
  document.addEventListener("click", function (ev) {
    if (ev.button !== 0 || ev.target.closest && ev.target.closest("input, button, .pickbox")) return;
    const img = target(ev);
    if (!img) return;
    const scope = img.closest("[data-select-scope]");
    if (scope && (scope.classList.contains("selecting") || scope.querySelector("input.pick:checked"))) return; // selects
    ev.preventDefault();
    ev.stopImmediatePropagation();
    reveal(img.dataset.nsfw);
  }, true);
  // "Show NSFW photos without asking": a cookie of this browser (read by the server)
  document.addEventListener("change", function (ev) {
    const box = ev.target.closest && ev.target.closest("input[data-nsfw-show]");
    if (!box) return;
    document.cookie = box.checked ? "vt_nsfw=show; path=/; max-age=31536000; SameSite=Lax" : "vt_nsfw=; path=/; max-age=0; SameSite=Lax";
    window.vtLoading.start();
    window.location.reload();
  });
  restore(document);
  document.addEventListener("htmx:load", function (ev) { restore(ev.detail && ev.detail.elt); });
  // a covered picture made by a script (map popups): the same markup as web/nsfw.py writes
  function cover(img, src, id) {
    img.src = "/static/nsfw.svg";
    img.dataset.nsfw = String(id);
    img.setAttribute("data-nsfw-src", src);
    img.alt = "NSFW";
    img.title = vtT("Not safe for work – click to show");
    if (revealed().has(String(id))) show(img);
  }
  return { reveal: reveal, restore: restore, cover: cover };
})();

// Gallery selection: while photos are ticked (or after "Select"), a click on a photo ticks it
// instead of opening it.
document.addEventListener("click", function (ev) {
  if (ev.button !== 0 || !ev.target.closest) return;
  const tile = ev.target.closest(".tile");
  if (!tile || ev.target.closest("input")) return;
  const scope = tile.closest("[data-select-scope]");
  const box = tile.querySelector("input.pick");
  if (!scope || !box || !(scope.classList.contains("selecting") || scope.querySelector("input.pick:checked"))) return;
  ev.preventDefault();
  box.click();
});

// Helpers used from Alpine components.
window.vt = {
  count(scope) {
    return scope.querySelectorAll("input[type=checkbox].pick:checked").length;
  },
  toggleAll(scope, on) {
    scope.querySelectorAll("input[type=checkbox].pick").forEach((b) => (b.checked = on));
  },
};

// Quiz keyboard shortcuts: 1-4 pick an answer, Enter = next / skip.
document.addEventListener("keydown", function (ev) {
  const card = document.getElementById("quiz-card");
  if (!card || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  const tag = (ev.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  if (/^[1-9]$/.test(ev.key)) {
    const btn = card.querySelector('[data-key="' + ev.key + '"]');
    if (btn) { ev.preventDefault(); btn.click(); }
  } else if (ev.key === "Enter") {
    const next = card.querySelector("[data-next]");
    if (next) { ev.preventDefault(); next.click(); }
  }
});

// Generations: grow-up viewer. Two stacked, eye-aligned portraits are cross-faded while the
// age slider moves; while playing, each age is held briefly and then morphs into the next.
// Frame data comes from a <script type="application/json"> block (no data in inline JS).
window.vtGrowup = function (dataId) {
  return {
    frames: JSON.parse(document.getElementById(dataId).textContent),
    pos: 0,
    playing: false,
    speed: 1,
    _raf: null,
    _last: 0,
    get i() { return Math.max(0, Math.min(Math.floor(this.pos), this.frames.length - 1)); },
    get j() { return Math.min(this.i + 1, this.frames.length - 1); },
    get fade() {
      const f = this.pos - this.i;
      return Math.min(1, Math.max(0, (f - 0.35) / 0.5)); // hold, then cross-fade
    },
    get label() { return this.frames[Math.round(this.pos)].label; },
    init() {
      this.frames.forEach((f) => { const im = new Image(); im.src = f.src; });
      setTimeout(() => this.play(), 600);
    },
    toggle() { this.playing ? this.stop() : this.play(); },
    play() {
      if (this.playing) return;
      if (this.pos >= this.frames.length - 1) this.pos = 0;
      this.playing = true;
      this._last = performance.now();
      const step = (t) => {
        if (!this.playing) return;
        // rAF timestamps can be slightly older than performance.now(): never step backwards
        const dt = Math.max(0, Math.min(0.1, (t - this._last) / 1000));
        this._last = t;
        const next = this.pos + dt * 0.7 * this.speed;
        if (next >= this.frames.length - 1) {
          this.pos = this.frames.length - 1;
          this.playing = false;
          return;
        }
        this.pos = next;
        this._raf = requestAnimationFrame(step);
      };
      this._raf = requestAnimationFrame(step);
    },
    stop() { this.playing = false; if (this._raf) cancelAnimationFrame(this._raf); },
  };
};

// Images: timeline slider. Click a month for the start, click again for the end (a preview
// follows the pointer in between). Drag a handle to move start or end, drag inside the band to
// move the whole range, or press and sweep across the bars. Handles also work with the
// keyboard (arrows: one month, Shift: one year). The range becomes the from/to filter.
window.vtTimeline = function (dataId) {
  const data = JSON.parse(document.getElementById(dataId).textContent);
  const monthName = new Intl.DateTimeFormat(document.documentElement.lang || undefined, { month: "short", year: "numeric" });
  const months = data.months;
  const n = months.length;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const indexOf = (month, fallback) => {
    if (!month) return null;
    if (month < months[0]) return fallback === "end" ? null : 0;
    if (month > months[n - 1]) return fallback === "start" ? null : n - 1;
    const i = months.indexOf(month);
    return i >= 0 ? i : null;
  };
  let start = indexOf(data.from, "start");
  let end = indexOf(data.to, "end");
  if (data.from && !data.to && start !== null) end = n - 1;
  if (data.to && !data.from && end !== null) start = 0;
  if (start === null || end === null) start = end = null;
  return {
    months,
    counts: data.counts,
    n,
    start,
    end,
    pending: null, // first click of a click-click selection
    hover: null,
    drag: null,
    _timer: null,
    get max() { return Math.max(1, ...this.counts); },
    get hasSel() { return this.start !== null && this.end !== null; },
    get band() {
      let a = this.start, b = this.end;
      if (this.pending !== null) {
        a = Math.min(this.pending, this.hover ?? this.pending);
        b = Math.max(this.pending, this.hover ?? this.pending);
      }
      if (a === null || b === null) return null;
      return { left: (a / n) * 100, width: ((b - a + 1) / n) * 100 };
    },
    get ticks() {
      const years = [];
      months.forEach((m, i) => { if (i === 0 || m.endsWith("-01")) years.push({ i, year: m.slice(0, 4) }); });
      const width = this.$refs.track ? this.$refs.track.clientWidth : 800;
      const step = Math.max(1, Math.ceil(years.length / Math.max(1, Math.floor(width / 56))));
      return years.filter((_, k) => k % step === 0);
    },
    get summary() {
      if (this.pending !== null) return this.label(this.pending) + " – …";
      if (!this.hasSel) return vtT("All time · %(n)s photos", { n: vtNum(this.total(0, n - 1)) });
      const range = this.start === this.end ? this.label(this.start) : this.label(this.start) + " – " + this.label(this.end);
      return vtT("%(range)s · %(n)s photos", { range, n: vtNum(this.total(this.start, this.end)) });
    },
    get hint() {
      if (this.pending !== null) return vtT("Click the last month (Esc cancels)");
      return this.hasSel ? vtT("Drag the band or its edges") : vtT("Click the first month, then the last");
    },
    label(i) { const m = months[i]; return monthName.format(new Date(+m.slice(0, 4), +m.slice(5, 7) - 1, 15)); },
    total(a, b) { let t = 0; for (let i = a; i <= b; i++) t += this.counts[i]; return t; },
    pos(i, offset = 0.5) { return ((i + offset) / n) * 100; },
    barHeight(c) { return c ? Math.max(4, (c / this.max) * 100) : 0; },
    inSel(k) {
      const band = this.band;
      if (!band) return false;
      const a = Math.round((band.left / 100) * n);
      return k >= a && k < a + Math.round((band.width / 100) * n);
    },
    idx(ev) {
      const r = this.$refs.track.getBoundingClientRect();
      return clamp(Math.floor(((ev.clientX - r.left) / r.width) * n), 0, n - 1);
    },
    down(ev) {
      if (ev.button !== 0) return;
      const i = this.idx(ev);
      const handle = ev.target.closest("[data-handle]");
      this.$refs.track.setPointerCapture(ev.pointerId);
      if (handle && this.pending === null) {
        this.drag = { kind: handle.dataset.handle, moved: false };
      } else if (this.hasSel && this.pending === null && i >= this.start && i <= this.end) {
        this.drag = { kind: "move", origin: i, s0: this.start, e0: this.end, x0: ev.clientX, moved: false };
      } else {
        this.drag = { kind: "sweep", origin: i, x0: ev.clientX, moved: false };
      }
    },
    move(ev) {
      const i = this.idx(ev);
      this.hover = i;
      const d = this.drag;
      if (!d) return;
      if (!d.moved && d.x0 !== undefined && Math.abs(ev.clientX - d.x0) < 4) return; // still a click
      d.moved = true;
      if (d.kind === "sweep") {
        this.pending = null;
        this.start = Math.min(d.origin, i);
        this.end = Math.max(d.origin, i);
      } else if (d.kind === "start") {
        this.start = Math.min(i, this.end);
      } else if (d.kind === "end") {
        this.end = Math.max(i, this.start);
      } else if (d.kind === "move") {
        const len = d.e0 - d.s0;
        this.start = clamp(d.s0 + i - d.origin, 0, n - 1 - len);
        this.end = this.start + len;
      }
    },
    up(ev) {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.moved) return this.apply();
      const i = this.idx(ev); // a click: first one sets the start, second one the end
      if (this.pending === null) {
        this.pending = i;
        this.hover = i;
      } else {
        this.start = Math.min(this.pending, i);
        this.end = Math.max(this.pending, i);
        this.pending = null;
        this.apply();
      }
    },
    key(ev, which) {
      const step = ev.shiftKey ? 12 : 1;
      const delta = { ArrowLeft: -step, ArrowDown: -step, ArrowRight: step, ArrowUp: step }[ev.key];
      if (ev.key === "Home" || ev.key === "End") {
        this[which] = ev.key === "Home" ? (which === "start" ? 0 : this.start) : (which === "end" ? n - 1 : this.end);
      } else if (delta !== undefined) {
        if (which === "start") this.start = clamp(this.start + delta, 0, this.end);
        else this.end = clamp(this.end + delta, this.start, n - 1);
      } else {
        return;
      }
      ev.preventDefault();
      clearTimeout(this._timer);
      this._timer = setTimeout(() => this.apply(), 700); // apply once the keys rest
    },
    cancel() { this.pending = null; },
    clear() { this.start = this.end = null; this.apply(); },
    apply() {
      const params = new URLSearchParams(data.base);
      params.delete("from");
      params.delete("to");
      if (this.hasSel && !(this.start === 0 && this.end === n - 1)) {
        params.set("from", months[this.start]);
        params.set("to", months[this.end]);
      }
      const qs = params.toString();
      window.vtLoading.start();
      window.location.assign("/images" + (qs ? "?" + qs : ""));
    },
  };
};

// Language switcher: remember the choice in a cookie (read by the server) and reload.
document.addEventListener("change", function (ev) {
  const sel = ev.target.closest("select[data-lang-switch]");
  if (!sel) return;
  document.cookie = "vt_lang=" + encodeURIComponent(sel.value) + "; path=/; max-age=31536000; SameSite=Lax";
  window.vtLoading.start();
  window.location.reload();
});

// Import (local mode): pick a folder on this computer. Lists folders only.
window.vtFolderPicker = function () {
  return {
    open: false, path: "", parent: null, folders: [], images: 0, typed: "", loading: false,
    get info() { return vtT("Photos in this folder: %(n)s", { n: vtNum(this.images) }); },
    start() { this.open = true; this.go(""); },
    async go(path) {
      this.loading = true;
      try {
        const r = await fetch("/api/fs?path=" + encodeURIComponent(path || ""));
        if (!r.ok) throw new Error(await r.text());
        const d = await r.json();
        Object.assign(this, { path: d.path, parent: d.parent, folders: d.folders, images: d.images });
        this.typed = d.path;
      } catch (e) {
        this.folders = [];
      } finally {
        this.loading = false;
      }
    },
  };
};

// Import (remote mode): resumable uploads. The queue lives in IndexedDB (metadata in "queue",
// the files themselves in "files"), so closing the page loses nothing: when it is opened
// again, every unfinished file continues where the server stopped receiving it.
(function () {
  const DB = "vision-tagger-uploads";
  let dbPromise = null;
  function db() {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB, 1);
        req.onupgradeneeded = () => {
          req.result.createObjectStore("queue", { keyPath: "id" });
          req.result.createObjectStore("files");
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  }
  async function tx(stores, mode, fn) {
    const d = await db();
    return new Promise((resolve, reject) => {
      const t = d.transaction(stores, mode);
      const result = fn(t);
      t.oncomplete = () => resolve(result && result.result !== undefined ? result.result : result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }
  const idb = {
    all: () => tx(["queue"], "readonly", (t) => t.objectStore("queue").getAll()),
    putMeta: (item) => tx(["queue"], "readwrite", (t) => { t.objectStore("queue").put(item); }),
    addMany: (pairs) => tx(["queue", "files"], "readwrite", (t) => {
      for (const [item, file] of pairs) { t.objectStore("queue").put(item); t.objectStore("files").put(file, item.id); }
    }),
    file: (id) => tx(["files"], "readonly", (t) => t.objectStore("files").get(id)),
    dropFile: (id) => tx(["files"], "readwrite", (t) => { t.objectStore("files").delete(id); }),
    remove: (ids) => tx(["queue", "files"], "readwrite", (t) => {
      for (const id of ids) { t.objectStore("queue").delete(id); t.objectStore("files").delete(id); }
    }),
  };
  const OK = /\.(jpe?g|png|webp|heic|heif|tiff?|bmp|gif|zip)$/i;
  const PARALLEL = 3;

  async function entriesOf(entry, prefix, out) { // drag & drop of folders
    if (entry.isFile) {
      await new Promise((res) => entry.file((f) => { out.push([f, prefix + f.name]); res(); }, res));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res) => reader.readEntries(res, () => res([])));
        for (const e of batch) await entriesOf(e, prefix + entry.name + "/", out);
      } while (batch.length);
    }
  }

  window.vtUploader = function (configId) {
    const cfg = JSON.parse(document.getElementById(configId).textContent);
    return {
      items: [], paused: false, dragging: false, running: 0, storageNote: "", skipped: 0,
      get total() { return this.items.length; },
      get done() { return this.items.filter((i) => i.state === "done").length; },
      get failed() { return this.items.filter((i) => i.state === "failed").length; },
      get active() { return this.items.filter((i) => i.state === "uploading").length; },
      get waiting() { return this.items.filter((i) => i.state === "queued").length; },
      get pending() { return this.waiting + this.active; },
      get percent() {
        let size = 0, got = 0;
        for (const i of this.items) { size += i.size; got += i.state === "done" ? i.size : i.received; }
        return size ? Math.round((100 * got) / size) : 0;
      },
      get summary() {
        const vars = { done: vtNum(this.done), total: vtNum(this.total) };
        let text = vtT(this.paused ? "%(done)s of %(total)s files uploaded – paused" : "%(done)s of %(total)s files uploaded", vars);
        if (this.skipped) text += " · " + vtT("%(n)s files skipped (not photos)", { n: vtNum(this.skipped) });
        return text;
      },
      get visible() { // active first, then failed, then a few waiting / done
        const rank = { uploading: 0, failed: 1, queued: 2, done: 3 };
        return [...this.items].sort((a, b) => rank[a.state] - rank[b.state] || a.added - b.added).slice(0, 12);
      },
      fmt(bytes) {
        const u = ["B", "KB", "MB", "GB"]; let i = 0; let v = bytes;
        while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
        return v.toLocaleString(undefined, { maximumFractionDigits: 1 }) + " " + u[i];
      },
      async init() {
        if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
        const stored = await idb.all();
        for (const item of stored) if (item.state === "uploading") item.state = "queued";
        this.items = stored;
        this.pump();
        this.updateStorage();
      },
      async updateStorage() {
        if (!navigator.storage || !navigator.storage.estimate) return;
        const e = await navigator.storage.estimate();
        this.storageNote = this.pending ? vtT("Browser storage for waiting uploads: %(used)s of %(quota)s", { used: this.fmt(e.usage || 0), quota: this.fmt(e.quota || 0) }) : "";
      },
      pick(ev) { this.add([...ev.target.files].map((f) => [f, f.webkitRelativePath || f.name])); ev.target.value = ""; },
      async drop(ev) {
        this.dragging = false;
        const list = [];
        const entries = [...ev.dataTransfer.items].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
        if (entries.length) { for (const e of entries) await entriesOf(e, "", list); }
        else for (const f of ev.dataTransfer.files) list.push([f, f.name]);
        this.add(list);
      },
      async add(list) {
        const pairs = [];
        const now = Date.now();
        list.forEach(([file, name], k) => {
          if (!OK.test(name)) { this.skipped++; return; }
          pairs.push([{ id: crypto.randomUUID(), name, size: file.size, received: 0, state: "queued", serverId: null, error: "", added: now + k }, file]);
        });
        for (let k = 0; k < pairs.length; k += 200) await idb.addMany(pairs.slice(k, k + 200)); // copies into browser storage
        this.items.push(...pairs.map(([item]) => item));
        this.pump();
        this.updateStorage();
      },
      pause() { this.paused = true; },
      resume() { this.paused = false; this.pump(); },
      async clearFinished() {
        const ids = this.items.filter((i) => i.state === "done" || i.state === "failed").map((i) => i.id);
        await idb.remove(ids);
        this.items = this.items.filter((i) => !ids.includes(i.id));
        this.skipped = 0;
      },
      pump() {
        while (!this.paused && this.running < PARALLEL) {
          const next = this.items.find((i) => i.state === "queued" && !i.backoff);
          if (!next) break;
          this.running++;
          this.upload(next).finally(() => { this.running--; this.pump(); this.updateStorage(); });
        }
      },
      async api(url, opts = {}) {
        const r = await fetch(url, opts);
        if (!r.ok) { const e = new Error((await r.text()) || r.statusText); e.status = r.status; throw e; }
        return r.status === 204 ? null : r.json();
      },
      async upload(item) {
        item.state = "uploading";
        try {
          const file = await idb.file(item.id);
          if (!file) throw Object.assign(new Error(vtT("File no longer available")), { status: 400 });
          let state = null;
          if (item.serverId) {
            try { state = await this.api("/api/uploads/" + item.serverId); } catch (e) { if (e.status !== 404) throw e; }
          }
          if (!state) {
            state = await this.api("/api/uploads", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: item.name, size: item.size }) });
            item.serverId = state.id;
          }
          item.received = state.received;
          await idb.putMeta({ ...item });
          while (item.received < item.size) {
            if (this.paused) { item.state = "queued"; return; }
            const chunk = file.slice(item.received, item.received + cfg.chunk);
            state = await this.api("/api/uploads/" + item.serverId + "?offset=" + item.received, { method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: chunk });
            item.received = state.received;
            await idb.putMeta({ ...item });
          }
          await this.api("/api/uploads/" + item.serverId + "/complete", { method: "POST" });
          item.state = "done";
          item.error = "";
          await idb.putMeta({ ...item });
          await idb.dropFile(item.id); // free the browser storage right away
        } catch (e) {
          if (e.status && e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) {
            item.state = "failed"; // the server refused this file: do not retry
            item.error = e.status === 400 ? vtT("Not a photo or ZIP file") + " – " + e.message.slice(0, 120) : e.message.slice(0, 160);
            await idb.putMeta({ ...item });
            await idb.dropFile(item.id);
          } else {
            item.state = "queued"; // network or server trouble: try again later
            item.error = vtT("Connection lost – retrying…");
            item.backoff = true;
            setTimeout(() => { item.backoff = false; this.pump(); }, 5000);
          }
        }
      },
    };
  };
})();

// Installable app: register the service worker and offer an "Install app" button where the
// browser supports it (Chrome, Edge; Safari: "Add to Home Screen" in the share menu).
if ("serviceWorker" in navigator && (location.protocol === "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname))) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
(function () {
  let deferred = null;
  window.addEventListener("beforeinstallprompt", (ev) => {
    ev.preventDefault();
    deferred = ev;
    document.querySelectorAll("[data-install]").forEach((b) => (b.hidden = false));
  });
  window.addEventListener("appinstalled", () => {
    deferred = null;
    document.querySelectorAll("[data-install]").forEach((b) => (b.hidden = true));
  });
  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("[data-install]");
    if (!btn || !deferred) return;
    deferred.prompt();
    await deferred.userChoice;
    deferred = null;
    btn.hidden = true;
  });
})();
