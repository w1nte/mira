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

// "Show NSFW photos without asking" (web/nsfw.py): a cookie of this browser, read by the
// server, which then leaves the black covers out of the pages.
document.addEventListener("change", function (ev) {
  const box = ev.target.closest && ev.target.closest("input[data-nsfw-show]");
  if (!box) return;
  document.cookie = box.checked ? "vt_nsfw=show; path=/; max-age=31536000; SameSite=Lax" : "vt_nsfw=; path=/; max-age=0; SameSite=Lax";
  window.vtLoading.start();
  window.location.reload();
});

// Gallery selection: while photos are ticked (or after "Select"), a click on a photo ticks it
// instead of opening it (or the lightbox).
document.addEventListener("click", function (ev) {
  if (ev.button !== 0 || !ev.target.closest) return;
  const tile = ev.target.closest(".tile");
  if (!tile || ev.target.closest("input")) return;
  const box = tile.querySelector("input.pick");
  if (!box || !window.vt.selecting(tile)) return;
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
  // true while the selection of el's grid is on ("Select", or a photo is ticked)
  selecting(el) {
    const scope = el.closest("[data-select-scope]");
    return !!scope && (scope.classList.contains("selecting") || !!scope.querySelector("input.pick:checked"));
  },
};

// Lightbox (partials/lightbox.html): a plain click on a photo of a grid marked [data-lightbox]
// shows it large over the page instead of opening its page. Ctrl/⌘/Shift-click and the middle
// button still open the photo page (new tab or window), and so does a click during a selection
// (it ticks the photo) or on a covered NSFW tile (data-covered: the page has no URL of its
// picture). The tile links carry what the overlay shows: data-id, data-preview-url, data-title,
// data-date, data-people (JSON), data-w / data-h (pixels) and data-missing; never inline code.
// - History: opening adds one entry, so the browser's Back button closes the lightbox; closing
//   it with ×, Esc or a click beside the photo goes back over that entry, so Back afterwards
//   leaves the page at once. Previous / next replace the entry (it names the photo shown), and
//   "Open details" keeps it: Back from the photo page returns to the lightbox. Scroll
//   restoration is off meanwhile: the page stays at the photo last shown.
// - Previous / next: ← / →, the arrows, a swipe. At the last tile the gallery's next page is
//   loaded (its [data-lightbox-more] element); the neighbours' previews are fetched ahead.
// - A covered tile in between shows the black cover and "Open details"; nothing is fetched.
(function () {
  const dialog = document.querySelector("[data-lightbox-dialog]");
  if (!dialog || typeof dialog.showModal !== "function") return;
  const root = document.documentElement;
  const part = (name) => dialog.querySelector("[data-lb-" + name + "]");
  const stage = part("stage");
  const frame = part("frame");
  const photo = part("photo");
  const placeholder = part("placeholder");
  const cover = part("cover");
  const spinner = part("spinner");
  const missingNote = part("missing");
  const original = part("original");
  const prevBtn = part("prev");
  const nextBtn = part("next");
  const PREVIEW = parseInt(dialog.dataset.previewSize, 10) || 1600;
  const lang = root.lang || undefined;

  let scope = null; // the [data-lightbox] grid of the photo shown
  let current = null; // the tile link of the photo shown
  let seq = 0; // counts the photos shown: load events of earlier ones are ignored
  let loadingPhoto = false;
  let loadingMore = null; // Promise while the next page of tiles loads
  let waitingMore = false; // "next" at the last tile waits for it
  let backPending = false; // closed: waiting for history.back() to arrive
  let queued = null; // a tile clicked meanwhile
  let preloads = [];
  let swipe = null;
  let swipedAt = 0;

  const tiles = () => (scope ? Array.from(scope.querySelectorAll("a[data-id]")) : []);
  const moreLoader = () => scope && scope.querySelector("[data-lightbox-more]");
  const isCovered = (a) => a.hasAttribute("data-covered") || !a.dataset.previewUrl;
  const shownId = () => (history.state && history.state.vtLightbox) || null;

  function busy() {
    spinner.hidden = !(loadingPhoto || waitingMore);
  }

  // keep the tile shown in view behind the overlay (closing returns to it)
  function reveal(a) {
    const r = a.getBoundingClientRect();
    const top = parseFloat(getComputedStyle(root).getPropertyValue("--topbar-h")) || 0;
    if (r.top < top || r.bottom > window.innerHeight) a.scrollIntoView({ block: "center" });
  }

  function names(a) {
    try {
      const list = JSON.parse(a.dataset.people || "[]");
      return Array.isArray(list) ? list.map(String) : [];
    } catch (e) {
      return [];
    }
  }
  function listText(list) {
    if (!list.length) return "";
    try {
      return new Intl.ListFormat(lang, { type: "conjunction" }).format(list);
    } catch (e) {
      return list.join(", ");
    }
  }
  function dateText(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const opts = /T00:00$/.test(iso) ? { dateStyle: "long" } : { dateStyle: "long", timeStyle: "short" };
    try {
      return new Intl.DateTimeFormat(lang, opts).format(d);
    } catch (e) {
      return iso;
    }
  }

  // The frame gets the size the preview will be shown at (fitted, never enlarged), so the
  // thumbnail placeholder and the preview cover the same area and nothing jumps.
  function layout() {
    if (!current) return;
    const cs = getComputedStyle(stage);
    const availW = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const availH = stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const w0 = +current.dataset.w;
    const h0 = +current.dataset.h;
    let w = 640;
    let h = 480;
    if (isCovered(current)) {
      // the cover: 4:3 like the tiles' black placeholder
    } else if (frame.classList.contains("ready") && photo.naturalWidth) {
      w = photo.naturalWidth;
      h = photo.naturalHeight;
    } else if (w0 > 0 && h0 > 0) {
      const s = Math.min(1, PREVIEW / Math.max(w0, h0));
      w = w0 * s;
      h = h0 * s;
    } else if (placeholder.naturalWidth) {
      w = placeholder.naturalWidth;
      h = placeholder.naturalHeight;
    }
    const scale = Math.max(0, Math.min(1, availW / w, availH / h));
    frame.style.width = Math.max(1, Math.round(w * scale)) + "px";
    frame.style.height = Math.max(1, Math.round(h * scale)) + "px";
  }

  function updateNav() {
    const all = tiles();
    const i = all.indexOf(current);
    prevBtn.setAttribute("aria-disabled", i <= 0 ? "true" : "false");
    nextBtn.setAttribute("aria-disabled", i < 0 || (i >= all.length - 1 && !moreLoader()) ? "true" : "false");
  }

  // the previews of the neighbours, once the photo shown is there (covered ones: never)
  function preloadNeighbours() {
    const all = tiles();
    const i = all.indexOf(current);
    preloads = [all[i + 1], all[i - 1]]
      .filter((a) => a && !isCovered(a))
      .map((a) => {
        const im = new Image();
        im.decoding = "async";
        im.src = a.dataset.previewUrl;
        return im;
      });
  }

  function show(a) {
    current = a;
    const token = ++seq;
    const covered = isCovered(a);
    const missing = a.hasAttribute("data-missing");
    part("title").textContent = a.dataset.title || "";
    part("date").textContent = dateText(a.dataset.date);
    part("people").textContent = covered ? "" : listText(names(a));
    const page = a.getAttribute("href");
    dialog.querySelectorAll("[data-lb-details]").forEach((link) => link.setAttribute("href", page));
    original.hidden = covered || missing;
    original.setAttribute("href", covered || missing ? page : "/media/image/" + encodeURIComponent(a.dataset.id));
    cover.hidden = !covered;
    missingNote.hidden = !missing;
    frame.classList.remove("ready");
    frame.classList.toggle("covered", covered);
    photo.onload = photo.onerror = placeholder.onload = null;
    photo.removeAttribute("src");
    placeholder.removeAttribute("src");
    photo.alt = covered ? "" : a.dataset.title || "";
    loadingPhoto = !covered;
    if (!covered) {
      const thumb = a.querySelector("img");
      const thumbUrl = thumb && (thumb.currentSrc || thumb.getAttribute("src"));
      placeholder.onload = () => token === seq && layout();
      if (thumbUrl) placeholder.src = thumbUrl;
      const done = (ok) => {
        if (token !== seq) return;
        loadingPhoto = false;
        frame.classList.toggle("ready", ok);
        busy();
        layout();
        preloadNeighbours();
      };
      photo.onload = () => done(true);
      photo.onerror = () => done(false);
      photo.src = a.dataset.previewUrl;
    } else {
      preloadNeighbours();
    }
    busy();
    layout();
    updateNav();
    reveal(a);
    const all = tiles();
    if (all.indexOf(a) >= all.length - 3) loadMore(); // fetch the next page ahead
  }

  // the gallery's next page (HTMX, the element that loads it when scrolled into view):
  // resolves to true once new tiles are there
  function loadMore() {
    if (loadingMore) return loadingMore;
    const more = moreLoader();
    if (!more) return Promise.resolve(false);
    const before = tiles().length;
    const grid = scope;
    loadingMore = new Promise((resolve) => {
      let timer = null;
      const observer = new MutationObserver(() => {
        if (tiles().length > before) finish(true);
        else if (!grid.contains(more)) finish(false);
      });
      function finish(grew) {
        observer.disconnect();
        clearTimeout(timer);
        loadingMore = null;
        if (dialog.open) updateNav();
        resolve(grew);
      }
      observer.observe(grid, { childList: true });
      timer = setTimeout(() => finish(tiles().length > before), 20000);
      if (!more.hasAttribute("data-hx-revealed") && window.htmx) {
        more.setAttribute("data-hx-revealed", "true"); // HTMX's own scroll check: once
        window.htmx.trigger(more, "revealed");
      }
    });
    return loadingMore;
  }

  function step(delta) {
    if (!current) return;
    const all = tiles();
    const i = all.indexOf(current);
    const j = i + delta;
    if (i < 0 || j < 0) return;
    if (j < all.length) {
      show(all[j]);
      history.replaceState({ vtLightbox: all[j].dataset.id }, "");
      return;
    }
    if (delta < 0 || !moreLoader()) return;
    const from = current;
    waitingMore = true;
    busy();
    loadMore().then((grew) => {
      waitingMore = false;
      busy();
      if (grew && dialog.open && current === from) step(delta);
    });
  }

  function open(a, push) {
    if (backPending) {
      queued = a; // the history entry of the last lightbox is still being left
      return;
    }
    scope = a.closest("[data-lightbox]");
    if (!dialog.open) {
      const gutter = window.innerWidth - root.clientWidth; // keep the layout when the page
      root.style.setProperty("--lb-gutter", Math.max(0, gutter) + "px"); // loses its scrollbar
      root.classList.add("lb-open");
      dialog.showModal();
    }
    show(a);
    dialog.focus();
    if (push) {
      history.scrollRestoration = "manual"; // copied to the new entry
      history.pushState({ vtLightbox: a.dataset.id }, "");
    }
  }

  function close() {
    if (dialog.open) dialog.close();
  }

  // every way of closing ends here (×, Esc, backdrop, swipe, Back, a close request)
  dialog.addEventListener("close", function () {
    const a = current;
    seq++;
    current = null;
    preloads = [];
    loadingPhoto = false;
    waitingMore = false;
    photo.onload = photo.onerror = placeholder.onload = null;
    photo.removeAttribute("src");
    placeholder.removeAttribute("src");
    root.classList.remove("lb-open");
    root.style.removeProperty("--lb-gutter");
    if (a && a.isConnected) a.focus({ preventScroll: true });
    if (shownId()) {
      backPending = true;
      history.back();
    } else {
      history.scrollRestoration = "auto";
    }
  });
  dialog.addEventListener("cancel", function (ev) {
    ev.preventDefault();
    close();
  });

  function reopen(id) {
    const a = Array.from(document.querySelectorAll("[data-lightbox] a[data-id]")).find((t) => t.dataset.id === String(id));
    if (a) {
      open(a, false);
    } else {
      history.replaceState(null, "");
      history.scrollRestoration = "auto";
    }
  }

  window.addEventListener("popstate", function () {
    if (backPending) {
      backPending = false;
      history.scrollRestoration = "auto";
      const a = queued;
      queued = null;
      if (a && a.isConnected) open(a, true);
      return;
    }
    const id = shownId();
    if (!id && dialog.open) close();
    else if (id && !dialog.open) reopen(id); // Forward after Back
  });
  // back from another page (bfcache keeps the lightbox as it was; a reload opens it again)
  window.addEventListener("pageshow", function () {
    const id = shownId();
    if (!id && dialog.open) close();
    else if (id && !dialog.open) reopen(id);
  });

  document.addEventListener("click", function (ev) {
    if (ev.defaultPrevented || ev.button !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey) return;
    const a = ev.target.closest && ev.target.closest("[data-lightbox] a[data-id]");
    if (!a || isCovered(a) || window.vt.selecting(a)) return;
    ev.preventDefault();
    open(a, true);
  });

  dialog.addEventListener("click", function (ev) {
    const t = ev.target;
    if (t.closest("[data-lb-close]")) close();
    else if (t.closest("[data-lb-prev]")) step(-1);
    else if (t.closest("[data-lb-next]")) step(1);
    else if ((t === dialog || t === stage) && Date.now() - swipedAt > 500) close(); // beside the photo
  });

  // While the lightbox is open every key is its own: no gallery shortcuts, no "/" search.
  document.addEventListener(
    "keydown",
    function (ev) {
      if (!dialog.open) return;
      ev.stopImmediatePropagation();
      if (ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
      const onControl = ev.target !== dialog && ev.target.closest && ev.target.closest("a, button");
      if (ev.key === "Escape") {
        ev.preventDefault();
        close();
      } else if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
        ev.preventDefault();
        step(ev.key === "ArrowLeft" ? -1 : 1);
      } else if (ev.key === "Enter" && !onControl) {
        ev.preventDefault();
        part("details").click();
      }
    },
    true
  );

  // swipe left / right: next / previous; down: close (touch and pen)
  stage.addEventListener("pointerdown", function (ev) {
    swipe = ev.pointerType !== "mouse" && ev.isPrimary ? { x: ev.clientX, y: ev.clientY } : null;
  });
  stage.addEventListener("pointercancel", function () {
    swipe = null;
  });
  stage.addEventListener("pointerup", function (ev) {
    if (!swipe || !ev.isPrimary) return;
    const dx = ev.clientX - swipe.x;
    const dy = ev.clientY - swipe.y;
    swipe = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
      swipedAt = Date.now();
      step(dx < 0 ? 1 : -1);
    } else if (dy > 90 && dy > 1.5 * Math.abs(dx)) {
      swipedAt = Date.now();
      close();
    }
  });

  window.addEventListener("resize", function () {
    if (dialog.open) layout();
  });
})();

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
