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
  const NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
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
      if (!this.hasSel) return "All time · " + this.total(0, n - 1).toLocaleString() + " photos";
      const range = this.start === this.end ? this.label(this.start) : this.label(this.start) + " – " + this.label(this.end);
      return range + " · " + this.total(this.start, this.end).toLocaleString() + " photos";
    },
    get hint() {
      if (this.pending !== null) return "Click the last month (Esc cancels)";
      return this.hasSel ? "Drag the band or its edges" : "Click the first month, then the last";
    },
    label(i) { const m = months[i]; return NAMES[+m.slice(5, 7) - 1] + " " + m.slice(0, 4); },
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
      window.location.assign("/images" + (qs ? "?" + qs : ""));
    },
  };
};
