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
