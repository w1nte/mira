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
