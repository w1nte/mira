// Map page: heatmap of the filtered photos, place markers with a popup (thumbnails, link to
// the photos, rename), "Photos in this area". Data comes from /api/geo/*; configuration from
// data-* attributes of #map. Place names are user data: they only ever go into textContent.
(function () {
  "use strict";
  const el = document.getElementById("map");
  if (!el || !window.L) return;

  let strings = {};
  try { strings = JSON.parse(document.getElementById("map-i18n").textContent); } catch (e) { strings = {}; }
  const t = (key, vars) =>
    (strings[key] || key).replace(/%\((\w+)\)s/g, (m, name) => (vars && name in vars ? vars[name] : m));
  const num = (n) => Number(n).toLocaleString(document.documentElement.lang || undefined);
  const photos = (n) => (n === 1 ? t("1 photo") : t("%(n)s photos", { n: num(n) }));
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const ds = el.dataset;
  const query = ds.query || "";
  const maxZoom = parseInt(ds.maxZoom || "19", 10) || 19;
  const status = document.getElementById("map-status");
  let sayTimer = null;
  function say(text, seconds) {
    if (!status) return;
    status.textContent = text;
    status.hidden = !text;
    clearTimeout(sayTimer);
    if (text && seconds) sayTimer = setTimeout(() => { status.hidden = true; }, seconds * 1000);
  }

  const map = L.map(el, { maxZoom: maxZoom, minZoom: 1, worldCopyJump: true, zoomSnap: 0.5 });
  map.attributionControl.setPrefix(false);
  // Fixed layer order, whatever loads first: tiles (200) < offline outline < heatmap (the
  // overlay pane, 400) < places. Each vector pane gets its own SVG renderer.
  map.createPane("vt-world").style.zIndex = 250;
  map.createPane("vt-places").style.zIndex = 450;
  const worldRenderer = L.svg({ pane: "vt-world" });
  const placeRenderer = L.svg({ pane: "vt-places" });

  // ---------------------------------------------------------------- background
  // Tiles from geo.tile_url; without tiles (switched off, or unreachable) a coarse offline
  // world outline (Natural Earth, public domain) keeps the map readable.
  let world = null;
  function showWorld() {
    if (world || !ds.world) return;
    world = true;
    el.classList.add("no-tiles");
    fetch(ds.world)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => {
        world = L.geoJSON(data, {
          interactive: false,
          pane: "vt-world",
          renderer: worldRenderer,
          style: (f) => ({ className: f.properties.k === "border" ? "world-border" : "world-land" }),
        }).addTo(map);
      })
      .catch(() => { world = null; });
  }
  if (ds.tiles) {
    const tiles = L.tileLayer(ds.tiles, {
      maxZoom: maxZoom,
      attribution: ds.attribution || "",
    }).addTo(map);
    let loaded = 0;
    let failed = 0;
    tiles.on("tileload", () => { loaded++; say(""); });
    tiles.on("tileerror", () => {
      failed++;
      if (!loaded && failed >= 4 && !world) {
        say(t("Map tiles could not be loaded – showing photos without a background map."), 12);
        showWorld();
      }
    });
  } else {
    showWorld();
  }

  // ---------------------------------------------------------------- initial view
  function hashView() {
    const m = /^#(\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)$/.exec(location.hash);
    if (!m) return null;
    const z = parseFloat(m[1]), lat = parseFloat(m[2]), lon = parseFloat(m[3]);
    if (!(Math.abs(lat) <= 90 && Math.abs(lon) <= 180)) return null;
    return { z: Math.min(Math.max(z, 1), maxZoom), lat: lat, lon: lon };
  }
  const photo = ds.photoId ? [parseFloat(ds.photoLat), parseFloat(ds.photoLon)] : null;
  const fromHash = hashView();
  let viewSet = false;
  if (fromHash) {
    map.setView([fromHash.lat, fromHash.lon], fromHash.z);
    viewSet = true;
  } else if (photo) {
    map.setView(photo, 16);
    viewSet = true;
  } else {
    map.setView([30, 10], 2);
  }
  if (photo) {
    const marker = L.circleMarker(photo, {
      radius: 7,
      className: "photo-marker",
      pane: "vt-places",
      renderer: placeRenderer,
    }).addTo(map);
    marker.on("click", () => { location.assign("/images/" + encodeURIComponent(ds.photoId)); });
  }
  map.on("moveend", () => {
    const c = map.getCenter();
    const z = Math.round(map.getZoom() * 2) / 2;
    history.replaceState(null, "", "#" + z + "/" + c.lat.toFixed(5) + "/" + c.lng.toFixed(5));
  });

  // ---------------------------------------------------------------- heatmap
  const getJSON = (url) =>
    fetch(url, { headers: { Accept: "application/json" } }).then((r) =>
      r.ok ? r.json() : Promise.reject(new Error(String(r.status)))
    );

  // The heatmap reads its canvas back for every redraw; say so when the context is made
  // (leaflet.heat does not), which keeps it fast and the console quiet.
  if (L.HeatLayer) {
    const initCanvas = L.HeatLayer.prototype._initCanvas;
    L.HeatLayer.prototype._initCanvas = function () {
      const create = L.DomUtil.create;
      L.DomUtil.create = function (tag) {
        const node = create.apply(this, arguments);
        if (tag === "canvas") node.getContext("2d", { willReadFrequently: true });
        return node;
      };
      try {
        return initCanvas.call(this);
      } finally {
        L.DomUtil.create = create;
      }
    };
  }

  getJSON(ds.points)
    .then((data) => {
      const top = Math.log1p(data.max || 1);
      // weights 0.25-1 on a log scale: one photo is visible, thousands do not drown the rest
      const pts = data.points.map((p) => [p[0], p[1], 0.25 + 0.75 * (Math.log1p(p[2]) / top)]);
      if (L.heatLayer) {
        // full intensity from zoom 5 on; below it nearby cells add up (cities glow red)
        L.heatLayer(pts, { radius: 22, blur: 18, maxZoom: 5, max: 1.0, minOpacity: 0.3 }).addTo(map);
      }
      if (!viewSet && data.bounds) {
        const b = L.latLngBounds(data.bounds);
        map.fitBounds(b.pad(0.15), { maxZoom: 13 });
      }
    })
    .catch(() => say(t("Could not load the map data.")));

  // ---------------------------------------------------------------- places
  const markers = new Map();
  const LABEL_ZOOM = 11;
  let labelsOn = null;

  function label(place) {
    return make("span", "", place.name);
  }
  function bindLabel(marker, place, permanent) {
    marker.unbindTooltip();
    marker.bindTooltip(label(place), {
      direction: "top",
      offset: [0, -marker.getRadius()],
      className: "place-label",
      permanent: permanent,
    });
  }
  function updateLabels() {
    const on = map.getZoom() >= LABEL_ZOOM;
    if (on === labelsOn) return;
    labelsOn = on;
    markers.forEach((entry) => bindLabel(entry.marker, entry.place, on));
  }

  function popup(place) {
    const box = make("div", "place-pop");
    const title = make("h3", "", place.name);
    const meta = make("p", "meta", photos(place.count) + (place.user_named ? " · " + t("Named by you") : ""));
    const thumbs = make("div", "place-thumbs");
    // reserve the space of the thumbnails, so the popup does not grow (and get cut) later
    for (let i = 0; i < Math.min(8, place.count); i++) thumbs.append(make("a"));
    const actions = make("div", "actions");
    const show = make("a", "btn primary small", t("Show photos"));
    const params = new URLSearchParams(query);
    params.set("place", String(place.id));
    show.href = "/images?" + params.toString();
    actions.append(show);

    const form = make("form");
    form.method = "post";
    form.action = "/places/" + encodeURIComponent(place.id) + "/rename";
    const lab = make("label", "", t("Name"));
    const input = make("input");
    input.type = "text";
    input.name = "name";
    input.maxLength = 200;
    input.value = place.user_named ? place.name : "";
    input.placeholder = place.user_named ? t("Empty: automatic name") : place.name;
    lab.append(input);
    const save = make("button", "small", t("Save"));
    save.type = "submit";
    form.append(lab, save);
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      save.disabled = true;
      fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((res) => {
          place.name = res.name;
          place.user_named = res.user_named;
          title.textContent = res.name;
          input.value = res.user_named ? res.name : "";
          input.placeholder = res.user_named ? t("Empty: automatic name") : res.name;
          meta.textContent = photos(place.count) + (res.user_named ? " · " + t("Named by you") : "");
          const entry = markers.get(place.id);
          if (entry) {
            bindLabel(entry.marker, place, !!labelsOn);
            entry.marker.getElement() && entry.marker.getElement().classList.toggle("named", !!res.user_named);
          }
          document.querySelectorAll('[data-place-row="' + place.id + '"] [data-place-name]').forEach((a) => {
            a.textContent = res.name;
          });
        })
        .catch(() => { form.submit(); }) // fall back to the plain form post
        .finally(() => { save.disabled = false; });
    });

    box.append(title, meta, thumbs, actions, form);
    getJSON(ds.placeApi + encodeURIComponent(place.id) + (query ? "?" + query : ""))
      .then((d) => {
        meta.textContent = photos(d.count) + (d.user_named ? " · " + t("Named by you") : "");
        show.href = d.gallery;
        const photoQuery = d.gallery.split("?")[1] || "";
        thumbs.replaceChildren();
        d.images.forEach((id) => {
          const a = make("a");
          a.href = "/images/" + encodeURIComponent(id) + (photoQuery ? "?" + photoQuery : "");
          const img = make("img");
          img.src = "/media/thumb/" + encodeURIComponent(id) + "?size=160";
          img.alt = "";
          a.append(img);
          thumbs.append(a);
        });
        const entry = markers.get(place.id);
        const pop = entry && entry.marker.getPopup();
        if (pop && pop.isOpen()) pop.update(); // re-layout and pan into view
      })
      .catch(() => {});
    return box;
  }

  function select(id) {
    document.querySelectorAll("[data-place-row]").forEach((row) => {
      row.classList.toggle("current", row.dataset.placeRow === String(id));
    });
  }

  getJSON(ds.places)
    .then((data) => {
      const top = Math.max(1, ...data.places.map((p) => p.count));
      data.places.forEach((place) => {
        const radius = 4 + 7 * Math.sqrt(place.count / top);
        const marker = L.circleMarker([place.lat, place.lon], {
          radius: radius,
          className: "place-marker" + (place.user_named ? " named" : ""),
          bubblingMouseEvents: false,
          pane: "vt-places",
          renderer: placeRenderer,
        }).addTo(map);
        const entry = { marker: marker, place: place, box: null };
        // built when opened; kept while open (popup.update() asks for the content again)
        marker.bindPopup(() => (entry.box = entry.box || popup(place)), {
          maxWidth: 320,
          autoPanPadding: [24, 24],
        });
        marker.on("popupopen", () => select(place.id));
        marker.on("popupclose", () => { entry.box = null; });
        markers.set(place.id, entry);
      });
      updateLabels();
      map.on("zoomend", updateLabels);
    })
    .catch(() => say(t("Could not load the map data.")));

  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest && ev.target.closest("[data-focus]");
    if (!btn) return;
    const entry = markers.get(parseInt(btn.dataset.focus, 10));
    if (!entry) return;
    map.flyTo(entry.marker.getLatLng(), Math.max(map.getZoom(), 13), { duration: 0.6 });
    map.once("moveend", () => entry.marker.openPopup());
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  // ---------------------------------------------------------------- photos in this area
  const areaBtn = document.querySelector("[data-map-area]");
  if (areaBtn) {
    areaBtn.hidden = false;
    areaBtn.addEventListener("click", () => {
      const b = map.getBounds();
      const clampLat = (lat) => Math.max(-90, Math.min(90, lat));
      const fmt = (v) => v.toFixed(5);
      let qs = ds.areaQuery || "";
      if (b.getEast() - b.getWest() < 360) {
        // shift into [-180, 180); an east edge beyond 180 means the area crosses the date
        // line (west > east, which the server understands)
        const shift = 360 * Math.floor((b.getWest() + 180) / 360);
        const west = b.getWest() - shift;
        let east = b.getEast() - shift;
        if (east > 180) east -= 360;
        const box = [west, clampLat(b.getSouth()), east, clampLat(b.getNorth())];
        qs += (qs ? "&" : "") + "bbox=" + box.map(fmt).join(",");
      }
      location.assign("/images" + (qs ? "?" + qs : ""));
    });
  }
})();
