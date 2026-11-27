# Places and map

Phones and many cameras store where a photo was taken (EXIF GPS). Mira keeps that
position in `images.lat` / `images.lon` and groups the photos into **places**:

- Photos closer than `geo.cluster_km` (750 m) to each other form a place (DBSCAN with
  haversine distance). Groups wider than `geo.max_place_km` (5 km) are split.
- **One town, one place.** Groups with the same automatic name within `geo.merge_km` (25 km) of
  each other are joined, so a town and its outskirts never show up twice. Same-named towns
  farther apart stay separate, and if their names would still be identical, they are told apart
  by the region or the nearest larger town ("Neustadt (Coburg), Germany" and "Neustadt
  (Mannheim), Germany"; "Springfield, IL" and "Springfield, MO" differ already).
- Every photo gets the place as `images.place_id` and as the properties **place**, **city** and
  **country**. They work everywhere other properties work: gallery facets and filters, the
  properties page, XMP export and the search index. `/images?place=12` shows one place
  (`place=12&place=14` either of two), `/images?bbox=minlon,minlat,maxlon,maxlat` an area of
  the map.
- Places are found at the end of every scan, with **Find places** on the map page, or with
  `vision-tagger places` (`--refresh` groups everything again and renames every place you did
  not name).
- **Your names are kept.** Rename a place in its popup on the map: the name, the place and its
  photos survive regrouping (like locked face clusters), and new photos nearby join it. An empty
  name returns to the automatic one. Values you merge or rename on the properties page stay
  merged too.

**Offline names (default).** A place is named after the town it lies in: the nearest one, where
big towns reach farther, so a photo in a suburb gets the city's name and a photo in a village
the village's name. The towns come from [GeoNames](https://www.geonames.org/) ("cities with a
population of at least 1000", 130,000 towns, CC BY 4.0). The first time places are named,
Mira downloads a copy of that data (3 MB) from the npm registry (the
[`all-the-cities`](https://www.npmjs.com/package/all-the-cities) package, version 3.1.0, MIT),
checks its size and SHA-256 and converts it into `data/models/geo`. GeoNames' own server
publishes a new dump every day, which can't be checked against a fixed checksum. To use an
official dump instead, download `cities500.zip` (or `cities1000.zip`, ...) from
<https://download.geonames.org/export/dump/> and set `geo.cities_file`; with
`geo.download_gazetteer = false` nothing is downloaded at all (places are then named by their
coordinates, e.g. `52.516° N, 13.378° E`).

**OpenStreetMap names (optional).** With `geo.nominatim = true`, places are named from
OpenStreetMap through [Nominatim](https://nominatim.org/) reverse geocoding, down to the district
("Prenzlauer Berg, Berlin"), in `geo.language`. This **sends the position of every new place to
the Nominatim server** (rounded to about 100 m), so it is off by default. Mira follows
the [usage policy](https://operations.osmfoundation.org/policies/nominatim/) of the public
server: at most one request per second, an identifying User-Agent without personal data, every
answer cached in the database (`geocode_cache`) and never asked again, and no further requests
in a run after the server refuses one. A self-hosted server can be set with `geo.nominatim_url`.

**Map tiles.** The map background comes from `geo.tile_url`, by default the OpenStreetMap tile
server (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`, with its required attribution). The
heatmap data stays on your machine, but **every tile request tells the tile server which area
you are looking at.** Point `geo.tile_url` (and `geo.tile_attribution`) at your own tile
server, or set it to `""`: the map then shows a coarse offline world outline (Natural Earth,
public domain) instead. The same outline appears when the tiles can't be loaded.

**Licenses.** Map data and OpenStreetMap names: © OpenStreetMap contributors, available under
the [Open Database License](https://www.openstreetmap.org/copyright). Offline place names:
GeoNames, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). World outline: Natural
Earth, public domain. Vendored libraries (in `web/static/vendor`, with their license files):
[Leaflet](https://leafletjs.com/) 1.9.4 and [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat)
0.2.0 (both BSD 2-Clause).
