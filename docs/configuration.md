# Configuration

`vision-tagger init` writes `config.toml` (see
[`config.example.toml`](../src/vision_tagger/resources/config.example.toml)). Any value can also be
set with an environment variable, for example `VT_VLM__MODEL=qwen3-vl:8b` or
`VT_PIPELINE__BATCH_SIZE=32`.

Important knobs:

- `mode`: `local` (scan folders on this computer) or `remote` (uploads, with `remote.upload_dir`).
- `web.password` (better: `VT_WEB__PASSWORD`): login password, meant for remote mode.
- `pipeline.resume_on_start`: continue a pipeline that was paused when the app was closed.
- `faces.cluster_eps` / `cluster_min_samples`: how strict face clustering is.
- `faces.reference_threshold` / `auto_accept_threshold`: when reference matches are suggested
  or assigned.
- `animals.cluster_eps` / `assign_threshold` / `min_score`: how strict pet grouping and animal
  detection are (see [Animals and pets](animals.md)).
- `nsfw.threshold` (0.7): from which classifier score a photo is covered as NSFW (see
  [Not safe for work](nsfw.md)).
- `visual.model`, `visual.providers`, `visual.batch_size`: the visual search model and where it
  runs (see [Visual search](search.md#visual-search)).
- `search.weights`: how much the visual, semantic, property and keyword signals count.
- `vlm.model`, `vlm.concurrency`: set `concurrency` to match `OLLAMA_NUM_PARALLEL` for
  parallel requests.
- `vlm.auto_pull` (default `true`): a pipeline run downloads a missing Ollama model first. With
  `false` it stops and asks for the download (the buttons and `vision-tagger models pull` still
  work).
- `derived.work_path_keywords`: folder or file-name hints for work images.
- `geo.cluster_km`, `geo.nominatim`, `geo.tile_url`: places and the map (see
  [Places and map](places.md)).
- `pipeline.stages`: leave it out to run every stage, including stages that later versions
  add. List stages only to turn some off. The Pipeline page shows the stages your config turns
  off, so you can still run them once.
- `updates.check` (`"auto"`, `"on"`, `"off"`) and `updates.prereleases`: looking for new
  releases (see [Updates](installation.md#update-check)).

**Updates.** `config.toml` carries a `config_version`. When a new version of Mira changes what
a setting means or where it lives, it updates older files itself when it loads them, keeping
your values and comments. It writes a backup copy next to the file first
(`config.toml.bak-v1-<time>`). `vision-tagger migrate` shows what was changed. If the file can't
be written (read-only), Mira uses the updated settings anyway and logs a warning.

Version 2 removes the `pipeline.stages` list that `vision-tagger init` and the desktop app
copied from the example config. That copy kept every stage added later switched off: ages, the
search index and animals. A list you changed yourself is kept.
