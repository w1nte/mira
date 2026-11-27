# Properties

1. **Closed vocabularies.** Almost all fields are enums in the JSON schema. Ollama's structured
   output enforces them during decoding.
2. **Deterministic decoding.** `temperature = 0`, a fixed `seed`, a fixed prompt, and the same
   resize for every image.
3. **Normalisation of free-text lists** (objects, tags, …). Values are lower-cased, articles are
   dropped, words are singularised, then [`synonyms.yaml`](../src/vision_tagger/resources/synonyms.yaml)
   and the merges you make in the UI are applied.
4. **Versioning.** The model, prompt version and raw answer are stored. Changing the model or
   the vocabulary marks results as outdated, and the dashboard can re-queue them.

All properties end up in one key/value table (`image_properties`) with `source`
(`vlm`, `inferred`, `derived`, `exif`) and `confidence`. That table is the basis for filtering
and for clustering images by their properties later (one-hot vectors per image).

To customise the fields, copy `vocabulary.yaml` and set `vlm.vocabulary_file` in `config.toml`.
