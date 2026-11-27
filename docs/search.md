# Search

## How search works

1. **Interpretation.** The request goes to the LLM (`search.model`, by default the VLM model)
   together with the vocabulary, the most frequent free-text values, the person and pet names
   and the date range. The JSON answer is checked against the database: unknown persons or values
   become free-text keywords instead of failing. Without an LLM (*basic mode*), a dictionary
   does this; the visual search still understands the whole request.
2. **Candidates.** Persons, pets, required and excluded properties, alternatives and dates filter
   strictly. Photos the vision model hasn't described yet can't have the required properties;
   they stay candidates when everything else matches, and the search note says how many. If
   nothing is left, required properties are relaxed into preferences. Without a strict part
   every photo is a candidate, and photos that match nothing about the request are left out.
3. **Ranking** combines four signals (`search.weights`):
   - **visual** (0.45): the similarity of the request to the photo itself (see
     [Visual search](#visual-search)), without the names, places, dates and excluded things
     the filters take care of; in LLM mode the English rewrite;
   - **semantic** (0.25): the similarity of an English rewrite of the request to each image's
     document embedding (caption, visible text, properties, folder, date);
   - **properties** (0.20): matches with the preferred properties, weighted by the LLM and by
     confidence;
   - **keywords** (0.10): hits in captions, visible text, values and paths.

   A photo is scored by the signals it has: one without a vector or a description yet is ranked
   by the others. The visual similarity is calibrated on the measurements above (0 for photos
   that don't show the request, 0.75 for a clear match), so a request nothing matches can't
   push unrelated photos up. Person and pet names are not part of the embedded documents, so
   renaming people or pets never requires re-indexing.
4. **Deep search.** The LLM scores the top results from short descriptions (including person
   names) and gives a one-line reason in the language of your request. Photos without a
   description yet keep their place.

The search index is kept up to date automatically: images whose VLM results or property values
change are queued for Stage 3 again. Changed photos are embedded again by the visual stage.

### Alternatives ("or")

Alternatives inside a condition are OR-ed, conditions are AND-ed, and exclusions always apply:

| Request | Understood as |
|---|---|
| Anna or Ben at the beach | (Anna or Ben) and beach |
| Anna oder Ben am Strand | the same, in German |
| dogs or cats | animals: dog or cat |
| beach or mountains in summer 2021 | (beach or mountains) and summer, 2021 |
| photos from 2019 or 2021 | taken in 2019 or in 2021 |
| Paris or Rome, not at work | city Paris or Rome, without work photos |
| without Anna or Ben | neither Anna nor Ben |

"Anna and Ben" still means photos with both. The LLM understands "or" in any language; the
keyword fallback knows *or, oder, ou, o, of, lub, или, 或, または* and *か* (between two
recognised names, values, places or dates, so names that contain such a word stay whole). The
search page shows each group as one chip ("Anna *or* Ben"), and **Open in filter builder** keeps
the groups.

In the gallery, a facet's **or** switch lists the values you could add as alternatives (counted
without the condition on that property); Ctrl+click (⌘+click) on a value does the same. The
filter chips show alternatives in one chip, each with its own remove button. In the URL, `|`
separates the alternatives of one `t` condition; a photo has one date and one place, so repeated
`d` and `place` parameters are alternatives as well:

```
/images?t=location_type:beach|location_type:mountain&t=season:summer   (beach or mountains) and summer
/images?p=3&p=4&pm=any                                                   Anna or Ben (pm=all: both)
/images?d=2019&d=2021-06..2021-08                                        2019 or summer 2021
/images?place=12&place=14                                                either of two places
```

Older URLs and saved collections keep their meaning: repeated `t` parameters are all required,
and `tm=any` (the former "any property" switch) makes them one "or" condition.

## Visual search

The **visual** stage (0, it runs first) turns every photo into a vector with the image encoder
of [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) (Google, Apache-2.0). Its
text encoder turns a search request into a vector of the same space; photos whose vectors point
the same way show what the request describes. This needs neither the vision model nor Ollama,
works for photos the vision model hasn't described yet, and understands requests in many
languages (English and German were measured; the other UI languages are among its training
languages).

- **Search.** Requests are scored against the photos' vectors (see
  [How search works](#how-search-works)). The search page shows how many photos have a vector
  (*visual · 85 % indexed*); the score of each part is in the tooltip of a result.
- **Similar photos.** The button on a photo's page (or `vision-tagger similar <id>`) lists the
  photos that look most like it, without the photo itself and without copies (pHash within 4
  of 64 bits) of a photo already listed.
- **Speed.** About 0.11 s per photo on a 4-core CPU, a few milliseconds on a GPU
  (`visual.providers`, by default `faces.providers`). The stage reads the gallery thumbnail
  when it is large enough (as good as the original in the measurements below), else the
  original at a reduced size. The search keeps the vectors in memory (154 MB for 50,000 photos,
  loaded in 0.4 s): comparing a request with all of them takes about 10 ms, a whole search over
  50,000 photos about 0.1 s. A new request takes about 50 ms on the CPU (then it is cached); the
  text encoder always runs on the CPU, so the GPU stays free for the pipeline. It needs about
  0.8 GB of memory while it is loaded (the first search loads it in about 2 s) and is unloaded
  after ten minutes without a new request.
- **Privacy and downloads.** The model files (970 MB: the image encoder in float32, the text
  encoder in float16, the tokenizer) are downloaded once from Hugging Face
  (`onnx-community/siglip2-base-patch16-224-ONNX`, pinned to one commit and checked against
  their SHA-256 checksums) into `<faces.model_root>/models/visual` (default
  `~/.insightface/models/visual`; or `visual.model_dir`). The Pipeline page shows the download
  progress; pause and stop work meanwhile. Without internet the stage is skipped (the other
  stages still run), and the message names the URL and the place for the file. No photo and no
  request leaves the computer.
- **Settings.** `visual.model = "siglip2-base-fast"` (SigLIP 2 B/32 at 256 px) is three times
  as fast on the CPU and finds a little less; changing the model embeds the photos again.
  `visual.batch_size` (16) photos go through the model at once.

**How the model was chosen.** Text-to-photo search on 1158 labelled photos: the 158 photos of the
[DreamBooth dataset](https://github.com/google/dreambooth) (30 objects and animals, Unsplash
license) plus 1000 random photos of the COCO 2014 test split
(`nlphuji/mscoco_2014_5k_test_image_text_retrieval`). Requests: 38 hand-written ones in English
and German ("a dog on a sofa" / "ein Hund auf dem Sofa", "something to drink from", "toys",
"a train at a station", …; the relevant photos labelled by hand or found through the COCO
captions), 40 COCO captions with a German translation (one right photo each), and the first
caption of each of the 1000 COCO photos. R@5 is the share of the relevant photos among the first
five (of at most five), R@1 the share of requests whose first result is right, MRR the mean of
1 / rank of the first right result. Speed: the image encoder on a 4-core Xeon CPU (onnxruntime,
batches of 16).

| Model (image / text encoder) | Download | CPU per photo | 38 requests R@5 / MRR, en · de | 40 captions R@5, en · de | 1000 COCO captions R@1 / R@5 |
|---|---|---|---|---|---|
| **SigLIP 2 B/16-224, fp32 / fp16** (default) | 971 MB | 106 ms | 0.94 / 0.98 · 0.92 / 0.97 | 0.88 · 0.88 | 0.68 / 0.90 |
| SigLIP 2 B/16-224, fp32 / int8 | 689 MB | 106 ms | 0.94 / 0.97 · 0.82 / 0.88 | 0.88 · 0.75 | 0.66 / 0.89 |
| SigLIP 2 B/16-224, int8 / int8 | 412 MB | 92 ms | 0.89 / 0.96 · 0.74 / 0.85 | 0.58 · 0.63 | 0.34 / 0.59 |
| **SigLIP 2 B/32-256, fp32 / fp16** (`siglip2-base-fast`) | 977 MB | 33 ms | 0.95 / 0.98 · 0.91 / 0.96 | 0.90 · 0.88 | 0.65 / 0.88 |
| SigLIP 2 B/32-256, int8 / int8 | 413 MB | 32 ms | 0.81 / 0.97 · 0.72 / 0.86 | 0.48 · 0.48 | 0.25 / 0.49 |
| nomic-embed-vision / -text v1.5, fp32 | 922 MB | 107 ms | 0.94 / 1.00 · 0.66 / 0.77 | 0.85 · 0.35 | 0.47 / 0.75 |
| nomic-embed-vision / -text v1.5, int8 | 234 MB | 46 ms | 0.90 / 0.97 · 0.56 / 0.71 | 0.80 · 0.38 | 0.40 / 0.71 |
| CLIP ViT-B/32 / multilingual text (mCLIP), fp32 | 900 MB | 23 ms | 0.85 / 0.92 · 0.67 / 0.81 | 0.78 · 0.73 | 0.44 / 0.76 |
| CLIP ViT-B/32 / multilingual text (mCLIP), int8 | 225 MB | 22 ms | 0.86 / 0.94 · 0.67 / 0.78 | 0.70 · 0.63 | 0.44 / 0.76 |

SigLIP 2 finds the most in both languages. Its int8 files are much smaller but much worse (the
int8 image encoder's vectors differ from the float32 ones by a cosine of 0.2), while the float16
text encoder gives the same vectors as float32 (cosine 0.99999) at half the size; the float16
image encoder is as exact but twice as slow on the CPU. nomic-embed-vision (Apache-2.0) is
aligned with `nomic-embed-text` v1.5 (also without the layer norm of its model card), but it
understands only English. CLIP with the multilingual text encoder is the fastest and finds the
least. All candidates run on onnxruntime with the `tokenizers` library; no PyTorch.
