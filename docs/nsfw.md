# Not safe for work (NSFW)

Photos with nudity, sexual content, graphic violence or gore get the property `nsfw = yes`.
Wherever such a photo appears in a list (gallery, search results, Home, collections, map popups,
the person, cluster, faces, pet, quiz and generations pages, property examples, the table) it is
a **black tile with "NSFW"**, and so are face and animal pictures cut from it. A click on the
tile opens the photo page, and that page shows the photo, with an *NSFW* note next to it: you
chose to open it (← / → then go to the neighbouring photos, which show as well). Back in the
list, the tiles are black again. The pictures behind the tiles are never loaded into a list, so
a slow connection or a screenshot never shows them. In the face and animal grids, a click selects
a tile as before; its *Open photo* button leads to the photo.

- **Where the verdict comes from.**
  - **The NSFW check** (stage `nsfw`, the first stage of the pipeline): the
    [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)
    classifier (a ViT-base/16 fine-tuned on 80,000 photos, two classes "normal" / "nsfw";
    Apache-2.0), in half precision as ONNX from `onnx-community/nsfw_image_detection-ONNX`.
    It scores a photo from 0 to 1; from `nsfw.threshold` (0.7) on, the photo is NSFW. About
    0.2 s per photo on a 4-core CPU, a few milliseconds on a GPU. It mainly recognises nudity
    and sexual content.
  - **The vision model** answers the question for every photo it analyses (field `nsfw` of the
    vocabulary: nudity, sexual content, graphic violence or gore). Photos analysed before this
    version show up as *outdated* on the Pipeline page; *Re-queue outdated* lets the vision
    model look at them again. That's your choice: the NSFW check covers the whole library
    anyway, and the vision model is slow.
  - A photo is NSFW when either of them says so.
- **Your choice wins.** On the photo page: **Mark as NSFW** / **Not NSFW**; in the gallery:
  **Select**, tick photos and use the bar at the bottom. A photo you marked keeps your verdict
  when the classifier or the vision model look at it again, and when the file changes.
  **Automatic** forgets your choice. The source (`manual`, `classifier` or `vlm`) and the
  classifier's score are shown on the photo page.
- **Filters.** The gallery's *NSFW* section has **Hide NSFW photos** (`xt=nsfw:yes`) and
  **Only NSFW photos** (`t=nsfw:yes`); both work like any property, also with "or" and in
  collections, and the search knows "nsfw" ("summer without nsfw").
- **Show NSFW photos without asking** (gallery and photo page) turns the covers off in this
  browser only (cookie `vt_nsfw`), so the library can stay covered on another device.
- **How the threshold was chosen.** On 32 clearly safe photos (28 objects and pets from the
  [DreamBooth dataset](https://github.com/google/dreambooth) and 4 photos of people that come
  with InsightFace) the median score was 0.0002 and all but one stayed below 0.01; the highest
  was 0.34 (pink sunglasses on white bed sheets: skin-like colours). 0.7 leaves twice that
  margin, while the model gives clear cases scores near 1. The 8-bit versions of the model
  scored that photo 0.65 and 0.80, so the half-precision model (172 MB, the same scores as the
  full one) is used. Lower `nsfw.threshold` for more covers, raise it for fewer; the results of
  the NSFW check then show up as outdated, and re-running them takes minutes.
- **Privacy and download.** The model (172 MB) is downloaded once into
  `<faces.model_root>/models/nsfw` (or `nsfw.model_dir`) from Hugging Face, pinned to one
  commit and checked against its SHA-256 checksum. Without internet the run stops with a message
  that names the URL and the place for the file. It runs locally on onnxruntime (CUDA or CPU,
  `faces.providers`); no photo leaves the computer.
