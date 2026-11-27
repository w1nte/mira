# Features

- **Index (step 0).** Recursive scan that records path, file name, recording date (EXIF →
  file-name pattern → mtime), a dynamic JSON metadata column (camera, lens, exposure, GPS, …)
  and a perceptual hash (`imagehash.phash`).
  - A file whose pHash is already indexed **and** whose dHash nearly matches (re-encodes,
    resized copies) is **not** added again; it is recorded as a duplicate. Distinct images
    that only collide on the pHash (blank pages, same-template forms) stay separate.
  - Moved files are re-linked, and changed files are re-queued. If an original disappears,
    one of its duplicates takes its place.
  - System folders (`$RECYCLE.BIN`, `System Volume Information`) are skipped, and link loops
    can't trap the scan.
- **Resumable batches.** Every image has a state per stage (`pending / running / done / error /
  skipped`) in the database. Runs claim work in batches and commit after every image.
  - You can **pause, resume, stop** or kill a run at any time; interrupted images are picked up
    on the next start.
  - Errors are retried automatically up to `max_attempts`.
  - Results produced with an older model, prompt or vocabulary show up as *outdated* and can be
    re-queued.
- **Not safe for work.** A small local classifier checks every photo first (fast, long before
  the vision model got there), and the vision model also answers it. NSFW photos show as a black
  "NSFW" tile until you open them; mark photos yourself, and your choice always wins. See
  [Not safe for work (NSFW)](nsfw.md).
- **Stage 1: faces.** InsightFace (`buffalo_l`) detection plus 512-d embeddings. Clustering uses
  DBSCAN (cosine distance) and **keeps your manual work**:
  - Clusters you named, merged or split are locked and never dissolved by re-clustering.
  - Re-running the stage (e.g. after changing the detector settings) carries names, manual
    moves and removed faces over to the new detections.
  - A **reference DB** (`references/<Person Name>/*.jpg`) names clusters automatically or
    suggests a name.
- **Stage 1c: animals and pets.** YOLOX-s finds animals (dogs, cats, birds, horses, …),
  DINOv2-small tells them apart, and photos of the same animal are grouped like faces. Name a
  group ("Bello") and search for it: "Bello at the beach", "Anna with Bello". Names, merges and
  moves survive re-grouping. See [Animals and pets](animals.md).
- **Stage 2: properties.** Qwen2.5-VL (or any Ollama vision model) answers a fixed JSON schema
  built from [`vocabulary.yaml`](../src/vision_tagger/resources/vocabulary.yaml).
  - Closed vocabularies, temperature 0, a fixed seed and normalisation (singular forms,
    synonyms, your merges) keep the values stable enough to cluster on later.
  - **Assumed** properties come with a confidence: work related, work category, personal
    memory, likely purpose, sensitive info.
  - A derived `work_image = yes / maybe / no` combines these with file metadata to filter out
    work photos.
- **Stage 0: visual search.** SigLIP 2 turns every photo into a vector that lives in the same
  space as text, so the search finds photos by what they show ("a dog on a sofa", "ein Hund auf
  dem Sofa", "something to drink from") right after the import: without the vision model,
  without Ollama, in any language. The same vectors power **Similar photos** on every photo.
  See [Visual search](search.md#visual-search).
- **Natural-language search.** Ask in any language, for example "Anna and Ben at the beach in
  summer 2021, no work photos".
  - The local LLM turns the request into persons, dates and required, preferred or excluded
    properties, validated against your real vocabulary and person list.
  - Alternatives work too: "Anna or Ben at the beach", "dogs or cats", "Paris or Rome",
    "photos from 2019 or 2021" (see [Alternatives](search.md#alternatives-or)).
  - A text-embedding index (Stage 3) adds semantic similarity over everything extracted per
    image: caption, visible text, all properties, folder and date.
  - **Deep search** lets the LLM double-check the best ~40 results and explain why each fits.
  - Without an LLM, a keyword fallback still works: English and German words, the vocabulary
    in the language of the UI, names of persons, pets and places, dates, and "or" in every UI
    language.
- **Ages and doubtful dates.** Give a person a birth year, or let Mira estimate one
  from photo dates and face ages.
  - **Face ages from two models.** InsightFace's age model turns babies and children into young
    adults (a 12-year-old easily comes out as 40). Mira adds AgeNet, a small network
    from [face-api](https://github.com/vladmandic/face-api) (MIT license) trained on many
    children and old people, and combines both. AgeNet is used alone for children and teenagers,
    and both count equally from about 35 years on.
  - **Your birth years make it better.** Every birth year you set also shows how far off the
    face model is on *your* photos. Mira learns a correction per age range from those
    faces and applies it to everyone else.
  - **Robust estimate.** The birth year is a weighted median of "photo year − face age". It
    trusts EXIF dates more than file times, faces of children more than those of adults (their
    age is easier to see), and sharp faces more than tiny ones, and it drops outliers.
  - Every face shows the person's age in that photo, and each person gets a "through the years"
    timeline.
  - Photos where the date can't be right (camera clock reset to 2000, scans that only carry the
    file date, "taken" before birth) are flagged with a suggested year and a one-click fix. A
    date is only doubted when the faces disagree far beyond the model's typical error at that
    age; dates you set by hand are never doubted. Fixed dates survive re-scans.
  - **Date wrong or face age wrong?** The photo page offers both fixes: *Set year to …* when
    the date is wrong, or *Anna is 7 here* when the date is right and only the face looks older
    or younger. **Edit ages** under *People* sets any age by hand (*Automatic* goes back to the
    estimate). A manual age replaces the face estimate everywhere (date checks, birth-year
    estimate, timelines) and is kept when the faces are detected again.
  - **Sort by age.** With exactly one person (or one unnamed cluster) in the gallery filter,
    *Sort* offers *Age: youngest first* and *Age: oldest first*: the photos in the order of that
    person's age, which is the age you set, else the age from birth year and date, else the face
    estimate when the date is doubtful (`~7` on the tile). Photos with a wrong date still land
    where they belong.
- **Correcting dates** (Organize → Dates). Dates go wrong in batches: scans and copies carry the
  time they were scanned or copied, messenger apps strip the camera date, a camera clock was never
  set. Mira groups the photos that belong together, so a whole batch is fixed at once:
  - In one folder, photos with a camera date (or a date in the file name) at most 4 hours apart
    form one event. Consecutive shots or scans (`IMG_0104`, `IMG_0105`; `scan_007`, `scan_008`)
    and, once the *Visual search* stage has run, neighbouring photos showing the same scene join
    them. Copies of the same photo are grouped across folders.
  - Groups whose camera dates are more than two days apart are never joined: two Christmases in
    the same living room stay two events, and a photo with only a file time can't bridge them.
  - A photo whose date doesn't fit its group is marked, with a suggestion: the date of its nearest
    neighbour by file name that has a good date. Groups without any camera date are listed too.
  - **Default dates.** A camera whose clock was never set (or lost its battery) writes a default
    date that looks real but isn't. Mira treats these as untrustworthy, whatever their source:
    - around New Year 1970, 1980, 2000 or 2001, including the shift of a time zone
      (`2000-12-31 22:00` is `2001-01-01 00:00` two hours away) and two days of a reset clock
      running on;
    - New Year's midnight of any year at a round time (seconds zero, a quarter hour);
    - the same date, on the hour, in three or more different photos in more than one folder;
    - dates in the future.

    A default date never sets a group's date. It is marked and gets the date of the photos around
    it. Photos with a default date that belong to no group are collected under *Default dates*.
    A real photo that happens to match (New Year's Eve at exactly 23:00:00) is listed too; set its
    date by hand once and it is never doubted again.
  - **Set date** gives every ticked photo the same day (each keeps its time of day, unless you
    enter one). **Shift dates** moves them together: the earliest gets the date, the others keep
    their distance, for a camera clock that was set wrong.
  - Dates you set count as *set by hand*: re-indexing keeps them, ages trust them, and exports
    write them into the XMP data. Nothing about the groups is stored; they are worked out when you
    open the page.
- **"Who's that?" quiz** 🎲. Name unnamed clusters as a game, or test yourself on the people you
  already named.
  - After each answer you see how sure the face recognition was.
  - "That's not X!" pulls misfiled faces out of a cluster, and the quiz offers to move a face
    when the AI agrees with you.
- **Generations** ✨. Watch the people in your photos grow up.
  - **Grow-up timelapse**: one face per age (the sharpest, most frontal one), aligned on the
    eyes and cross-faded from the youngest to the oldest. Scrub through the years with a slider,
    or download it as an animated WebP to share.
  - Ages come from the birth year and the photo date. When a date can't be right, the face's
    estimated age is used instead (shown as "~7 years").
  - **Same age, side by side**: compare two people at matching ages ("Dad at 6 and his son at
    6"), with how similar the faces are.
  - **Looks most like…**: who in the family resembles a person most, compared at matching ages
    where possible.
- **Places and map.** Photos with a GPS position are grouped into places, each with a name to
  filter by ("Berlin, Germany"; with the optional OpenStreetMap lookup "Prenzlauer Berg,
  Berlin"). The map shows a heatmap of all geotagged photos, honouring the gallery filters, with
  the places on top. See [Places and map](places.md).
- **Web UI.**
  - Face cluster overview with inline naming, merge, split, move, auto-split and remove.
  - Persons, including merging persons.
  - Property explorer: combine values with AND/OR, merge/rename/delete values.
  - Gallery with a filter builder (persons AND/OR, properties AND/OR, NOT, date, folder,
    camera, …) and export (copy / hardlink / symlink) **with the metadata for other apps**
    (see [Using the results in other applications](export.md)).
  - Animals page: pet groups with inline naming, merge, move and remove.
  - Image detail with face and animal boxes.
  - Timeline / data overview and a metadata table.

## Notes and limitations

- Deduplication: a file is a duplicate if its pHash matches and its dHash is within
  `index.dhash_max_distance` bits. Every collapsed file stays listed under *duplicates*, so
  nothing is lost.
- Symlink export needs Developer Mode on Windows. Otherwise Mira falls back to
  copying.
- InsightFace's pretrained `buffalo_l` models are licensed for **non-commercial research
  purposes** only. The animal models (YOLOX-s, DINOv2-small) and the visual search model
  (SigLIP 2) are Apache-2.0.
- Face ages: the first faces run downloads AgeNet (430 KB, MIT license) from GitHub into
  `~/.insightface/models/agegender_faceapi`. The download is checked against a SHA-256 checksum
  and turned into an ONNX model locally. The **ages** stage re-estimates faces that were found
  before (no new face detection). It is added automatically whenever **faces** runs. Set
  `faces.age_model = "insightface"` to keep the old behaviour.
- Face ages are estimates. Expect about ±2 years for children and ±5–10 years for adults. Set
  the birth years of a few people for exact ages; Mira then also corrects the others.
