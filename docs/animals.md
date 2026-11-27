# Animals and pets

The **animals** stage (1c, after the faces) finds the animals in every photo, groups the photos
of the same animal, and lets you name them. A named pet works like a person: a filter
(`/images?pet=5`, chips and a facet in the gallery, AND/OR together with persons), a collection
on the home page, and the search: "Bello am Strand", "photos of Minka in winter", "Anna with
Bello".

- **Detection: [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, Apache-2.0),
  trained on COCO. It finds birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras and
  giraffes, each with a box, the species and a score. Detections below `animals.min_score`
  (0.5) or smaller than `animals.min_box_px` (32 px) are dropped; uncertain or small ones (below
  `group_min_score` 0.6 or `group_min_px` 64 px) are kept but not grouped automatically.
- **Telling animals apart: [DINOv2-small](https://github.com/facebookresearch/dinov2)** (Meta,
  Apache-2.0, as ONNX from `onnx-community/dinov2-small`). Each animal is cut out with a small
  margin, padded to a square (never stretched) and described by DINOv2's class token, averaged
  with the mirrored crop.
- **Grouping** works like face clustering: DBSCAN with the cosine distance `animals.cluster_eps`
  (0.3) and at least `cluster_min_samples` (2) photos; new photos join a pet when they are at
  least `animals.assign_threshold` (0.7) similar to its average. Animals are grouped per
  species, but the species the detector mixes up on pets (a chow chow detected as a bear, a
  poodle as a sheep) count as one (`animals.species_pools`: cat, dog, bear, sheep). A pet's
  species is what most of its photos show; you can correct it on its page.
- **Naming** happens on the **Animals** page (sidebar, *People*): type a name under a group.
  Typing the name of another group merges the two. Open a group to move photos to another pet
  (by name) or into a new group, to take them out ("Not this one") or to remove them (not an
  animal). Like locked face clusters, **named and edited groups, your moves and removals are
  kept** when the animals are grouped again (*Group all again* on the Pipeline page,
  `vision-tagger pets --full`) and when photos are analysed again. A named pet stays even
  without photos, so its animal finds its way back.
- **Properties.** Every photo gets `animal = <species>` and, for each named pet in it,
  `pet = <name>` (source `animal`), so the facets, filters, the properties page (group *Pets*),
  XMP keywords and the search work as for places. The vision model's own free-text **Animals**
  property stays as it is (it also names animals the detector doesn't know, like ducks or
  fish); the detector's property is shown as **Species** to tell them apart. The species is part
  of the search document; pet names are not (like person names, they filter strictly), so
  renaming a pet needs no re-indexing.
- **How the thresholds were chosen.** On 107 real photos: the 9 dogs and cats of the
  [DreamBooth dataset](https://github.com/google/dreambooth) (5–6 photos of each animal,
  Unsplash license) and, as other animals, 60 photos of 10 breeds from the Oxford-IIIT Pet test
  set (CC BY-SA 4.0). Every photo had an animal at a score of 0.5 or more, and 27 photos without
  animals had none; the species was wrong for 9 of the 107 photos (mostly fluffy dogs as bears
  or sheep, hence the pools). Photos of the same animal had a cosine similarity of 0.80 on
  average (10th percentile 0.70), other animals of the same species 0.14 (99th percentile 0.49).
  With the defaults the 9 animals form 9 groups (pairwise precision 1.00, recall 0.95: one photo
  stays ungrouped). Animals of one breed and colour are hard to tell apart: the grey cat was
  joined by Russian Blues, and different pugs or black cats form one group. Split them with
  *New group*.
- **Privacy and downloads.** Both models (125 MB) are downloaded once into
  `<faces.model_root>/models/animals` (default `~/.insightface/models/animals`; or
  `animals.model_dir`) from GitHub and Hugging Face, pinned to fixed files and checked against
  their SHA-256 checksums. Without internet, the error message names the URL and the place for
  the file. Everything else runs locally on onnxruntime (CUDA or CPU, `faces.providers`), about
  0.25 s per photo on a 4-core CPU. No photo leaves the computer.

The animals stage runs by default. A `config.toml` written by an earlier version is updated
on the next start (see [Configuration](configuration.md)); only a stage list you changed
yourself stays as it is, so add `"animals"` there if you want it.
