"""Animals and pets: find animals in photos, group photos of the same animal, name them.

* :mod:`.detector`: YOLOX-s (COCO, Apache-2.0) finds the animals and their species,
* :mod:`.embedder`: DINOv2-small (Apache-2.0) describes each animal as a vector, so photos of
  the same animal end up close together,
* :mod:`.analyzer`: both models together (what the ``animals`` pipeline stage runs),
* :mod:`.grouping`: groups (pets) with DBSCAN per species, and the user's edits (names, merges,
  moves, removals), which re-grouping and re-detection never undo,
* :mod:`.sync`: the ``pet`` and ``animal`` properties of every photo.

Both models are downloaded once (pinned files, verified by SHA-256) and run locally through
onnxruntime, like the face models. Nothing leaves the computer.
"""

from __future__ import annotations

#: COCO class index (as YOLOX numbers them) -> species. Stored values stay English; the UI
#: translates them like other property values.
COCO_ANIMALS: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}
SPECIES: tuple[str, ...] = tuple(COCO_ANIMALS.values())

__all__ = ["COCO_ANIMALS", "SPECIES"]
