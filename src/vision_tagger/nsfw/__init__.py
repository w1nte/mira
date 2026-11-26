"""Not safe for work (NSFW): which photos get a cover in the web UI.

A photo is NSFW when the property ``nsfw = yes`` is set (``image_properties``, like every other
property, so the gallery filters, "or" conditions, the properties page, the search and the XMP
export know it without extra code). There is one ``nsfw`` row per photo: the verdict, who made it
(``source``) and the classifier's score (``confidence``). It comes from

* :mod:`.classifier`: a small local image classifier (the ``nsfw`` pipeline stage, fast: the
  whole library is covered long before the vision model got through it),
* the vision model (the ``nsfw`` field of the vocabulary): nudity, sexual content, graphic
  violence or gore, for photos it analyses,
* the user (image page, gallery selection), which always wins: a photo marked "not NSFW" by hand
  stays uncovered when the classifier or the vision model look at it again.

See :mod:`.verdict` for how the three are combined. The model is downloaded once (pinned,
verified by SHA-256) and runs through onnxruntime, like the face and animal models. Nothing
leaves the computer.
"""

from __future__ import annotations

KEY = "nsfw"
YES, NO = "yes", "no"

__all__ = ["KEY", "NO", "YES"]
