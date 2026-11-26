"""Visual search: find photos by what they show, in any language.

An image encoder turns every photo into a vector, a text encoder turns a search request into a
vector of the same space; photos whose vectors point the same way as the request's show what
it describes. The same vectors find similar photos.

* :mod:`.models`: the model files (SigLIP 2, Google, Apache-2.0): pinned downloads, verified by
  size and SHA-256, like the animal models,
* :mod:`.encoder`: preprocessing, the image encoder the ``visual`` pipeline stage runs and the
  text encoder the search runs for each new request.

Everything runs locally through onnxruntime; nothing leaves the computer.
"""
