"""vision-tagger: index, recognise and tag a local photo collection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vision-tagger")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
