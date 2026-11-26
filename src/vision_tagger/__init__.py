"""Mira (package: vision-tagger): index, recognise and tag a local photo collection."""

from importlib.metadata import PackageNotFoundError, version

#: the name users see (UI, window, installable app, installer); the Python package, the
#: command line and the data folder keep the technical name "vision-tagger"
APP_NAME = "Mira"

try:
    __version__ = version("vision-tagger")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
