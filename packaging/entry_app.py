"""The windowed desktop app (Mira.exe on Windows, Mira.app on macOS, Mira on Linux)."""

import sys

from vision_tagger.desktop import main

if __name__ == "__main__":
    sys.exit(main())
