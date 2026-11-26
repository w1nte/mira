"""The command line next to the app (vision-tagger-cli.exe).

Example: ``vision-tagger-cli index D:\\Photos``
"""

import multiprocessing

from vision_tagger.cli import app

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app(prog_name="vision-tagger-cli")
