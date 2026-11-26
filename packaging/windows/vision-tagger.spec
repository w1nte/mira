# PyInstaller spec of the desktop app: vision-tagger.exe (window) and vision-tagger-cli.exe
# (command line) in one folder, which installer.iss packs into a setup program.
#
#   uv sync --locked --no-dev --no-editable --extra gpu --extra desktop --group packaging
#   uv run --no-sync pyinstaller packaging/windows/vision-tagger.spec --noconfirm --clean
#   dist\vision-tagger\vision-tagger-cli.exe self-test
#
# It works on Linux and macOS as well (without the Windows icon), which is handy for testing.
# ruff: noqa
import sys
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

HERE = Path(SPECPATH)
ROOT = HERE.parent.parent

# Windows icon, made from the app icon
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
ICON = BUILD / "vision-tagger.ico"
Image.open(ROOT / "src/vision_tagger/web/static/icons/icon-512.png").save(
    ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
)

datas = []
datas += collect_data_files("vision_tagger")  # templates, static files, vocabularies, catalogs
# Alembic reads the migration scripts from disk, so they ship as files too
datas += collect_data_files("vision_tagger.migrations", include_py_files=True)
datas += copy_metadata("vision-tagger")  # __version__
datas += collect_data_files("insightface", includes=["**/*.pkl", "**/*.npy", "**/*.txt"])

hiddenimports = []
hiddenimports += collect_submodules("vision_tagger")
hiddenimports += collect_submodules("uvicorn")  # protocols and loops are imported by name

a = Analysis(
    [str(HERE / "entry_app.py"), str(HERE / "entry_cli.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "pytest", "_pytest", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)


def scripts_for(entry):
    """The boot and runtime-hook scripts of the analysis plus one entry script."""
    entries = {"entry_app", "entry_cli"}
    return [s for s in a.scripts if s[0] not in entries or s[0] == entry]


icon = str(ICON) if sys.platform == "win32" else None
app_exe = EXE(
    pyz,
    scripts_for("entry_app"),
    exclude_binaries=True,
    name="vision-tagger",
    console=False,
    icon=icon,
    upx=False,
)
cli_exe = EXE(
    pyz,
    scripts_for("entry_cli"),
    exclude_binaries=True,
    name="vision-tagger-cli",
    console=True,
    icon=icon,
    upx=False,
)
coll = COLLECT(app_exe, cli_exe, a.binaries, a.datas, name="vision-tagger", upx=False)
