# PyInstaller spec of the desktop app, shared by Windows, macOS and Linux: Mira (the window) and
# vision-tagger-cli (the command line) in one folder, dist/vision-tagger; on macOS also
# dist/Mira.app with both inside (Contents/MacOS).
#
#   uv sync --frozen --no-dev --no-editable --extra cpu --extra desktop --group packaging
#   uv run --no-sync pyinstaller packaging/vision-tagger.spec --noconfirm --clean
#   dist/vision-tagger/vision-tagger-cli self-test    # macOS: dist/Mira.app/Contents/MacOS/...
#
# Then windows/installer.iss, macos/build_pkg.sh or linux/build_tarball.sh pack it. The version
# comes from the installed package (on release builds the tag, see version.py).
# ruff: noqa
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

HERE = Path(SPECPATH)
ROOT = HERE.parent
VERSION = dist_version("vision-tagger")
BUNDLE_ID = "io.github.w1nte.mira"  # the macOS package identifier is the same (macos/)

# icons, made from the app icon: .ico for Windows (also used by installer.iss), .icns for macOS
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
APP_ICON = Image.open(ROOT / "src/vision_tagger/web/static/icons/icon-512.png")
ICO = BUILD / "vision-tagger.ico"
APP_ICON.save(ICO, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
ICNS = BUILD / "Mira.icns"
APP_ICON.save(ICNS)

datas = []
datas += collect_data_files("vision_tagger")  # templates, static files, vocabularies, catalogs
# Alembic reads the migration scripts from disk, so they ship as files too
datas += collect_data_files("vision_tagger.migrations", include_py_files=True)
datas += copy_metadata("vision-tagger")  # __version__
# the update check tells the cpu and gpu builds apart by the installed ONNX Runtime
for onnx_dist in ("onnxruntime-gpu", "onnxruntime"):
    try:
        datas += copy_metadata(onnx_dist)
    except PackageNotFoundError:
        pass
datas += collect_data_files("insightface", includes=["**/*.pkl", "**/*.npy", "**/*.txt"])

hiddenimports = []
hiddenimports += collect_submodules("vision_tagger")
hiddenimports += collect_submodules("uvicorn")  # protocols and loops are imported by name
# visual search: the tokenizer of the text encoder (a compiled module, imported when needed)
hiddenimports += collect_submodules("tokenizers")

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


icon = str(ICO) if sys.platform == "win32" else None
app_exe = EXE(
    pyz,
    scripts_for("entry_app"),
    exclude_binaries=True,
    name="Mira",
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
# the app last: COLLECT (and BUNDLE) take the console setting from the last EXE
coll = COLLECT(cli_exe, app_exe, a.binaries, a.datas, name="vision-tagger", upx=False)

if sys.platform == "darwin":
    # Apple wants plain numbers here (1.2.3rc1 -> 1.2.3); the package metadata keeps the rest
    release = re.match(r"\d+(\.\d+){0,2}", VERSION).group()
    usage = "Mira indexes the photos in the folders you choose."
    app = BUNDLE(
        coll,
        name="Mira.app",
        icon=str(ICNS),
        bundle_identifier=BUNDLE_ID,
        version=release,
        info_plist={
            "CFBundleName": "Mira",
            "CFBundleDisplayName": "Mira",
            "CFBundleShortVersionString": release,
            "CFBundleVersion": release,
            "LSBackgroundOnly": False,  # a window app, even though the CLI is a console EXE
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "15.0",  # the oldest macOS the bundled wheels support
            "LSApplicationCategoryType": "public.app-category.photography",
            "NSDesktopFolderUsageDescription": usage,
            "NSDocumentsFolderUsageDescription": usage,
            "NSDownloadsFolderUsageDescription": usage,
            "NSRemovableVolumesUsageDescription": usage,
            "NSNetworkVolumesUsageDescription": usage,
        },
    )
    # macOS starts the bundle's CFBundleExecutable: PyInstaller takes the first EXECUTABLE of
    # the (alphabetically sorted) COLLECT, which must be the app, not the command line
    assert Path(app.exename).name == "Mira", app.exename
