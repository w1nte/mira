"""Migrations of config.toml (``config_migrate``)."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
import tomlkit
from typer.testing import CliRunner

from vision_tagger import config as config_mod
from vision_tagger.config import DEFAULT_STAGES, load_settings
from vision_tagger.config_migrate import CONFIG_VERSION, STEPS, load_config

EXAMPLE = Path(config_mod.__file__).parent / "resources" / "config.example.toml"

# what `vision-tagger init` wrote with the first release (shortened; the comments matter)
FIRST_RELEASE = """\
# vision-tagger configuration
# Every value can also be overridden with environment variables

[paths]
image_roots = ['D:\\Photos']        # folders scanned recursively
data_dir = "data"                   # database, thumbnails and face crops

[pipeline]
batch_size = 8                      # images claimed per batch
max_attempts = 3                    # automatic retries per image and stage
stages = ["faces", "vlm"]

[vlm]
model = "qwen2.5vl:7b"              # my model
"""


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setattr(config_mod, "last_config_migration", None)
    monkeypatch.delenv("VT_CONFIG", raising=False)


def write(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + ".bak-*"))


def test_steps_are_numbered_without_gaps():
    assert sorted(STEPS) == list(range(2, CONFIG_VERSION + 1))


def test_the_example_config_is_current_and_pins_no_stages(tmp_path):
    doc = tomlkit.parse(EXAMPLE.read_text(encoding="utf-8"))
    assert doc["config_version"] == CONFIG_VERSION
    assert "stages" not in doc["pipeline"]  # the default (all stages) applies
    path = write(tmp_path, EXAMPLE.read_text(encoding="utf-8"))
    settings = load_settings(path)
    assert settings.pipeline.stages == list(DEFAULT_STAGES)
    assert backups(path) == []  # nothing to migrate
    assert path.read_text(encoding="utf-8") == EXAMPLE.read_text(encoding="utf-8")


def test_first_release_config_gets_all_stages(tmp_path):
    """The bug: a config from the first release kept animals and the search index off."""
    path = write(tmp_path, FIRST_RELEASE)
    settings = load_settings(path)
    assert settings.pipeline.stages == list(DEFAULT_STAGES)
    # the user's own settings and comments survive
    assert settings.pipeline.batch_size == 8 and settings.vlm.model == "qwen2.5vl:7b"
    text = path.read_text(encoding="utf-8")
    assert "# my model" in text and "# images claimed per batch" in text
    assert 'stages = ["faces", "vlm"]' not in text
    doc = tomlkit.parse(text)
    assert doc["config_version"] == CONFIG_VERSION
    assert text.index("config_version") < text.index("[paths]")
    # a backup of the original file
    [backup] = backups(path)
    assert backup.name.startswith("config.toml.bak-v1-")
    assert backup.read_text(encoding="utf-8") == FIRST_RELEASE
    # what happened is remembered for `vision-tagger migrate`
    done = config_mod.last_config_migration
    assert done is not None and done.before == 1 and done.after == CONFIG_VERSION
    assert done.written and done.backup == backup
    assert any("pipeline.stages" in note for note in done.notes)
    # loading again changes nothing
    load_settings(path)
    assert len(backups(path)) == 1


@pytest.mark.parametrize(
    "stages",
    [
        '["faces", "vlm", "embed"]',
        '["faces", "ages", "vlm", "embed"]   # "ages" is added automatically with "faces"',
        '["faces", "ages", "animals", "vlm", "embed"]',
    ],
)
def test_every_old_default_list_is_removed(tmp_path, stages):
    path = write(tmp_path, f"[pipeline]\nstages = {stages}\n")
    data, result = load_config(path)
    assert "stages" not in data["pipeline"]
    assert result.written and result.notes


def test_a_stage_list_the_user_chose_is_kept(tmp_path):
    path = write(tmp_path, '[pipeline]\nstages = ["faces", "embed"]   # no Ollama here\n')
    settings = load_settings(path)
    assert settings.pipeline.stages == ["faces", "ages", "embed"]
    text = path.read_text(encoding="utf-8")
    assert 'stages = ["faces", "embed"]   # no Ollama here' in text
    assert tomlkit.parse(text)["config_version"] == CONFIG_VERSION
    assert config_mod.last_config_migration.notes == []  # versioned, nothing else changed


def test_windows_line_endings_and_bom_survive(tmp_path):
    text = FIRST_RELEASE.replace("\n", "\r\n")
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))  # Notepad's UTF-8 with BOM
    settings = load_settings(path)
    assert settings.pipeline.stages == list(DEFAULT_STAGES)
    raw = path.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_a_newer_config_is_left_alone(tmp_path, caplog):
    text = f'config_version = {CONFIG_VERSION + 1}\n[pipeline]\nstages = ["faces", "vlm"]\n'
    path = write(tmp_path, text)
    settings = load_settings(path)
    assert settings.pipeline.stages == ["faces", "ages", "vlm"]  # used as it is
    assert path.read_text(encoding="utf-8") == text and backups(path) == []
    assert "newer than this app" in caplog.text


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a read-only folder (root ignores permissions)",
)
def test_read_only_config_is_migrated_in_memory(tmp_path, caplog):
    folder = tmp_path / "ro"
    folder.mkdir()
    path = write(folder, FIRST_RELEASE)
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    path.chmod(stat.S_IRUSR)
    try:
        settings = load_settings(path)
    finally:
        folder.chmod(stat.S_IRWXU)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert settings.pipeline.stages == list(DEFAULT_STAGES)
    assert path.read_text(encoding="utf-8") == FIRST_RELEASE
    assert "vision-tagger migrate" in caplog.text


def test_failed_write_keeps_the_file_and_uses_the_update(tmp_path, monkeypatch, caplog):
    path = write(tmp_path, FIRST_RELEASE)

    def locked(*_args, **_kwargs):
        raise PermissionError("locked by another program")

    monkeypatch.setattr(Path, "replace", locked)
    data, result = load_config(path)
    assert "stages" not in data["pipeline"]
    assert not result.written and "locked" in (result.error or "")
    assert path.read_text(encoding="utf-8") == FIRST_RELEASE
    assert backups(path) == [] and not list(tmp_path.glob("*.tmp-*"))
    assert "vision-tagger migrate" in caplog.text


def test_migrate_command_reports_the_config(tmp_path):
    from vision_tagger.cli import app

    write(tmp_path, FIRST_RELEASE.replace('data_dir = "data"', f"data_dir = '{tmp_path / 'd'}'"))
    runner = CliRunner()
    first = runner.invoke(app, ["migrate"])
    assert first.exit_code == 0, first.output
    assert f"version 1 -> {CONFIG_VERSION}" in first.output
    assert "config.toml.bak-v1-" in first.output and "pipeline.stages" in first.output
    config_mod.last_config_migration = None
    again = runner.invoke(app, ["migrate", "--check"])
    assert again.exit_code == 0, again.output
    assert f"version {CONFIG_VERSION}, up to date" in again.output


def test_init_and_the_desktop_app_write_current_configs(tmp_path, monkeypatch):
    from vision_tagger.cli import app
    from vision_tagger.desktop import ensure_user_config

    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    for path in (tmp_path / "config.toml", ensure_user_config(tmp_path / "desktop")):
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        assert doc["config_version"] == CONFIG_VERSION
        assert "stages" not in doc["pipeline"]


def test_pipeline_page_offers_stages_the_config_turns_off(settings):
    import re

    from fastapi.testclient import TestClient

    from vision_tagger.web.app import create_app

    settings.pipeline.stages = ["faces", "ages", "vlm"]
    with TestClient(create_app(settings)) as c:
        page = c.get("/pipeline").text
    boxes = dict(re.findall(r'name="stages" value="(\w+)"( checked)?', page))
    assert set(boxes) == set(DEFAULT_STAGES)  # every stage is listed
    assert {s for s, on in boxes.items() if on} == {"faces", "ages", "vlm"}
    assert page.count("off in config.toml") == 3  # visual search, animals, the search index
