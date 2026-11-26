from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vision_tagger import config as config_mod
from vision_tagger.cli import app

from .conftest import make_image


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    for i in range(3):
        make_image(photos / f"IMG_2020010{i + 1}_120000.jpg", seed=i)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[paths]\n"
        f"image_roots = ['{photos.as_posix()}']\n"
        f"data_dir = '{(tmp_path / 'data').as_posix()}'\n"
        "[vlm]\nbackend = 'fake'\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    return cfg, tmp_path


def invoke(*args):
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output


def test_cli_index_status_run_export(cli_env):
    cfg, tmp = cli_env
    out = invoke("-c", str(cfg), "index")
    assert "new 3" in out
    out = invoke("-c", str(cfg), "status")
    assert "pending" in out and "vlm" in out
    out = invoke("-c", str(cfg), "run", "--stage", "vlm", "--batch-size", "2")
    assert "completed" in out
    out = invoke("-c", str(cfg), "export", str(tmp / "out"), "--filter", "from=2020")
    assert "exported 3" in out
    assert len(list(Path(tmp / "out").glob("*.jpg"))) == 3


def test_cli_init_writes_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setenv("VT_PATHS__DATA_DIR", str(tmp_path / "data"))
    out = invoke("init")
    assert (tmp_path / "config.toml").exists()
    assert "database" in out


def test_cli_search(cli_env):
    cfg, _tmp = cli_env
    invoke("-c", str(cfg), "index")
    invoke("-c", str(cfg), "run", "--stage", "vlm", "--stage", "embed")
    out = invoke("-c", str(cfg), "search", "trees and a car", "--limit", "2")
    assert "mode=basic" in out and "semantic=on" in out
    assert out.count(".jpg") == 2


def test_cli_serve_remote_and_migrate(cli_env, monkeypatch):
    from vision_tagger.db.migrate import current_revision, head_revision, migration_engine

    from .test_migrations import build_at, url_of

    cfg, tmp = cli_env
    db_path = tmp / "data" / "vision_tagger.sqlite3"
    db_path.parent.mkdir(parents=True)
    build_at(db_path, "0003")
    started = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.append((app, kw)))

    result = CliRunner().invoke(app, ["-c", str(cfg), "serve", "--remote"])
    assert result.exit_code == 2 and "upload" in result.output  # remote mode needs a folder
    result = CliRunner().invoke(app, ["-c", str(cfg), "serve"])
    assert result.exit_code != 0 and not started  # outdated database: refused

    invoke("-c", str(cfg), "serve", "--remote", "--upload-dir", str(tmp / "up"), "--migrate")
    engine = migration_engine(url_of(db_path))
    assert current_revision(engine) == head_revision()
    engine.dispose()
    assert list(db_path.parent.glob("*.bak-0003-*"))  # backup made before upgrading
    assert len(started) == 1
