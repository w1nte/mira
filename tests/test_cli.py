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
    args = ["-c", str(cfg), "serve", "--remote", "--upload-dir", str(tmp / "up"), "--migrate"]
    result = CliRunner().invoke(app, args)  # remote mode without a password: refused
    assert result.exit_code == 2 and "VT_WEB__PASSWORD" in result.output
    assert not started and not list(db_path.parent.glob("*.bak-*"))  # before touching the db

    monkeypatch.setenv("VT_WEB__PASSWORD", "a long passphrase")
    invoke(*args)
    engine = migration_engine(url_of(db_path))
    assert current_revision(engine) == head_revision()
    engine.dispose()
    assert list(db_path.parent.glob("*.bak-0003-*"))  # backup made before upgrading
    assert len(started) == 1


def test_cli_serve_refuses_network_addresses_without_a_password(cli_env, monkeypatch):
    cfg, _tmp = cli_env
    started = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.append(kw))
    monkeypatch.delenv("VT_WEB__PASSWORD", raising=False)
    for host in ("0.0.0.0", "::", "192.168.1.20"):
        result = CliRunner().invoke(app, ["-c", str(cfg), "serve", "--host", host])
        assert result.exit_code == 2, host
        assert "reached from other computers" in result.output, host
    assert not started
    invoke("-c", str(cfg), "serve", "--host", "127.0.0.1")  # this computer only: fine
    monkeypatch.setenv("VT_WEB__PASSWORD", "a long passphrase")
    invoke("-c", str(cfg), "serve", "--host", "0.0.0.0")
    assert [kw["host"] for kw in started] == ["127.0.0.1", "0.0.0.0"]


def test_cli_serve_refuses_the_example_password_and_trusts_only_set_proxies(cli_env, monkeypatch):
    cfg, _tmp = cli_env
    started = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.append(kw))
    monkeypatch.setenv("VT_WEB__PASSWORD", "choose-a-good-one")
    result = CliRunner().invoke(app, ["-c", str(cfg), "serve", "--host", "0.0.0.0"])
    assert result.exit_code == 2 and "example" in result.output and not started
    monkeypatch.setenv("VT_WEB__PASSWORD", "short")
    out = invoke("-c", str(cfg), "serve", "--host", "0.0.0.0")
    assert "shorter than 12 characters" in out
    assert started[-1]["proxy_headers"] is True
    assert started[-1]["forwarded_allow_ips"] == ["127.0.0.1"]
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")  # uvicorn's own setting (Traefik compose)
    invoke("-c", str(cfg), "serve", "--host", "0.0.0.0")
    assert started[-1]["forwarded_allow_ips"] == "*"
    monkeypatch.setenv("VT_WEB__TRUSTED_PROXIES", '["172.18.0.0/16"]')  # Mira's setting wins
    invoke("-c", str(cfg), "serve", "--host", "0.0.0.0")
    assert started[-1]["forwarded_allow_ips"] == ["172.18.0.0/16"]
