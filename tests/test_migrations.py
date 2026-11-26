"""Database migrations: models and migrations must agree, old databases must upgrade."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import select
from typer.testing import CliRunner

from vision_tagger.db.migrate import (
    DatabaseTooNew,
    MigrationRequired,
    alembic_config,
    current_revision,
    head_revision,
    migration_engine,
    schema_status,
    upgrade,
    upgrade_if_needed,
)
from vision_tagger.db.models import Base, Image, Person
from vision_tagger.db.session import Database


def url_of(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def build_at(path: Path, revision: str, *, keep_version_table: bool = True) -> None:
    """Create a database at an old revision.

    Without the ``alembic_version`` table it looks like a pre-migrations database.
    """
    engine = migration_engine(url_of(path))
    cfg = alembic_config(url_of(path))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)
        if not keep_version_table:
            conn.exec_driver_sql("DROP TABLE alembic_version")
    engine.dispose()


FIRST_VERSION_ROWS = [
    "INSERT INTO images (id, path, path_key, root, rel_path, folder, filename, ext, size_bytes,"
    " mtime, date_source, phash, metadata, indexed_at, missing) VALUES (1, '/p/a.jpg', '/p/a.jpg',"
    " '/p', 'a.jpg', '', 'a.jpg', '.jpg', 1, 0, 'exif', 'abcdef0123456789', '{}',"
    " '2024-01-01 00:00:00', 0)",
    "INSERT INTO persons (id, name, created_at) VALUES (1, 'Anna', '2024-01-01 00:00:00')",
    "INSERT INTO clusters (id, person_id, locked, face_count, created_at)"
    " VALUES (1, 1, 1, 1, '2024-01-01 00:00:00')",
    "INSERT INTO faces (id, image_id, x1, y1, x2, y2, det_score, embedding, face_px,"
    " cluster_id, assignment, ignored, clusterable)"
    " VALUES (1, 1, 0, 0, 1, 1, 0.9, x'00', 50, 1, 'manual', 0, 1)",
    "INSERT INTO stage_status (image_id, stage, status, attempts) VALUES (1, 'faces', 'done', 0)",
    "INSERT INTO image_properties (image_id, key, value, source)"
    " VALUES (1, 'year', '2024', 'exif')",
    "INSERT INTO duplicate_files (image_id, path, path_key, size_bytes, mtime, found_at)"
    " VALUES (1, '/p/b.jpg', '/p/b.jpg', 1, 0, '2024-01-01 00:00:00')",
]


def fill_first_version(path: Path) -> None:
    con = sqlite3.connect(path)
    for sql in FIRST_VERSION_ROWS:
        con.execute(sql)
    con.commit()
    con.close()


def counts(path: Path) -> dict[str, int]:
    con = sqlite3.connect(path)
    tables = [
        "images",
        "persons",
        "clusters",
        "faces",
        "stage_status",
        "image_properties",
        "duplicate_files",
    ]
    out = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    con.close()
    return out


def test_migrations_match_models(tmp_path):
    """Running every migration from scratch must produce exactly the current models.

    If this fails, a model change is missing its migration (see CLAUDE.md).
    """
    path = tmp_path / "scratch.sqlite3"
    result = upgrade(url_of(path), backup=False)
    assert result.after == head_revision()
    engine = migration_engine(url_of(path))
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert diff == [], f"models and migrations differ - add a migration: {diff}"


def test_new_database_is_created_and_stamped(tmp_path):
    db = Database(tmp_path / "new.sqlite3")
    db.create_all()
    assert current_revision(db.engine) == head_revision()
    db.create_all()  # idempotent
    db.dispose()


def test_first_version_database_upgrades_with_data(tmp_path):
    path = tmp_path / "v1.sqlite3"
    build_at(path, "0001", keep_version_table=False)  # exactly what version 1 created
    fill_first_version(path)
    before = counts(path)

    db = Database(path)
    with pytest.raises(MigrationRequired, match="vision-tagger migrate"):
        db.create_all()
    db.dispose()

    result = upgrade(url_of(path))
    assert result.state_before == "unversioned" and result.after == head_revision()
    assert result.backup is not None and result.backup.exists()
    assert counts(result.backup) == before  # the backup is a full copy
    assert counts(path) == before  # nothing lost (no cascade deletes during the migration)

    db = Database(path)
    db.create_all()  # now accepted
    with db.write() as s:
        img = s.get(Image, 1)
        assert img.dhash == ""
        s.get(Person, 1).birth_year = 1990
        # pHash is no longer unique: a distinct image with the same pHash can be stored
        s.add(
            Image(
                path="/p/c.jpg",
                path_key="/p/c.jpg",
                root="/p",
                rel_path="c.jpg",
                filename="c.jpg",
                ext=".jpg",
                size_bytes=1,
                mtime=0,
                phash=img.phash,
                dhash="ffff",
                meta={},
            )
        )
    with db.session() as s:
        assert len(s.scalars(select(Image)).all()) == 2
    db.dispose()
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert "image_embeddings" in {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    con.close()


def test_partially_patched_legacy_database(tmp_path):
    """Pre-migration builds may already have some of the new columns; upgrades must cope."""
    path = tmp_path / "patched.sqlite3"
    build_at(path, "0001", keep_version_table=False)
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE persons ADD COLUMN birth_year INTEGER")
    con.execute("ALTER TABLE images ADD COLUMN dhash VARCHAR(32) NOT NULL DEFAULT ''")
    con.execute("CREATE INDEX ix_images_dhash ON images (dhash)")
    con.commit()
    con.close()
    result = upgrade(url_of(path), backup=False)
    assert result.after == head_revision()
    engine = migration_engine(url_of(path))
    assert schema_status(engine).state == "current"
    engine.dispose()


def test_downgrade_and_upgrade_again(tmp_path):
    path = tmp_path / "roundtrip.sqlite3"
    upgrade(url_of(path), backup=False)
    from vision_tagger.db.migrate import downgrade

    downgrade(url_of(path), "0001", backup=False)
    engine = migration_engine(url_of(path))
    assert current_revision(engine) == "0001"
    engine.dispose()
    assert upgrade(url_of(path), backup=False).after == head_revision()


def test_cli_migrate(tmp_path, monkeypatch):
    from vision_tagger import config as config_mod
    from vision_tagger.cli import app

    data = tmp_path / "data"
    data.mkdir()
    build_at(data / "vision_tagger.sqlite3", "0001", keep_version_table=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setenv("VT_PATHS__DATA_DIR", str(data))
    runner = CliRunner()

    r = runner.invoke(app, ["status"])
    assert r.exit_code == 2 and "vision-tagger migrate" in r.output
    r = runner.invoke(app, ["migrate", "--check"])
    assert r.exit_code == 1 and "pre-migrations" in r.output
    r = runner.invoke(app, ["migrate"])
    assert r.exit_code == 0, r.output
    assert "migrated: pre-migrations -> " + head_revision() in r.output and "backup:" in r.output
    r = runner.invoke(app, ["migrate", "--check"])
    assert r.exit_code == 0 and "up to date" in r.output
    assert runner.invoke(app, ["status"]).exit_code == 0


def test_database_of_a_newer_version_is_refused(tmp_path, monkeypatch):
    """A downgrade of the app must not touch (or half-migrate) the newer database."""
    from vision_tagger import config as config_mod
    from vision_tagger.cli import app

    path = tmp_path / "data" / "vision_tagger.sqlite3"
    path.parent.mkdir()
    upgrade(url_of(path), backup=False)
    con = sqlite3.connect(path)
    con.execute("UPDATE alembic_version SET version_num = '0999'")
    con.commit()
    con.close()
    engine = migration_engine(url_of(path))
    assert schema_status(engine).state == "newer"
    engine.dispose()
    with pytest.raises(DatabaseTooNew, match="newer version"):
        upgrade(url_of(path))
    with pytest.raises(DatabaseTooNew):
        upgrade_if_needed(path)  # what the desktop app and `serve --migrate` do on start
    db = Database(path)
    with pytest.raises(DatabaseTooNew):
        db.create_all()
    db.dispose()
    assert list(path.parent.iterdir()) == [path]  # no backup, nothing changed

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "_config_file", None)
    monkeypatch.setenv("VT_PATHS__DATA_DIR", str(path.parent))
    for args in (["migrate"], ["migrate", "--check"]):
        r = CliRunner().invoke(app, args)
        assert r.exit_code == 3 and "newer version" in r.output, r.output
    r = CliRunner().invoke(app, ["status"])
    assert r.exit_code == 2 and "newer version" in r.output


def test_0003_replaces_indexes_and_keeps_data(tmp_path):
    """0003 only swaps indexes (covering ones for the hot queries); data must not change."""
    path = tmp_path / "v2.sqlite3"
    build_at(path, "0002")
    fill_first_version(path)
    before = counts(path)
    upgrade(url_of(path), backup=False)
    assert counts(path) == before
    con = sqlite3.connect(path)
    indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    assert {
        "ix_faces_image_ignored_cluster",
        "ix_faces_cluster_ignored_image_age",
        "ix_image_properties_key_value_image",
        "ix_stage_status_stage_status_image",
    } <= indexes
    assert (
        not {
            "ix_faces_image_id",
            "ix_faces_cluster_id",
            "ix_image_properties_key_value",
            "ix_stage_status_stage_status",
        }
        & indexes
    )
