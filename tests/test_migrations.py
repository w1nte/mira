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
    MigrationRequired,
    alembic_config,
    current_revision,
    head_revision,
    migration_engine,
    schema_status,
    upgrade,
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


def test_0006_keeps_the_automatic_place_names(tmp_path):
    from vision_tagger.db.migrate import downgrade

    path = tmp_path / "v5.sqlite3"
    build_at(path, "0005")
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO places (id, name, lat, lon, source, user_named, created_at)"
        " VALUES (1, 'Berlin, Germany', 52.5, 13.4, 'geonames', 0, '2024-01-01 00:00:00')"
    )
    con.execute(
        "INSERT INTO places (id, name, lat, lon, source, user_named, created_at)"
        " VALUES (2, 'Home', 48.8, 2.3, 'user', 1, '2024-01-01 00:00:00')"
    )
    con.commit()
    con.close()
    upgrade(url_of(path), backup=False)
    con = sqlite3.connect(path)
    rows = dict(con.execute("SELECT name, auto_name FROM places").fetchall())
    assert rows == {"Berlin, Germany": "Berlin, Germany", "Home": None}  # user names: unknown
    con.close()
    downgrade(url_of(path), "0005", backup=False)
    con = sqlite3.connect(path)
    assert "auto_name" not in {r[1] for r in con.execute("PRAGMA table_info(places)")}
    assert con.execute("SELECT count(*) FROM places").fetchone() == (2,)
    con.close()


def test_0005_backfills_gps_columns_and_downgrades(tmp_path):
    """0005 copies the GPS position from the metadata JSON into images.lat / images.lon."""
    from vision_tagger.db.migrate import downgrade

    path = tmp_path / "v4.sqlite3"
    build_at(path, "0004")
    fill_first_version(path)  # image 1: metadata '{}'
    metadata = {
        2: '{"gps": {"lat": 52.520008, "lon": 13.404954, "alt": 34.0}, "make": "Apple"}',
        3: '{"gps": {"lat": -33.856784, "lon": 151.215297}}',
        4: '{"gps": {"lat": 40.689247, "lon": -74.044502}}',
        5: '{"gps": {"lat": 0, "lon": 0}}',  # "no fix"
        6: '{"gps": {"lat": 123.0, "lon": 10.0}}',  # out of range
        7: '{"gps": {"lat": "north", "lon": 10.0}}',  # not a number
        8: '{"gps": {"lat": 10.0}}',  # half a position
        9: 'not json, but mentions "gps"',  # must not abort the migration
    }
    con = sqlite3.connect(path)
    for image_id, meta in metadata.items():
        name = f"{image_id}.jpg"
        con.execute(
            "INSERT INTO images (id, path, path_key, root, rel_path, folder, filename, ext,"
            " size_bytes, mtime, date_source, phash, metadata, indexed_at, missing)"
            " VALUES (?, ?, ?, '/p', ?, '', ?, '.jpg', 1, 0, 'exif', ?, ?,"
            " '2024-01-01 00:00:00', 0)",
            (image_id, f"/p/{name}", f"/p/{name}", name, name, f"{image_id:016x}", meta),
        )
    con.commit()
    con.close()
    before = counts(path)

    upgrade(url_of(path), backup=False)
    assert counts(path) == before  # the images table rebuild keeps every row and child row
    con = sqlite3.connect(path)
    got = {i: (lat, lon) for i, lat, lon in con.execute("SELECT id, lat, lon FROM images")}
    assert got[2] == (52.520008, 13.404954)
    assert got[3] == (-33.856784, 151.215297)
    assert got[4] == (40.689247, -74.044502)
    for image_id in (1, 5, 6, 7, 8, 9):
        assert got[image_id] == (None, None), image_id
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    meta = con.execute("SELECT metadata FROM images WHERE id = 2").fetchone()[0]
    assert '"make": "Apple"' in meta  # the metadata itself is untouched
    # deleting a place leaves its photos without one (ON DELETE SET NULL)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute(
        "INSERT INTO places (id, name, lat, lon, created_at)"
        " VALUES (1, 'Berlin, Germany', 52.52, 13.4, '2024-01-01 00:00:00')"
    )
    con.execute("UPDATE images SET place_id = 1 WHERE id = 2")
    con.execute(
        "INSERT INTO image_properties (image_id, key, value, source)"
        " VALUES (2, 'place', 'Berlin, Germany', 'place')"
    )
    con.commit()
    con.execute("DELETE FROM places WHERE id = 1")
    assert con.execute("SELECT place_id FROM images WHERE id = 2").fetchone()[0] is None
    con.execute(
        "INSERT INTO places (id, name, lat, lon, created_at)"
        " VALUES (2, 'Berlin', 52.5, 13.4, '2024-01-01 00:00:00')"
    )
    con.execute("UPDATE images SET place_id = 2 WHERE id = 2")
    con.commit()
    con.close()

    downgrade(url_of(path), "0004", backup=False)
    con = sqlite3.connect(path)
    columns = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    assert not {"lat", "lon", "place_id"} & columns
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not {"places", "geocode_cache"} & tables
    left = con.execute("SELECT count(*) FROM image_properties WHERE source = 'place'").fetchone()
    assert left == (0,)
    assert con.execute("SELECT count(*) FROM images").fetchone()[0] == before["images"]
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    con.close()
