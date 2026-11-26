"""GPS coordinates as columns, named places and a reverse-geocoding cache.

Changes since 0004:

* new columns ``images.lat`` / ``images.lon`` (decimal degrees, index ``ix_images_lat_lon``)
  for the map and the area filter. They are filled from the GPS position the indexer has
  always stored in ``images.metadata`` (``{"gps": {"lat": ..., "lon": ...}}``); positions
  that are not numbers, out of range or exactly 0/0 (a common "no fix" value) stay empty.
* new table ``places`` (photos grouped by distance, with a name) and column
  ``images.place_id`` (foreign key, ``ON DELETE SET NULL``, indexed). Places are found by
  ``vision-tagger places`` or at the end of the next index run; the migration creates none.
* new table ``geocode_cache``: answers of the optional online reverse geocoder (Nominatim).

SQLite cannot add a foreign key to an existing table, so ``images`` is rebuilt (batch mode;
foreign keys are off while migrations run, so no child rows are touched). An inline
``ADD COLUMN ... REFERENCES`` would avoid the copy, but SQLAlchemy does not reflect its
``ON DELETE`` rule, so models and migrations would no longer compare equal.

The downgrade drops the columns and tables again (batch rebuild of ``images``) and removes the
place properties (``image_properties.source = 'place'``), which older versions cannot update.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "geocode_cache",
        sa.Column("key", sa.String(length=96), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "places",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("radius_m", sa.Float(), server_default="0", nullable=False),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("admin", sa.Text(), nullable=True),
        sa.Column("locality", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=16), server_default="geonames", nullable=False),
        sa.Column("user_named", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("images") as batch:
        batch.add_column(sa.Column("lat", sa.Float(), nullable=True))
        batch.add_column(sa.Column("lon", sa.Float(), nullable=True))
        batch.add_column(sa.Column("place_id", sa.Integer(), nullable=True))
        batch.create_index("ix_images_lat_lon", ["lat", "lon"], unique=False)
        batch.create_index("ix_images_place_id", ["place_id"], unique=False)
        batch.create_foreign_key(
            "fk_images_place_id_places", "places", ["place_id"], ["id"], ondelete="SET NULL"
        )

    # Backfill from the metadata JSON. CASE guards json_extract against invalid JSON (which
    # would abort the whole statement); the second statement drops unusable positions.
    op.execute(
        """
        UPDATE images
        SET lat = CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.gps.lat') END,
            lon = CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.gps.lon') END
        WHERE metadata LIKE '%"gps"%'
        """
    )
    op.execute(
        """
        UPDATE images SET lat = NULL, lon = NULL
        WHERE (lat IS NOT NULL OR lon IS NOT NULL)
          AND (lat IS NULL OR lon IS NULL
               OR typeof(lat) NOT IN ('real', 'integer')
               OR typeof(lon) NOT IN ('real', 'integer')
               OR lat < -90 OR lat > 90 OR lon < -180 OR lon > 180
               OR (lat = 0 AND lon = 0))
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM image_properties WHERE source = 'place'")
    op.drop_index("ix_images_place_id", table_name="images")
    op.drop_index("ix_images_lat_lon", table_name="images")
    with op.batch_alter_table("images") as batch:  # the rebuild drops the foreign key too
        batch.drop_column("place_id")
        batch.drop_column("lon")
        batch.drop_column("lat")
    op.drop_table("places")
    op.drop_table("geocode_cache")
