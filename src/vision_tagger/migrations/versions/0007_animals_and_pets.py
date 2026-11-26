"""Animals found in photos, and pets: groups of photos of the same animal that can be named.

Changes since 0006:

* new table ``pets``: a group of animal detections (see ``animals.grouping``) with an optional
  name, the species most of its photos show, ``locked`` (named or edited by the user: kept when
  the animals are grouped again), the average embedding, the number of detections and the
  detection shown as its picture.
* new table ``animals``: one row per animal the detector found (box in original pixels,
  species, score, DINOv2 embedding, size), its pet (``ON DELETE SET NULL``), whether the user
  assigned it (``assignment``), removed it (``ignored``) and whether it is good enough to be
  grouped automatically (``groupable``). Rows go with their image (``ON DELETE CASCADE``).
  Two covering indexes serve "animals of an image" and "animals of a pet", like the faces ones.

Both tables are new, so existing data is not touched; the ``animals`` pipeline stage fills them
(images get their stage rows when the stage first runs).

The downgrade drops both tables, the ``pet`` / ``animal`` properties they produced
(``image_properties.source = 'animal'``) and the ``animals`` stage rows: after an upgrade the
stage then runs again instead of treating photos as done whose animals are gone.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("species", sa.String(length=32), nullable=False),
        sa.Column("locked", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("centroid", sa.LargeBinary(), nullable=True),
        sa.Column("animal_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cover_animal_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "animals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("species", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("box_px", sa.Integer(), nullable=False),
        sa.Column("pet_id", sa.Integer(), nullable=True),
        sa.Column("assignment", sa.String(length=16), nullable=False),
        sa.Column("ignored", sa.Boolean(), nullable=False),
        sa.Column("groupable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pet_id"], ["pets.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("animals") as batch:  # plain CREATE INDEX, no table rebuild
        batch.create_index(
            "ix_animals_image_ignored_pet", ["image_id", "ignored", "pet_id"], unique=False
        )
        batch.create_index(
            "ix_animals_pet_ignored_image", ["pet_id", "ignored", "image_id"], unique=False
        )


def downgrade() -> None:
    props = sa.table("image_properties", sa.column("source", sa.String))
    op.execute(props.delete().where(props.c.source == "animal"))
    stages = sa.table("stage_status", sa.column("stage", sa.String))
    op.execute(stages.delete().where(stages.c.stage == "animals"))
    with op.batch_alter_table("animals") as batch:
        batch.drop_index("ix_animals_pet_ignored_image")
        batch.drop_index("ix_animals_image_ignored_pet")
    op.drop_table("animals")
    op.drop_table("pets")
