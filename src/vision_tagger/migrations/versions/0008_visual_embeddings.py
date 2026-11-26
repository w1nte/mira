"""Visual search: an embedding of every photo in a shared image-text space.

Changes since 0007:

* new table ``visual_embeddings``: one row per photo with the vector the ``visual`` pipeline
  stage computed from the pixels (float16, L2-normalised), the vector space it belongs to
  (``model``), the exact model files and preprocessing (``stage_version``: a change re-embeds
  the photos), its dimension and when it was written. Rows go with their image (``ON DELETE
  CASCADE``). The index on ``(model, created_at)`` lets the search index count and load one
  model's rows without reading the vectors.

The table is new, so existing data is not touched; the ``visual`` stage fills it (images get
their stage rows when the stage first runs).

The downgrade drops the table and the ``visual`` stage rows: after an upgrade the stage then
runs again instead of treating photos as done whose vectors are gone.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "visual_embeddings",
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("stage_version", sa.String(length=64), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("image_id"),
    )
    with op.batch_alter_table("visual_embeddings") as batch:  # plain CREATE INDEX
        batch.create_index(
            "ix_visual_embeddings_model_created", ["model", "created_at"], unique=False
        )


def downgrade() -> None:
    stages = sa.table("stage_status", sa.column("stage", sa.String))
    op.execute(stages.delete().where(stages.c.stage == "visual"))
    with op.batch_alter_table("visual_embeddings") as batch:
        batch.drop_index("ix_visual_embeddings_model_created")
    op.drop_table("visual_embeddings")
