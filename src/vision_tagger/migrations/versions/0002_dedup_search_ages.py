"""Stricter dedup (dhash), search index (image_embeddings) and birth years.

Changes since 0001:

* ``images.phash`` is no longer unique (only indexed): distinct images may share a pHash,
  duplicates are now ``same pHash AND close dHash``.
* new column ``images.dhash`` (+ index).
* new column ``persons.birth_year``.
* new table ``image_embeddings`` (search index).

Every step checks the current state first, because databases created by pre-migration
versions of vision-tagger may already contain some of these changes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _indexes(table: str) -> dict[str, dict]:
    return {i["name"]: i for i in sa.inspect(op.get_bind()).get_indexes(table)}


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    # images.phash: unique index -> plain index
    phash = _indexes("images").get("ix_images_phash")
    if phash is None or phash.get("unique"):
        if phash is not None:
            op.drop_index("ix_images_phash", table_name="images")
        op.create_index("ix_images_phash", "images", ["phash"], unique=False)

    # images.dhash
    if "dhash" not in _columns("images"):
        op.add_column(
            "images",
            sa.Column("dhash", sa.String(length=32), nullable=False, server_default=""),
        )
    if "ix_images_dhash" not in _indexes("images"):
        op.create_index("ix_images_dhash", "images", ["dhash"], unique=False)

    # persons.birth_year
    if "birth_year" not in _columns("persons"):
        op.add_column("persons", sa.Column("birth_year", sa.Integer(), nullable=True))

    # image_embeddings (search index)
    if "image_embeddings" not in _tables():
        op.create_table(
            "image_embeddings",
            sa.Column("image_id", sa.Integer(), nullable=False),
            sa.Column("model", sa.Text(), nullable=False),
            sa.Column("dim", sa.Integer(), nullable=False),
            sa.Column("vector", sa.LargeBinary(), nullable=False),
            sa.Column("doc_hash", sa.String(length=40), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("image_id"),
        )
    if "ix_image_embeddings_created_at" not in _indexes("image_embeddings"):
        op.create_index(
            "ix_image_embeddings_created_at", "image_embeddings", ["created_at"], unique=False
        )


def downgrade() -> None:
    # Fails if images now share a pHash (they could not exist in the 0001 schema).
    op.drop_index("ix_image_embeddings_created_at", table_name="image_embeddings")
    op.drop_table("image_embeddings")
    with op.batch_alter_table("persons") as batch:
        batch.drop_column("birth_year")
    op.drop_index("ix_images_dhash", table_name="images")
    with op.batch_alter_table("images") as batch:
        batch.drop_column("dhash")
    op.drop_index("ix_images_phash", table_name="images")
    op.create_index("ix_images_phash", "images", ["phash"], unique=True)
