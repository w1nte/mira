"""Saved collections.

New table ``collections``: a name for a gallery filter (the ``ImageFilter`` query string, e.g.
``p=3&t=location_type:beach``), shown on the home page. Existing data is not touched; the
downgrade drops the table (and the saved collections with it).

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), server_default="", nullable=False),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("collections")
