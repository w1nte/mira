"""Manual face ages: the user can set a person's age in a photo.

Changes since 0008:

* new nullable column ``faces.manual_age``: the age the user set for this face (when the face
  age estimate doesn't match the photo date and the date is right). It is used instead of the
  estimate ``faces.age`` for date checks, birth-year estimates and sorting by age, and it is
  carried over when the faces of a photo are detected again.

Existing faces get NULL (no manual age); no data is changed. The downgrade drops the column.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("faces", schema=None) as batch_op:
        batch_op.add_column(sa.Column("manual_age", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("faces", schema=None) as batch_op:
        batch_op.drop_column("manual_age")
