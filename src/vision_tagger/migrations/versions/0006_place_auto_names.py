"""Automatic place names: ``places.auto_name``.

A place's display name (``places.name``) is now made unique when two places would get the
same automatic name ("Neustadt (Coburg), Germany" and "Neustadt (Mannheim), Germany"). The new
nullable column ``places.auto_name`` keeps the plain automatic name ("Neustadt, Germany"), so
places with the same automatic name that lie close together can still be joined, and a name can
be made unique again when places change.

Existing places were named before names were made unique: for the automatically named ones
the automatic name is their name (backfilled); for places the user named it is unknown (NULL)
until the place is named automatically again. The downgrade drops the column; display names
stay as they are.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("places") as batch:  # a plain ADD COLUMN, no table rebuild
        batch.add_column(sa.Column("auto_name", sa.Text(), nullable=True))
    places = sa.table(
        "places",
        sa.column("name", sa.Text),
        sa.column("auto_name", sa.Text),
        sa.column("user_named", sa.Boolean),
    )
    op.execute(
        places.update().where(places.c.user_named == sa.false()).values(auto_name=places.c.name)
    )


def downgrade() -> None:
    with op.batch_alter_table("places") as batch:
        batch.drop_column("auto_name")
