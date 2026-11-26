"""Covering indexes for the hot read paths (performance only, no data changes).

* ``faces``: ``(image_id)`` -> ``(image_id, ignored, cluster_id)`` and ``(cluster_id)`` ->
  ``(cluster_id, ignored, image_id, age)``. Face rows carry a 2 KB embedding, so person
  filters, person / image counts and age lookups read the much smaller index instead.
* ``image_properties``: ``(key, value)`` -> ``(key, value, image_id, confidence)``. The
  gallery's "image has key=value" checks become one index seek (before, every image scanned
  all rows with that value), and value counts and the "best examples" of a value (ordered by
  confidence) need no table lookups.
* ``stage_status``: ``(stage, status)`` -> ``(stage, status, image_id, stage_version)``: the
  runner claims pending images in image order without sorting the whole stage, and the
  dashboard's outdated-version counts are covered.

Only indexes change; on SQLite they are created directly (no table rebuild).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-26
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_faces_image_id", table_name="faces")
    op.drop_index("ix_faces_cluster_id", table_name="faces")
    op.create_index(
        "ix_faces_image_ignored_cluster", "faces", ["image_id", "ignored", "cluster_id"]
    )
    op.create_index(
        "ix_faces_cluster_ignored_image_age", "faces", ["cluster_id", "ignored", "image_id", "age"]
    )

    op.drop_index("ix_image_properties_key_value", table_name="image_properties")
    op.create_index(
        "ix_image_properties_key_value_image",
        "image_properties",
        ["key", "value", "image_id", "confidence"],
    )

    op.drop_index("ix_stage_status_stage_status", table_name="stage_status")
    op.create_index(
        "ix_stage_status_stage_status_image",
        "stage_status",
        ["stage", "status", "image_id", "stage_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_stage_status_stage_status_image", table_name="stage_status")
    op.create_index("ix_stage_status_stage_status", "stage_status", ["stage", "status"])

    op.drop_index("ix_image_properties_key_value_image", table_name="image_properties")
    op.create_index("ix_image_properties_key_value", "image_properties", ["key", "value"])

    op.drop_index("ix_faces_cluster_ignored_image_age", table_name="faces")
    op.drop_index("ix_faces_image_ignored_cluster", table_name="faces")
    op.create_index("ix_faces_cluster_id", "faces", ["cluster_id"])
    op.create_index("ix_faces_image_id", "faces", ["image_id"])
