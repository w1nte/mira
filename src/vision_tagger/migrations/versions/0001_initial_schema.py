"""Initial schema (first released version, commit c1e357b).

Frozen: never edit this file. Schema changes go into new revisions.

Revision ID: 0001
Revises:
Create Date: 2026-11-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = [
    "faces",
    "vlm_results",
    "stage_status",
    "person_references",
    "image_properties",
    "duplicate_files",
    "clusters",
    "runs",
    "property_aliases",
    "persons",
    "images",
    "control",
    "app_settings",
]


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "control",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("desired_state", sa.String(length=16), nullable=False),
        sa.Column("active_stage", sa.String(length=32), nullable=True),
        sa.Column("active_run_id", sa.Integer(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("progress", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("path_key", sa.Text(), nullable=False),
        sa.Column("root", sa.Text(), nullable=False),
        sa.Column("rel_path", sa.Text(), nullable=False),
        sa.Column("folder", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("ext", sa.String(length=16), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mtime", sa.Float(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=True),
        sa.Column("date_source", sa.String(length=32), nullable=False),
        sa.Column("phash", sa.String(length=32), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("camera", sa.Text(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=False),
        sa.Column("missing", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_images_camera"), "images", ["camera"], unique=False)
    op.create_index(op.f("ix_images_ext"), "images", ["ext"], unique=False)
    op.create_index(op.f("ix_images_folder"), "images", ["folder"], unique=False)
    op.create_index(op.f("ix_images_path_key"), "images", ["path_key"], unique=True)
    op.create_index(op.f("ix_images_phash"), "images", ["phash"], unique=True)
    op.create_index(op.f("ix_images_recorded_at"), "images", ["recorded_at"], unique=False)
    op.create_table(
        "persons",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "property_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("from_value", sa.Text(), nullable=False),
        sa.Column("to_value", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", "from_value"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("stages", sa.Text(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("limit", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("processed", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=True),
        sa.Column("suggested_person_id", sa.Integer(), nullable=True),
        sa.Column("suggestion_score", sa.Float(), nullable=True),
        sa.Column("locked", sa.Boolean(), nullable=False),
        sa.Column("centroid", sa.LargeBinary(), nullable=True),
        sa.Column("face_count", sa.Integer(), nullable=False),
        sa.Column("cover_face_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["suggested_person_id"], ["persons.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_clusters_person_id"), "clusters", ["person_id"], unique=False)
    op.create_table(
        "duplicate_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("path_key", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mtime", sa.Float(), nullable=False),
        sa.Column("found_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_duplicate_files_image_id"), "duplicate_files", ["image_id"], unique=False
    )
    op.create_index(
        op.f("ix_duplicate_files_path_key"), "duplicate_files", ["path_key"], unique=True
    )
    op.create_table(
        "image_properties",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_image_properties_image_key", "image_properties", ["image_id", "key"], unique=False
    )
    op.create_index(
        "ix_image_properties_key_value", "image_properties", ["key", "value"], unique=False
    )
    op.create_table(
        "person_references",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=True),
        sa.Column("face_id", sa.Integer(), nullable=True),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_person_references_person_id"), "person_references", ["person_id"], unique=False
    )
    op.create_table(
        "stage_status",
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("stage_version", sa.String(length=64), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("image_id", "stage"),
    )
    op.create_index(
        "ix_stage_status_stage_status", "stage_status", ["stage", "status"], unique=False
    )
    op.create_table(
        "vlm_results",
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("image_id"),
    )
    op.create_table(
        "faces",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("det_score", sa.Float(), nullable=False),
        sa.Column("landmarks", sa.JSON(), nullable=True),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("gender", sa.String(length=1), nullable=True),
        sa.Column("face_px", sa.Integer(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column("assignment", sa.String(length=16), nullable=False),
        sa.Column("ignored", sa.Boolean(), nullable=False),
        sa.Column("clusterable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["cluster_id"], ["clusters.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_faces_cluster_id"), "faces", ["cluster_id"], unique=False)
    op.create_index(op.f("ix_faces_image_id"), "faces", ["image_id"], unique=False)


def downgrade() -> None:
    for table in TABLES:
        op.drop_table(table)
