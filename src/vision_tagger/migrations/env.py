"""Alembic environment for vision-tagger.

Used by ``vision-tagger migrate`` (which passes an open connection) and by the alembic CLI
for development, e.g.::

    uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 upgrade head
    uv run alembic -x db=sqlite:///.scratch/schema.sqlite3 revision --autogenerate -m "..."
"""

from __future__ import annotations

from alembic import context

from vision_tagger.db.migrate import migration_engine
from vision_tagger.db.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    x = context.get_x_argument(as_dictionary=True)
    if x.get("db"):
        return x["db"]
    if config.get_main_option("sqlalchemy.url"):
        return config.get_main_option("sqlalchemy.url")
    from vision_tagger.config import load_settings

    return f"sqlite:///{load_settings().db_path.as_posix()}"


def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        render_as_batch=True,  # SQLite needs table rebuilds for most ALTERs
        compare_type=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(url=_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:  # vision-tagger migrate: shared connection, one transaction
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = migration_engine(_url())
    with engine.connect() as conn:
        _configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
