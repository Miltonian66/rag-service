"""initial: vector extension, chunks table, ivfflat index

Revision ID: 0001
Revises:
Create Date: 2026-01-01
"""

import pgvector.sqlalchemy
import sqlalchemy as sa

from alembic import op
from app.config import EMBED_DIM  # single source of truth, shared with the ORM model

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=512), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBED_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_source", "chunks", ["source"])
    # ivfflat cosine index for approximate nearest-neighbour search.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_cosine ON chunks "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_cosine", table_name="chunks")
    op.drop_index("ix_chunks_source", table_name="chunks")
    op.drop_table("chunks")
    op.execute("DROP EXTENSION IF EXISTS vector")
