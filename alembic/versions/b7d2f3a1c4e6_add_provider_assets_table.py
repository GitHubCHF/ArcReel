"""add_provider_assets_table

Revision ID: b7d2f3a1c4e6
Revises: a3f1c9b27e54
Create Date: 2026-06-17 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d2f3a1c4e6"
down_revision: str | Sequence[str] | None = "a3f1c9b27e54"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "provider_assets" in inspector.get_table_names():
        return
    op.create_table(
        "provider_assets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("asset_type", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "content_hash", name="uq_provider_asset_hash"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("provider_assets")
