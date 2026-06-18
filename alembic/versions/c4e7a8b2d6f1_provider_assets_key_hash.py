"""provider_assets add key_hash dimension

Revision ID: c4e7a8b2d6f1
Revises: b7d2f3a1c4e6
Create Date: 2026-06-17 19:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e7a8b2d6f1"
down_revision: str | Sequence[str] | None = "b7d2f3a1c4e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create(unique_cols: list[str], name: str) -> None:
    op.create_table(
        "provider_assets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("asset_type", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(*unique_cols, name=name),
    )


def upgrade() -> None:
    """Upgrade schema.

    旧表无 key_hash 维度,缓存的 assetId 未绑定密钥;直接 drop + recreate(顺带清空旧缓存,
    它们会在下次生成时按新维度重建),避免 SQLite 改唯一约束的繁琐表重建。
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "provider_assets" in inspector.get_table_names():
        op.drop_table("provider_assets")
    _create(["provider", "key_hash", "content_hash"], "uq_provider_asset_key_hash")


def downgrade() -> None:
    """Downgrade schema: 回到无 key_hash 的旧结构(同样清空)。"""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "provider_assets" in inspector.get_table_names():
        op.drop_table("provider_assets")
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
