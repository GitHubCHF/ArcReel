"""ProviderAsset ORM: 第三方供应商资产库 assetId 缓存。

钛动(tecdo)等供应商要求"真人/角色图"先登记到其资产库(异步过审)再用 ``asset://{id}``
引用。登记是异步的且每次都重做很浪费,故按图片内容哈希持久化缓存 assetId 复用。
"""

from __future__ import annotations

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from lib.db.base import Base, TimestampMixin


class ProviderAsset(TimestampMixin, Base):
    __tablename__ = "provider_assets"
    __table_args__ = (UniqueConstraint("provider", "content_hash", name="uq_provider_asset_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(16), nullable=False, default="Image")
