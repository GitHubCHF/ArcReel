"""ProviderAsset ORM: 第三方供应商资产库 assetId 缓存。

钛动(tecdo)等供应商要求"真人/角色图"先登记到其资产库(异步过审)再用 ``asset://{id}``
引用。登记是异步的且每次都重做很浪费,故按图片内容哈希持久化缓存 assetId 复用。

资产在不同密钥(X-App-Secret)之间是隔离的:一个 key 登记的 assetId 换 key 后不可用。
故缓存再按 ``key_hash``(密钥 sha256,不存明文)分一层,唯一键为
``(provider, key_hash, content_hash)``。
"""

from __future__ import annotations

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from lib.db.base import Base, TimestampMixin


class ProviderAsset(TimestampMixin, Base):
    __tablename__ = "provider_assets"
    __table_args__ = (UniqueConstraint("provider", "key_hash", "content_hash", name="uq_provider_asset_key_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    # 密钥指纹(sha256(api_key)),按密钥隔离资产缓存;不存明文密钥。
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(16), nullable=False, default="Image")
