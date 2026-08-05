"""ProviderAssetRepository: 第三方供应商资产库 assetId 缓存的异步读写。"""

from __future__ import annotations

from sqlalchemy import delete, select

from lib.db.models.provider_asset import ProviderAsset
from lib.db.repositories.base import BaseRepository


class ProviderAssetRepository(BaseRepository):
    async def get(self, provider: str, key_hash: str, content_hash: str) -> ProviderAsset | None:
        return (
            await self.session.execute(
                select(ProviderAsset).where(
                    ProviderAsset.provider == provider,
                    ProviderAsset.key_hash == key_hash,
                    ProviderAsset.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()

    async def upsert(
        self,
        *,
        provider: str,
        key_hash: str,
        content_hash: str,
        asset_id: str,
        status: str,
        asset_type: str = "Image",
    ) -> ProviderAsset:
        row = await self.get(provider, key_hash, content_hash)
        if row is None:
            row = ProviderAsset(
                provider=provider,
                key_hash=key_hash,
                content_hash=content_hash,
                asset_id=asset_id,
                status=status,
                asset_type=asset_type,
            )
            self.session.add(row)
        else:
            row.asset_id = asset_id
            row.status = status
            row.asset_type = asset_type
        await self.session.flush()
        return row

    async def delete_all(self, provider: str | None = None) -> int:
        """清空供应商资产缓存,返回删除行数。provider 为 None 时清空全部。"""
        stmt = delete(ProviderAsset)
        if provider is not None:
            stmt = stmt.where(ProviderAsset.provider == provider)
        result = await self.session.execute(stmt)
        await self.session.flush()
        # DML 语句返回的实际是 CursorResult(带 rowcount)；基类 Result 类型上无此属性。
        return getattr(result, "rowcount", 0) or 0
