"""ProviderAssetRepository: 第三方供应商资产库 assetId 缓存的异步读写。"""

from __future__ import annotations

from sqlalchemy import select

from lib.db.models.provider_asset import ProviderAsset
from lib.db.repositories.base import BaseRepository


class ProviderAssetRepository(BaseRepository):
    async def get(self, provider: str, content_hash: str) -> ProviderAsset | None:
        return (
            await self.session.execute(
                select(ProviderAsset).where(
                    ProviderAsset.provider == provider,
                    ProviderAsset.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()

    async def upsert(
        self,
        *,
        provider: str,
        content_hash: str,
        asset_id: str,
        status: str,
        asset_type: str = "Image",
    ) -> ProviderAsset:
        row = await self.get(provider, content_hash)
        if row is None:
            row = ProviderAsset(
                provider=provider,
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
