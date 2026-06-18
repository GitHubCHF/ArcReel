"""ProviderAssetRepository 单元测试（内存 sqlite）。"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from lib.db.base import Base
from lib.db.repositories.provider_asset_repo import ProviderAssetRepository


@pytest.fixture()
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_get_missing_returns_none(factory):
    async with factory() as s:
        assert await ProviderAssetRepository(s).get("tecdo", "k1", "deadbeef") is None


async def test_upsert_then_get(factory):
    async with factory() as s:
        repo = ProviderAssetRepository(s)
        await repo.upsert(provider="tecdo", key_hash="k1", content_hash="h1", asset_id="a1", status="Active")
        await s.commit()
    async with factory() as s:
        row = await ProviderAssetRepository(s).get("tecdo", "k1", "h1")
        assert row is not None
        assert row.asset_id == "a1"
        assert row.status == "Active"
        assert row.asset_type == "Image"


async def test_upsert_updates_existing(factory):
    async with factory() as s:
        repo = ProviderAssetRepository(s)
        await repo.upsert(provider="tecdo", key_hash="k1", content_hash="h1", asset_id="a1", status="Processing")
        await s.commit()
    async with factory() as s:
        repo = ProviderAssetRepository(s)
        await repo.upsert(provider="tecdo", key_hash="k1", content_hash="h1", asset_id="a2", status="Active")
        await s.commit()
    async with factory() as s:
        row = await ProviderAssetRepository(s).get("tecdo", "k1", "h1")
        assert row is not None
        assert row.asset_id == "a2"
        assert row.status == "Active"


async def test_scoped_by_provider(factory):
    async with factory() as s:
        repo = ProviderAssetRepository(s)
        await repo.upsert(provider="tecdo", key_hash="k1", content_hash="h1", asset_id="a1", status="Active")
        await s.commit()
    async with factory() as s:
        assert await ProviderAssetRepository(s).get("other", "k1", "h1") is None


async def test_scoped_by_key_hash(factory):
    """同图不同密钥 → 互不命中(资产按密钥隔离)。"""
    async with factory() as s:
        repo = ProviderAssetRepository(s)
        await repo.upsert(provider="tecdo", key_hash="k1", content_hash="h1", asset_id="a1", status="Active")
        await s.commit()
    async with factory() as s:
        assert await ProviderAssetRepository(s).get("tecdo", "k2", "h1") is None
        row = await ProviderAssetRepository(s).get("tecdo", "k1", "h1")
        assert row is not None and row.asset_id == "a1"
