"""TecDoVideoBackend（钛极模型网关协议）单元测试（mock httpx + OSS uploader）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lib.providers import PROVIDER_TECDO
from lib.video_backends.base import (
    ResumeExpiredError,
    VideoCapability,
    VideoCapabilityError,
    VideoGenerationRequest,
)

_BASE = "https://api.tcgateway.top"
_TASKS_URL = f"{_BASE}/api/v3/contents/generations/tasks"
_ARK_URL = f"{_BASE}/api/ark"


def _oss_cfg() -> dict[str, str]:
    return {
        "endpoint": "oss-cn-hangzhou.aliyuncs.com",
        "bucket": "bkt",
        "access_key_id": "id",
        "access_key_secret": "sec",
        "upload_prefix": "arcreel",
    }


class _FakeUploader:
    """替身 OSSUploader：upload_file 按文件名返回稳定 URL，不触网。"""

    def __init__(self, config, **_kw):  # noqa: ANN001
        self.config = config

    def upload_file(self, path: Path, *, key: str | None = None) -> str:
        return f"https://oss.example.com/{path.name}"


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


def _make_http_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"{_TASKS_URL}/task-x")
    response = httpx.Response(status_code, request=request, text=message)
    return httpx.HTTPStatusError(f"error '{status_code}'", request=request, response=response)


def _submit_resp(task_id: str = "task-1") -> MagicMock:
    return _make_response(200, {"id": task_id, "model": "Seedance2.0", "status": "queued", "content": {}})


def _query_resp(
    status: str,
    url: str | None = None,
    total_tokens: int | None = None,
    error: str | dict | None = None,
) -> MagicMock:
    body: dict = {"id": "task-1", "model": "Seedance2.0", "status": status}
    if url is not None:
        body["content"] = {"video_url": url}
    if total_tokens is not None:
        body["usage"] = {"total_tokens": total_tokens}
    if error is not None:
        body["error"] = error
    return _make_response(200, body)


def _group_create_resp(group_id: str = "group-1") -> MagicMock:
    return _make_response(200, {"ResponseMetadata": {"Action": "CreateAssetGroup"}, "Result": {"Id": group_id}})


def _asset_create_resp(asset_id: str = "asset-1") -> MagicMock:
    return _make_response(
        200,
        {"ResponseMetadata": {"Action": "CreateAsset"}, "Result": {"Id": asset_id, "Status": "Pending"}},
    )


def _asset_get_resp(status: str = "Active", asset_id: str = "asset-1") -> MagicMock:
    return _make_response(
        200,
        {"ResponseMetadata": {"Action": "GetAsset"}, "Result": {"Id": asset_id, "Status": status}},
    )


def _fake_download():
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4-bytes")

    return AsyncMock(side_effect=_fake)


def _patches(mock_client, fake_download, *, with_oss: bool = False):
    base = [
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
        patch("lib.video_backends.tecdo._ASSET_POLL_INTERVAL_SECONDS", 0.0),
        patch("lib.video_backends.tecdo.download_video", fake_download),
    ]
    if with_oss:
        base.append(patch("lib.video_backends.tecdo.OSSUploader", _FakeUploader))
    return base


async def _memory_session_factory():
    """自带内存 sqlite session_factory(建表),用于资产缓存命中测试。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from lib.db.base import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False), engine


def _client(*, post_side=None, get_side=None) -> AsyncMock:
    c = AsyncMock()
    c.post = AsyncMock(side_effect=post_side) if post_side is not None else AsyncMock()
    c.get = AsyncMock(side_effect=get_side) if get_side is not None else AsyncMock()
    c.__aenter__ = AsyncMock(return_value=c)
    c.__aexit__ = AsyncMock(return_value=None)
    return c


def _img(tmp_path: Path, name: str = "a.png") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\nfake")
    return p


def _req(tmp_path: Path, **kw) -> VideoGenerationRequest:
    base = dict(prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5)
    base.update(kw)
    return VideoGenerationRequest(**base)  # type: ignore[arg-type]


class TestMeta:
    def test_name_and_model(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k", model="Seedance2.0")
        assert b.name == PROVIDER_TECDO
        assert b.model == "Seedance2.0"

    def test_default_model_and_base_url(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")
        assert b.model == "Seedance2.0"
        assert b._base_url == _BASE

    def test_host_only_base_url_normalized(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k", base_url="relay.example.com/")
        assert b._base_url == "https://relay.example.com"

    def test_capabilities(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")
        assert VideoCapability.TEXT_TO_VIDEO in b.capabilities
        assert VideoCapability.IMAGE_TO_VIDEO in b.capabilities
        assert VideoCapability.GENERATE_AUDIO in b.capabilities
        assert VideoCapability.SEED_CONTROL in b.capabilities

    def test_video_capabilities_fixed_limit(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        caps = TecDoVideoBackend.video_capabilities_for_model("anything")
        assert caps.last_frame is True
        assert caps.reference_images is True
        assert caps.max_reference_images == 9


class TestContentDispatch:
    async def test_text_to_video_no_image(self, tmp_path: Path):
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("succeeded", "https://cdn/v.mp4")])
        fake_download = _fake_download()

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", fake_download),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", model="Seedance2.0")
            result = await b.generate(_req(tmp_path, prompt="a cat"))

        submit_call = mock_client.post.call_args_list[0]
        assert submit_call.args[0] == _TASKS_URL
        assert submit_call.kwargs["headers"]["Authorization"] == "Bearer k"
        body = submit_call.kwargs["json"]
        assert body["model"] == "Seedance2.0"
        assert body["duration"] == 5  # int, 不转 str
        assert body["ratio"] == "9:16"
        assert body["watermark"] is False
        assert "seed" not in body  # None → 不下传
        assert body["content"] == [{"type": "text", "text": "a cat"}]
        assert result.provider == PROVIDER_TECDO
        assert result.task_id == "task-1"
        assert result.video_path.read_bytes() == b"mp4-bytes"

    async def test_seed_forwarded_when_set(self, tmp_path: Path):
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("succeeded", "https://cdn/v.mp4")])
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", _fake_download()),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, seed=42))

        assert mock_client.post.call_args_list[0].kwargs["json"]["seed"] == 42

    async def test_image_to_video_uploads_first_frame(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("succeeded", "https://cdn/v.mp4")])
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg())
            await b.generate(_req(tmp_path, start_image=img))
        finally:
            patch.stopall()

        content = mock_client.post.call_args_list[0].kwargs["json"]["content"]
        assert content[0] == {"type": "text", "text": "p"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://oss.example.com/a.png"},
            "role": "first_frame",
        }
        assert not any(c.get("role") == "last_frame" for c in content)

    async def test_image_to_video_with_last_frame(self, tmp_path: Path):
        first, last = _img(tmp_path, "first.png"), _img(tmp_path, "last.png")
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("succeeded", "https://cdn/v.mp4")])
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg())
            await b.generate(_req(tmp_path, start_image=first, end_image=last))
        finally:
            patch.stopall()

        content = mock_client.post.call_args_list[0].kwargs["json"]["content"]
        roles = [(c.get("role"), c.get("image_url", {}).get("url")) for c in content if c["type"] == "image_url"]
        assert roles == [
            ("first_frame", "https://oss.example.com/first.png"),
            ("last_frame", "https://oss.example.com/last.png"),
        ]

    async def test_reference_goes_through_asset_library(self, tmp_path: Path):
        """参考图(角色集)→ 建组 → OSS → CreateAsset → 轮询 Active → content 用 asset://{id}。"""
        refs = [_img(tmp_path, f"r{i}.png") for i in range(2)]
        mock_client = _client(
            post_side=[
                _group_create_resp("group-1"),  # 首次需要先建资产组
                _asset_create_resp("asset-r0"),
                _asset_get_resp("Active", "asset-r0"),
                _asset_create_resp("asset-r1"),
                _asset_get_resp("Active", "asset-r1"),
                _submit_resp(),  # 视频 create 在所有资产就绪后
            ],
            get_side=[_query_resp("succeeded", "https://cdn/v.mp4")],
        )
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg())  # 无 session_factory → 不缓存
            await b.generate(_req(tmp_path, reference_images=refs))
        finally:
            patch.stopall()

        # 资产组 create → Ark Action 端点
        group_call = mock_client.post.call_args_list[0]
        assert group_call.args[0] == _ARK_URL
        assert group_call.kwargs["params"] == {"Action": "CreateAssetGroup", "Version": "2024-01-01"}
        # 资产 create 带 GroupId
        asset_call = mock_client.post.call_args_list[1]
        assert asset_call.kwargs["params"]["Action"] == "CreateAsset"
        assert asset_call.kwargs["json"]["GroupId"] == "group-1"
        assert asset_call.kwargs["json"]["AssetType"] == "Image"
        # 最后一个 post 是视频 create，content 用 asset://
        video_call = mock_client.post.call_args_list[-1]
        assert video_call.args[0] == _TASKS_URL
        content = video_call.kwargs["json"]["content"]
        ref_urls = [c["image_url"]["url"] for c in content if c.get("role") == "reference_image"]
        assert ref_urls == ["asset://asset-r0", "asset://asset-r1"]
        assert content[0] == {"type": "text", "text": "p"}

    async def test_reference_asset_failed_raises(self, tmp_path: Path):
        ref = _img(tmp_path, "r.png")
        mock_client = _client(
            post_side=[_group_create_resp(), _asset_create_resp("asset-x"), _asset_get_resp("Failed", "asset-x")],
            get_side=[],
        )
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg())
            with pytest.raises(RuntimeError, match="资产审核失败"):
                await b.generate(_req(tmp_path, reference_images=[ref]))
        finally:
            patch.stopall()

    async def test_reference_asset_cache_hit_skips_create(self, tmp_path: Path):
        """缓存命中(Active)→ 跳过建组/CreateAsset，直接用 asset://{cached}。"""
        import hashlib

        from lib.db.repositories.provider_asset_repo import ProviderAssetRepository
        from lib.providers import PROVIDER_TECDO
        from lib.video_backends.tecdo import _sha256_file

        ref = _img(tmp_path, "r.png")
        content_hash = _sha256_file(ref)
        key_hash = hashlib.sha256(b"k").hexdigest()  # 与 backend(api_key="k") 一致
        factory, engine = await _memory_session_factory()
        async with factory() as s:
            await ProviderAssetRepository(s).upsert(
                provider=PROVIDER_TECDO,
                key_hash=key_hash,
                content_hash=content_hash,
                asset_id="cached-1",
                status="Active",
            )
            await s.commit()

        # 命中缓存后不应有任何资产组/资产调用，只剩视频 create + 查询
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("succeeded", "https://cdn/v.mp4")])
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg(), session_factory=factory)
            await b.generate(_req(tmp_path, reference_images=[ref]))
        finally:
            patch.stopall()
            await engine.dispose()

        # 只有 1 次 post（视频 create），没有资产相关调用
        assert mock_client.post.call_count == 1
        content = mock_client.post.call_args_list[0].kwargs["json"]["content"]
        ref_urls = [c["image_url"]["url"] for c in content if c.get("role") == "reference_image"]
        assert ref_urls == ["asset://cached-1"]

    async def test_reference_asset_cache_miss_on_different_key(self, tmp_path: Path):
        """缓存行属于另一个密钥 → 不命中,重新登记(资产按密钥隔离)。"""
        from lib.db.repositories.provider_asset_repo import ProviderAssetRepository
        from lib.providers import PROVIDER_TECDO
        from lib.video_backends.tecdo import _sha256_file

        ref = _img(tmp_path, "r.png")
        content_hash = _sha256_file(ref)
        factory, engine = await _memory_session_factory()
        async with factory() as s:
            await ProviderAssetRepository(s).upsert(
                provider=PROVIDER_TECDO,
                key_hash="other-key-hash",  # 非当前 api_key 的指纹
                content_hash=content_hash,
                asset_id="cached-other",
                status="Active",
            )
            await s.commit()

        mock_client = _client(
            post_side=[
                _group_create_resp(),
                _asset_create_resp("asset-new"),
                _asset_get_resp("Active", "asset-new"),
                _submit_resp(),
            ],
            get_side=[_query_resp("succeeded", "https://cdn/v.mp4")],
        )
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg(), session_factory=factory)
            await b.generate(_req(tmp_path, reference_images=[ref]))
        finally:
            patch.stopall()
            await engine.dispose()

        # 未命中 → 走了建组 + CreateAsset,最终 content 用新登记的 asset
        assert mock_client.post.call_args_list[0].kwargs["params"]["Action"] == "CreateAssetGroup"
        assert mock_client.post.call_args_list[1].kwargs["params"]["Action"] == "CreateAsset"
        content = mock_client.post.call_args_list[-1].kwargs["json"]["content"]
        ref_urls = [c["image_url"]["url"] for c in content if c.get("role") == "reference_image"]
        assert ref_urls == ["asset://asset-new"]

    async def test_group_id_persisted_and_reused(self, tmp_path: Path):
        """资产组 GroupId 经 DB 哨兵行持久化:第二个 backend 实例不再重复建组。"""
        from lib.video_backends.tecdo import _ASSET_GROUP_SENTINEL_HASH, TecDoVideoBackend

        ref = _img(tmp_path, "r.png")
        factory, engine = await _memory_session_factory()

        mock_client = _client(
            post_side=[
                _group_create_resp("group-1"),
                _asset_create_resp("asset-a"),
                _asset_get_resp("Active", "asset-a"),
                _submit_resp(),
            ],
            get_side=[_query_resp("succeeded", "https://cdn/v.mp4")],
        )
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg(), session_factory=factory)
            await b.generate(_req(tmp_path, reference_images=[ref]))

            # 第二个实例 + 新图片(内容不同,避免命中资产缓存):组走 DB 缓存,
            # 只有 CreateAsset/GetAsset/视频 create
            ref2 = tmp_path / "r2.png"
            ref2.write_bytes(b"\x89PNG\r\nother-content")
            mock_client2 = _client(
                post_side=[
                    _asset_create_resp("asset-b"),
                    _asset_get_resp("Active", "asset-b"),
                    _submit_resp("task-2"),
                ],
                get_side=[_query_resp("succeeded", "https://cdn/v2.mp4")],
            )
            with patch("httpx.AsyncClient", return_value=mock_client2):
                b2 = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg(), session_factory=factory)
                await b2.generate(_req(tmp_path, reference_images=[ref2]))

            actions = [c.kwargs.get("params", {}).get("Action") for c in mock_client2.post.call_args_list]
            assert "CreateAssetGroup" not in actions
            assert mock_client2.post.call_args_list[0].kwargs["json"]["GroupId"] == "group-1"

            # 哨兵行确实落库
            cached = await b2._get_cached_asset(_ASSET_GROUP_SENTINEL_HASH)
            assert cached == "group-1"
        finally:
            patch.stopall()
            await engine.dispose()

    async def test_recreate_mode_bypasses_cache_hit(self, tmp_path: Path):
        """recreate 模式:即便缓存命中(Active)也重新登记资产、不读不写资产缓存。"""
        import hashlib

        from lib.db.repositories.provider_asset_repo import ProviderAssetRepository
        from lib.providers import PROVIDER_TECDO
        from lib.video_backends.tecdo import ASSET_CACHE_MODE_RECREATE, TecDoVideoBackend, _sha256_file

        ref = _img(tmp_path, "r.png")
        content_hash = _sha256_file(ref)
        key_hash = hashlib.sha256(b"k").hexdigest()
        factory, engine = await _memory_session_factory()
        async with factory() as s:
            await ProviderAssetRepository(s).upsert(
                provider=PROVIDER_TECDO,
                key_hash=key_hash,
                content_hash=content_hash,
                asset_id="cached-1",
                status="Active",
            )
            await s.commit()

        mock_client = _client(
            post_side=[
                _group_create_resp(),
                _asset_create_resp("asset-fresh"),
                _asset_get_resp("Active", "asset-fresh"),
                _submit_resp(),
            ],
            get_side=[_query_resp("succeeded", "https://cdn/v.mp4")],
        )
        for p in _patches(mock_client, _fake_download(), with_oss=True):
            p.start()
        try:
            b = TecDoVideoBackend(
                api_key="k",
                oss_config=_oss_cfg(),
                session_factory=factory,
                asset_cache_mode=ASSET_CACHE_MODE_RECREATE,
            )
            await b.generate(_req(tmp_path, reference_images=[ref]))

            # 重新登记了资产,content 用新 asset 而非缓存的 cached-1
            assert mock_client.post.call_args_list[1].kwargs["params"]["Action"] == "CreateAsset"
            content = mock_client.post.call_args_list[-1].kwargs["json"]["content"]
            ref_urls = [c["image_url"]["url"] for c in content if c.get("role") == "reference_image"]
            assert ref_urls == ["asset://asset-fresh"]

            # 且未写资产缓存:原缓存行仍是 cached-1（未被覆盖为 asset-fresh）
            async with factory() as s:
                row = await ProviderAssetRepository(s).get(PROVIDER_TECDO, key_hash, content_hash)
                assert row is not None and row.asset_id == "cached-1"
        finally:
            patch.stopall()
            await engine.dispose()


class TestPollAndErrors:
    async def test_polls_through_running(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[
                _query_resp("queued"),
                _query_resp("running"),
                _query_resp("succeeded", "https://cdn/v.mp4"),
            ],
        )
        fake_download = _fake_download()
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", fake_download),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.task_id == "task-1"
        assert mock_client.post.call_count == 1
        assert mock_client.get.call_count == 3
        fake_download.assert_called_once()
        query_call = mock_client.get.call_args
        assert query_call.args[0] == f"{_TASKS_URL}/task-1"

    async def test_compat_completed_status_is_done(self, tmp_path: Path):
        """网关兼容态 completed/success 也视为成功。"""
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("completed", "https://cdn/v.mp4")])
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", _fake_download()),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.task_id == "task-1"

    async def test_cost_fields_left_none(self, tmp_path: Path):
        """网关只回报 usage.total_tokens(平台侧结算),无金额可透传 → actual_cost 为 None。"""
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_query_resp("succeeded", "https://cdn/v.mp4", total_tokens=87726)],
        )
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", _fake_download()),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.actual_cost is None
        assert result.actual_currency is None

    async def test_failed_status_raises_with_error_message(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_query_resp("failed", error={"message": "upstream boom"})],
        )
        fake_download = _fake_download()
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", fake_download),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="upstream boom"):
                await b.generate(_req(tmp_path))
        fake_download.assert_not_called()

    async def test_failed_status_with_string_error(self, tmp_path: Path):
        """error 为字符串形态(非 {message}) 也能提取。"""
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_query_resp("error", error="quota exceeded")],
        )
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", _fake_download()),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="quota exceeded"):
                await b.generate(_req(tmp_path))

    async def test_duration_out_of_range_raises(self, tmp_path: Path):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")
        with pytest.raises(VideoCapabilityError) as ei:
            await b.generate(_req(tmp_path, duration_seconds=20))
        assert ei.value.code == "video_duration_not_supported"

    async def test_missing_image_raises(self, tmp_path: Path):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k", oss_config=_oss_cfg())
        with pytest.raises(VideoCapabilityError) as ei:
            await b.generate(_req(tmp_path, start_image=tmp_path / "missing.png"))
        assert ei.value.code == "video_start_image_unreadable"

    async def test_oss_not_configured_raises(self, tmp_path: Path):
        img = _img(tmp_path)
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")  # 无 oss_config
        with pytest.raises(RuntimeError, match="阿里云 OSS"):
            await b.generate(_req(tmp_path, start_image=img))

    async def test_submit_4xx_fails_fast(self, tmp_path: Path):
        bad = _make_response(400, {"error": "bad"})
        bad.raise_for_status = MagicMock(side_effect=_make_http_error(400, "bad request"))
        mock_client = _client()
        mock_client.post = AsyncMock(return_value=bad)
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.retry._compute_wait", lambda attempt, backoff: 0.0),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(httpx.HTTPStatusError):
                await b.generate(_req(tmp_path))
        assert mock_client.post.call_count == 1


class TestResume:
    async def test_resume_polls_without_create(self, tmp_path: Path):
        mock_client = _client(get_side=[_query_resp("succeeded", "https://cdn/resumed.mp4")])
        fake_download = _fake_download()
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
            patch("lib.video_backends.tecdo.download_video", fake_download),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.resume_video("task-resume", _req(tmp_path))

        assert mock_client.post.call_count == 0
        assert mock_client.get.call_count == 1
        query_call = mock_client.get.call_args
        assert query_call.args[0] == f"{_TASKS_URL}/task-resume"
        assert result.task_id == "task-resume"
        assert result.video_path.read_bytes() == b"mp4-bytes"

    async def test_resume_404_raises_expired_without_retry(self, tmp_path: Path):
        not_found = _make_response(404, {"error": "not found"})
        not_found.raise_for_status = MagicMock(side_effect=_make_http_error(404, "task not found"))
        mock_client = _client()
        mock_client.get = AsyncMock(return_value=not_found)
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
        ):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(ResumeExpiredError) as ei:
                await b.resume_video("task-404", _req(tmp_path))
            assert ei.value.job_id == "task-404"
            assert ei.value.provider == PROVIDER_TECDO
            assert mock_client.get.call_count == 1
