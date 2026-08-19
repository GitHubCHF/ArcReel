"""ModelVerseGrokImageBackend 单元测试（mock httpx，同步 OpenAI 风格 REST）。"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.image_backends.base import ImageCapability, ImageCapabilityError, ImageGenerationRequest, ReferenceImage
from lib.image_backends.modelverse_grok import ModelVerseGrokImageBackend


def _resp(data: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"created": 1, "data": data}
    return resp


def _mock_client(resp: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _request(tmp_path: Path, **kw) -> ImageGenerationRequest:
    return ImageGenerationRequest(prompt=kw.pop("prompt", "a cat"), output_path=tmp_path / "out.png", **kw)


def _backend() -> ModelVerseGrokImageBackend:
    return ModelVerseGrokImageBackend(api_key="sk-x", base_url="https://api.modelverse.cn")


def _make_ref(tmp_path: Path, name: str) -> ReferenceImage:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\nfake")
    return ReferenceImage(path=str(p))


def test_capabilities_t2i_and_i2i():
    assert _backend().capabilities == {ImageCapability.TEXT_TO_IMAGE, ImageCapability.IMAGE_TO_IMAGE}


async def test_t2i_posts_generations_and_downloads(tmp_path: Path):
    client = _mock_client(_resp([{"url": "https://x/out.png"}]))
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()) as dl,
    ):
        result = await _backend().generate(_request(tmp_path, aspect_ratio="9:16"))

    url, payload = client.post.await_args.args[0], client.post.await_args.kwargs["json"]
    assert url.endswith("/v1/images/generations")
    assert payload == {
        "model": "grok-imagine-image",
        "prompt": "a cat",
        "n": 1,
        "size": "2k",
        "aspect_ratio": "9:16",
        "response_format": "b64_json",
    }
    assert result.image_uri == "https://x/out.png"
    assert result.provider == "modelverse"
    # url 下载带浏览器 UA 头,规避 xAI imgen CDN 对 python-httpx 的 403
    dl.assert_awaited_once()
    assert dl.await_args.args[0] == "https://x/out.png"
    assert "User-Agent" in dl.await_args.kwargs["headers"]


async def test_size_maps_to_1k_when_requested(tmp_path: Path):
    client = _mock_client(_resp([{"url": "https://x/out.png"}]))
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()),
    ):
        await _backend().generate(_request(tmp_path, image_size="1K"))
    assert client.post.await_args.kwargs["json"]["size"] == "1k"


async def test_unsupported_aspect_ratio_falls_back_to_auto(tmp_path: Path):
    client = _mock_client(_resp([{"url": "https://x/out.png"}]))
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()),
    ):
        await _backend().generate(_request(tmp_path, aspect_ratio="7:11"))
    assert client.post.await_args.kwargs["json"]["aspect_ratio"] == "auto"


async def test_i2i_single_ref_uses_image_field(tmp_path: Path):
    client = _mock_client(_resp([{"url": "https://x/edited.png"}]))
    req = _request(tmp_path, reference_images=[_make_ref(tmp_path, "r.png")])
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()),
    ):
        await _backend().generate(req)
    url, payload = client.post.await_args.args[0], client.post.await_args.kwargs["json"]
    assert url.endswith("/v1/images/edits")
    assert payload["image"].startswith("data:image/")
    assert "images" not in payload


async def test_i2i_multi_refs_use_images_array(tmp_path: Path):
    client = _mock_client(_resp([{"url": "https://x/edited.png"}]))
    req = _request(tmp_path, reference_images=[_make_ref(tmp_path, "a.png"), _make_ref(tmp_path, "b.png")])
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()),
    ):
        await _backend().generate(req)
    payload = client.post.await_args.kwargs["json"]
    assert len(payload["images"]) == 2
    assert "image" not in payload


async def test_i2i_all_unreadable_raises(tmp_path: Path):
    req = _request(tmp_path, reference_images=[ReferenceImage(path=str(tmp_path / "missing.png"))])
    client = _mock_client(_resp([{"url": "https://x/out.png"}]))
    with patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client):
        with pytest.raises(ImageCapabilityError):
            await _backend().generate(req)


async def test_b64_json_response_saved(tmp_path: Path):
    b64 = base64.b64encode(b"fakebytes").decode()
    client = _mock_client(_resp([{"b64_json": b64, "mime_type": "image/jpeg"}]))
    out = tmp_path / "out.png"
    with patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client):
        result = await _backend().generate(ImageGenerationRequest(prompt="x", output_path=out))
    assert out.read_bytes() == b"fakebytes"
    assert result.image_uri is None


async def test_b64_preferred_over_url_when_both_present(tmp_path: Path):
    """同时返回 b64_json 与 url 时优先用 b64,绝不去 fetch 会 403 的 xAI CDN url。"""
    b64 = base64.b64encode(b"inlinebytes").decode()
    client = _mock_client(_resp([{"b64_json": b64, "url": "https://imgen.x.ai/blocked.png"}]))
    out = tmp_path / "out.png"
    with (
        patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_grok.download_image_to_path", new=AsyncMock()) as dl,
    ):
        await _backend().generate(ImageGenerationRequest(prompt="x", output_path=out))
    assert out.read_bytes() == b"inlinebytes"
    dl.assert_not_awaited()  # 没有触碰 url 下载


async def test_empty_data_raises(tmp_path: Path):
    client = _mock_client(_resp([]))
    with patch("lib.image_backends.modelverse_grok.httpx.AsyncClient", return_value=client):
        with pytest.raises(RuntimeError, match="data 为空"):
            await _backend().generate(_request(tmp_path))


def test_base_url_normalization_strips_v1():
    b = ModelVerseGrokImageBackend(api_key="sk-x", base_url="https://api.modelverse.cn/v1")
    assert b._base_url == "https://api.modelverse.cn"  # pyright: ignore[reportPrivateUsage]
