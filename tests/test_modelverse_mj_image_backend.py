"""ModelVerseMidjourneyImageBackend 单元测试（mock httpx，异步提交→轮询→U1 放大→下载）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.image_backends.base import ImageCapability, ImageCapabilityError, ImageGenerationRequest, ReferenceImage
from lib.image_backends.modelverse_mj import ModelVerseMidjourneyImageBackend

_U1_CUSTOM_ID = "MJ::JOB::upsample::1::431a5822-bfb2-4c55-8fc5-fc101abebd91"


def _submit_resp(task_id: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"output": {"task_id": task_id}, "request_id": "req-x"}
    return resp


def _status_resp(output: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"output": output}
    return resp


def _mock_client(*, post_side: list, get_side: list) -> AsyncMock:
    client = AsyncMock()
    client.post = AsyncMock(side_effect=post_side)
    client.get = AsyncMock(side_effect=get_side)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _request(tmp_path: Path, **kw) -> ImageGenerationRequest:
    return ImageGenerationRequest(prompt=kw.pop("prompt", "a cat"), output_path=tmp_path / "out.png", **kw)


def _backend() -> ModelVerseMidjourneyImageBackend:
    return ModelVerseMidjourneyImageBackend(api_key="sk-x", base_url="https://api.modelverse.cn")


def test_capabilities_is_t2i_only():
    assert _backend().capabilities == {ImageCapability.TEXT_TO_IMAGE}


async def test_imagine_then_upscale_returns_single_image(tmp_path: Path):
    """imagine 出四宫格 → 自动 U1 放大 → 返回单图 url；放大请求带上 imagine task_id + U1 custom_id。"""
    client = _mock_client(
        post_side=[_submit_resp("img-1"), _submit_resp("up-1")],
        get_side=[
            _status_resp(
                {
                    "task_id": "img-1",
                    "task_status": "Success",
                    "urls": ["https://x/grid.png"],
                    "buttons": [{"custom_id": _U1_CUSTOM_ID, "label": "U1"}],
                }
            ),
            _status_resp({"task_id": "up-1", "task_status": "Success", "urls": ["https://x/single.png"]}),
        ],
    )
    with (
        patch("lib.image_backends.modelverse_mj.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_mj.download_image_to_path", new=AsyncMock()) as dl,
    ):
        result = await _backend().generate(_request(tmp_path))

    assert result.image_uri == "https://x/single.png"
    assert result.provider == "modelverse"
    dl.assert_awaited_once()
    assert dl.await_args.args[0] == "https://x/single.png"

    # 第一次提交 = imagine，第二次 = upscale（带上一步 task_id + U1 custom_id）
    imagine_payload = client.post.await_args_list[0].kwargs["json"]
    assert imagine_payload["model"] == "midjourney-fast-imagine"
    upscale_payload = client.post.await_args_list[1].kwargs["json"]
    assert upscale_payload["model"] == "midjourney-fast-upscale"
    assert upscale_payload["parameters"] == {"mj_task_id": "img-1", "mj_custom_id": _U1_CUSTOM_ID}


async def test_no_upscale_button_falls_back_to_grid(tmp_path: Path):
    """imagine 完成但无放大按钮 → 不再二次提交，直接回退四宫格拼图。"""
    client = _mock_client(
        post_side=[_submit_resp("img-1")],
        get_side=[
            _status_resp({"task_id": "img-1", "task_status": "Success", "urls": ["https://x/grid.png"], "buttons": []})
        ],
    )
    with (
        patch("lib.image_backends.modelverse_mj.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_mj.download_image_to_path", new=AsyncMock()) as dl,
    ):
        result = await _backend().generate(_request(tmp_path))

    assert result.image_uri == "https://x/grid.png"
    assert client.post.await_count == 1  # 仅 imagine，无放大提交
    assert dl.await_args.args[0] == "https://x/grid.png"


async def test_aspect_ratio_appended_as_mj_flag(tmp_path: Path):
    """请求比例作为 Midjourney --ar 参数拼进 prompt。"""
    client = _mock_client(
        post_side=[_submit_resp("img-1")],
        get_side=[
            _status_resp({"task_id": "img-1", "task_status": "Success", "urls": ["https://x/grid.png"], "buttons": []})
        ],
    )
    with (
        patch("lib.image_backends.modelverse_mj.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_mj.download_image_to_path", new=AsyncMock()),
    ):
        await _backend().generate(_request(tmp_path, aspect_ratio="9:16"))

    assert "--ar 9:16" in client.post.await_args_list[0].kwargs["json"]["input"]["prompt"]


async def test_failure_status_raises(tmp_path: Path):
    client = _mock_client(
        post_side=[_submit_resp("img-1")],
        get_side=[_status_resp({"task_id": "img-1", "task_status": "Failure", "error_message": "blocked"})],
    )
    with (
        patch("lib.image_backends.modelverse_mj.httpx.AsyncClient", return_value=client),
        patch("lib.image_backends.modelverse_mj.download_image_to_path", new=AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="blocked"):
            await _backend().generate(_request(tmp_path))


async def test_reference_images_rejected(tmp_path: Path):
    """imagine 不支持参考图（I2I）→ 抛 ImageCapabilityError，不发任何请求。"""
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\nfake")
    req = _request(tmp_path, reference_images=[ReferenceImage(path=str(ref))])
    with pytest.raises(ImageCapabilityError):
        await _backend().generate(req)


def test_base_url_normalization_strips_v1():
    b = ModelVerseMidjourneyImageBackend(api_key="sk-x", base_url="https://api.modelverse.cn/v1/")
    assert b._base_url == "https://api.modelverse.cn"  # pyright: ignore[reportPrivateUsage]
