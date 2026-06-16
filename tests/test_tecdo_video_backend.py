"""TecDoVideoBackend 单元测试（mock httpx）。"""

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

_BASE = "https://open-power.tec-do.cn"


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


def _make_http_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"{_BASE}/tecpower/ai/openapi/video/task")
    response = httpx.Response(status_code, request=request, text=message)
    return httpx.HTTPStatusError(f"error '{status_code}'", request=request, response=response)


def _upload_resp(s3_host: str = "https://cdn.tec-do.cn", url: str = "/img.png") -> MagicMock:
    return _make_response(200, {"code": "0", "message": "success", "data": {"s3Host": s3_host, "url": url}})


def _submit_resp(task_id: str = "task-1") -> MagicMock:
    return _make_response(200, {"code": 200, "data": {"taskId": task_id}})


def _query_resp(status: str, url: str | None = None, actual_amount: float | None = None) -> MagicMock:
    data: dict = {"taskId": "task-1", "status": status, "error": ""}
    if url is not None:
        data["videoUrl"] = url
    if actual_amount is not None:
        data["actualAmount"] = actual_amount
    return _make_response(200, {"code": 200, "data": data})


def _fake_download():
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4-bytes")

    return AsyncMock(side_effect=_fake)


def _patches(mock_client, fake_download):
    return (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("lib.video_backends.tecdo._POLL_INTERVAL_SECONDS", 0.0),
        patch("lib.video_backends.tecdo.download_video", fake_download),
    )


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

        b = TecDoVideoBackend(api_key="k", model="seedance2.0")
        assert b.name == PROVIDER_TECDO
        assert b.model == "seedance2.0"

    def test_default_model_and_base_url(self):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")
        assert b.model == "seedance2.0"
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
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")])
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k", model="seedance2.0")
            result = await b.generate(_req(tmp_path, prompt="a cat"))

        submit_call = mock_client.post.call_args_list[0]
        assert submit_call.args[0] == f"{_BASE}/tecpower/ai/openapi/video/create"
        body = submit_call.kwargs["json"]
        assert body["model"] == "seedance2.0"
        assert body["duration"] == 5  # int, 不转 str
        assert body["ratio"] == "9:16"
        assert body["watermark"] is False
        assert "seed" not in body  # None → 不下传
        assert body["content"] == [{"type": "text", "text": "a cat"}]
        assert result.provider == PROVIDER_TECDO
        assert result.task_id == "task-1"
        assert result.video_path.read_bytes() == b"mp4-bytes"

    async def test_seed_forwarded_when_set(self, tmp_path: Path):
        mock_client = _client(post_side=[_submit_resp()], get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")])
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, seed=42))

        assert mock_client.post.call_args_list[0].kwargs["json"]["seed"] == 42

    async def test_image_to_video_uploads_first_frame(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = _client(
            post_side=[_upload_resp("https://cdn.tec-do.cn", "/first.png"), _submit_resp()],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, start_image=img))

        upload_call = mock_client.post.call_args_list[0]
        assert upload_call.args[0] == f"{_BASE}/uni-agency/openApi/v1/mediaAccountApplication/upload/file"
        assert upload_call.kwargs["data"] == {"type": "FILE"}
        assert "files" in upload_call.kwargs
        assert upload_call.kwargs["headers"]["X-App-Secret"] == "k"

        content = mock_client.post.call_args_list[1].kwargs["json"]["content"]
        assert content[0] == {
            "type": "image_url",
            "imageUrl": {"url": "https://cdn.tec-do.cn/first.png"},
            "role": "first_frame",
        }
        assert content[-1] == {"type": "text", "text": "p"}
        assert not any(c.get("role") == "last_frame" for c in content)

    async def test_image_to_video_with_last_frame(self, tmp_path: Path):
        first, last = _img(tmp_path, "first.png"), _img(tmp_path, "last.png")
        mock_client = _client(
            post_side=[
                _upload_resp("https://cdn.tec-do.cn", "/first.png"),
                _upload_resp("https://cdn.tec-do.cn", "/last.png"),
                _submit_resp(),
            ],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, start_image=first, end_image=last))

        content = mock_client.post.call_args_list[2].kwargs["json"]["content"]
        roles = [(c.get("role"), c.get("imageUrl", {}).get("url")) for c in content if c["type"] == "image_url"]
        assert roles == [
            ("first_frame", "https://cdn.tec-do.cn/first.png"),
            ("last_frame", "https://cdn.tec-do.cn/last.png"),
        ]

    async def test_reference_to_video(self, tmp_path: Path):
        refs = [_img(tmp_path, f"r{i}.png") for i in range(3)]
        mock_client = _client(
            post_side=[
                _upload_resp("https://cdn.tec-do.cn", "/r0.png"),
                _upload_resp("https://cdn.tec-do.cn", "/r1.png"),
                _upload_resp("https://cdn.tec-do.cn", "/r2.png"),
                _submit_resp(),
            ],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, reference_images=refs))

        content = mock_client.post.call_args_list[3].kwargs["json"]["content"]
        ref_urls = [c["imageUrl"]["url"] for c in content if c.get("role") == "reference_image"]
        assert ref_urls == [
            "https://cdn.tec-do.cn/r0.png",
            "https://cdn.tec-do.cn/r1.png",
            "https://cdn.tec-do.cn/r2.png",
        ]
        assert content[-1] == {"type": "text", "text": "p"}

    async def test_upload_absolute_url_not_prefixed(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = _client(
            post_side=[_upload_resp("https://cdn.tec-do.cn", "https://abs.example.com/x.png"), _submit_resp()],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            await b.generate(_req(tmp_path, start_image=img))

        content = mock_client.post.call_args_list[1].kwargs["json"]["content"]
        assert content[0]["imageUrl"]["url"] == "https://abs.example.com/x.png"


class TestPollAndErrors:
    async def test_polls_through_processing(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[
                _query_resp("PENDING"),
                _query_resp("PROCESSING"),
                _query_resp("COMPLETED", "https://cdn/v.mp4"),
            ],
        )
        fake_download = _fake_download()
        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.task_id == "task-1"
        assert mock_client.post.call_count == 1
        assert mock_client.get.call_count == 3
        fake_download.assert_called_once()
        query_call = mock_client.get.call_args
        assert query_call.args[0] == f"{_BASE}/tecpower/ai/openapi/video/task"
        assert query_call.kwargs["params"] == {"taskId": "task-1"}

    async def test_actual_amount_recorded(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4", actual_amount=1.23)],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.actual_cost == 1.23
        assert result.actual_currency == "CNY"

    async def test_no_actual_amount_leaves_cost_none(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_query_resp("COMPLETED", "https://cdn/v.mp4")],
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.generate(_req(tmp_path))

        assert result.actual_cost is None
        assert result.actual_currency is None

    async def test_failed_status_raises(self, tmp_path: Path):
        mock_client = _client(
            post_side=[_submit_resp()],
            get_side=[_make_response(200, {"data": {"status": "FAILED", "error": "upstream boom"}})],
        )
        fake_download = _fake_download()
        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="upstream boom"):
                await b.generate(_req(tmp_path))
        fake_download.assert_not_called()

    async def test_duration_out_of_range_raises(self, tmp_path: Path):
        from lib.video_backends.tecdo import TecDoVideoBackend

        b = TecDoVideoBackend(api_key="k")
        with pytest.raises(VideoCapabilityError) as ei:
            await b.generate(_req(tmp_path, duration_seconds=20))
        assert ei.value.code == "video_duration_not_supported"

    async def test_upload_missing_image_raises(self, tmp_path: Path):
        mock_client = _client(post_side=[], get_side=[])
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(VideoCapabilityError) as ei:
                await b.generate(_req(tmp_path, start_image=tmp_path / "missing.png"))
            assert ei.value.code == "video_start_image_unreadable"

    async def test_upload_nonzero_code_raises(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = _client()
        mock_client.post = AsyncMock(
            return_value=_make_response(200, {"code": "401", "message": "bad key", "data": None})
        )
        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3, patch("lib.retry._compute_wait", lambda attempt, backoff: 0.0):
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="文件上传失败"):
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
        mock_client = _client(get_side=[_query_resp("COMPLETED", "https://cdn/resumed.mp4")])
        fake_download = _fake_download()
        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.tecdo import TecDoVideoBackend

            b = TecDoVideoBackend(api_key="k")
            result = await b.resume_video("task-resume", _req(tmp_path))

        assert mock_client.post.call_count == 0
        assert mock_client.get.call_count == 1
        query_call = mock_client.get.call_args
        assert query_call.kwargs["params"] == {"taskId": "task-resume"}
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
