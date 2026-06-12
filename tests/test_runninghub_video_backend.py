"""RunningHubVideoBackend 单元测试（mock httpx）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lib.providers import PROVIDER_RUNNINGHUB
from lib.video_backends.base import (
    ResumeExpiredError,
    VideoCapability,
    VideoCapabilityError,
    VideoGenerationRequest,
)


def _make_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


def _make_http_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://www.runninghub.cn/openapi/v2/query")
    response = httpx.Response(status_code, request=request, text=message)
    return httpx.HTTPStatusError(f"error '{status_code}'", request=request, response=response)


def _upload_resp(download_url: str = "https://cos.example.com/img.png?sign=x") -> MagicMock:
    return _make_response(200, {"code": 0, "message": "success", "data": {"download_url": download_url}})


def _submit_resp(task_id: str = "task-1") -> MagicMock:
    return _make_response(200, {"taskId": task_id, "status": "RUNNING", "results": None})


def _query_resp(status: str, url: str | None = None) -> MagicMock:
    results = [{"url": url, "outputType": "mp4"}] if url else None
    return _make_response(200, {"taskId": "task-1", "status": status, "errorMessage": "", "results": results})


def _fake_download():
    async def _fake(url: str, output_path: Path, *, timeout: int = 120) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4-bytes")

    return AsyncMock(side_effect=_fake)


def _patches(mock_client, fake_download):
    return (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("lib.video_backends.runninghub._POLL_INTERVAL_SECONDS", 0.0),
        patch("lib.video_backends.runninghub.download_video", fake_download),
    )


def _img(tmp_path: Path, name: str = "a.png") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\nfake")
    return p


class TestMeta:
    def test_name_and_model(self):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        b = RunningHubVideoBackend(api_key="k", model="seedance-2.0")
        assert b.name == PROVIDER_RUNNINGHUB
        assert b.model == "seedance-2.0"

    def test_default_base_url(self):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        b = RunningHubVideoBackend(api_key="k")
        assert b._base_url == "https://www.runninghub.ai"

    def test_host_only_base_url_normalized(self):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        b = RunningHubVideoBackend(api_key="k", base_url="relay.example.com/")
        assert b._base_url == "https://relay.example.com"

    def test_capabilities(self):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        b = RunningHubVideoBackend(api_key="k")
        assert VideoCapability.TEXT_TO_VIDEO in b.capabilities
        assert VideoCapability.IMAGE_TO_VIDEO in b.capabilities
        assert VideoCapability.GENERATE_AUDIO in b.capabilities
        assert VideoCapability.SEED_CONTROL in b.capabilities

    def test_video_capabilities_fixed_limit(self):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        caps = RunningHubVideoBackend.video_capabilities_for_model("anything")
        assert caps.last_frame is True
        assert caps.reference_images is True
        assert caps.max_reference_images == 9


class TestEndpointDispatch:
    async def test_text_to_video_no_image(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[_submit_resp(), _query_resp("SUCCESS", "https://cdn/v.mp4")])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k", model="seedance-2.0")
            result = await b.generate(
                VideoGenerationRequest(
                    prompt="a cat",
                    output_path=tmp_path / "o.mp4",
                    aspect_ratio="9:16",
                    duration_seconds=5,
                )
            )

        # 第一次 POST 应命中 text-to-video 端点,且无图字段
        submit_call = mock_client.post.call_args_list[0]
        assert submit_call.args[0].endswith("/seedance-2.0-global/text-to-video")
        body = submit_call.kwargs["json"]
        assert body["prompt"] == "a cat"
        assert body["duration"] == "5"  # int → str
        assert body["ratio"] == "9:16"
        assert body["seed"] == -1  # None → -1
        assert "firstFrameUrl" not in body and "imageUrls" not in body
        assert result.provider == PROVIDER_RUNNINGHUB
        assert result.task_id == "task-1"
        assert result.video_path.read_bytes() == b"mp4-bytes"

    async def test_image_to_video_uploads_first_frame(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                _upload_resp("https://cos/first.png"),
                _submit_resp(),
                _query_resp("SUCCESS", "https://cdn/v.mp4"),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            await b.generate(
                VideoGenerationRequest(
                    prompt="p",
                    output_path=tmp_path / "o.mp4",
                    start_image=img,
                    aspect_ratio="9:16",
                    duration_seconds=6,
                )
            )

        # 第一次 POST 是上传(multipart files),第二次是 image-to-video 提交
        upload_call = mock_client.post.call_args_list[0]
        assert upload_call.args[0] == "https://www.runninghub.cn/openapi/v2/media/upload/binary"
        assert "files" in upload_call.kwargs
        submit_call = mock_client.post.call_args_list[1]
        assert submit_call.args[0].endswith("/seedance-2.0-global/image-to-video")
        assert submit_call.kwargs["json"]["firstFrameUrl"] == "https://cos/first.png"
        assert "lastFrameUrl" not in submit_call.kwargs["json"]

    async def test_image_to_video_with_last_frame(self, tmp_path: Path):
        first, last = _img(tmp_path, "first.png"), _img(tmp_path, "last.png")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                _upload_resp("https://cos/first.png"),
                _upload_resp("https://cos/last.png"),
                _submit_resp(),
                _query_resp("SUCCESS", "https://cdn/v.mp4"),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            await b.generate(
                VideoGenerationRequest(
                    prompt="p",
                    output_path=tmp_path / "o.mp4",
                    start_image=first,
                    end_image=last,
                    aspect_ratio="9:16",
                    duration_seconds=5,
                )
            )

        submit_call = mock_client.post.call_args_list[2]
        assert submit_call.args[0].endswith("/image-to-video")
        assert submit_call.kwargs["json"]["firstFrameUrl"] == "https://cos/first.png"
        assert submit_call.kwargs["json"]["lastFrameUrl"] == "https://cos/last.png"

    async def test_reference_to_video_multimodal(self, tmp_path: Path):
        refs = [_img(tmp_path, f"r{i}.png") for i in range(3)]
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                _upload_resp("https://cos/r0.png"),
                _upload_resp("https://cos/r1.png"),
                _upload_resp("https://cos/r2.png"),
                _submit_resp(),
                _query_resp("SUCCESS", "https://cdn/v.mp4"),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            await b.generate(
                VideoGenerationRequest(
                    prompt="p",
                    output_path=tmp_path / "o.mp4",
                    reference_images=refs,
                    aspect_ratio="9:16",
                    duration_seconds=5,
                )
            )

        submit_call = mock_client.post.call_args_list[3]
        assert submit_call.args[0].endswith("/multimodal-video")
        assert submit_call.kwargs["json"]["imageUrls"] == [
            "https://cos/r0.png",
            "https://cos/r1.png",
            "https://cos/r2.png",
        ]


class TestPollAndErrors:
    async def test_polls_through_running(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                _submit_resp(),
                _query_resp("RUNNING"),
                _query_resp("QUEUED"),
                _query_resp("SUCCESS", "https://cdn/v.mp4"),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            result = await b.generate(
                VideoGenerationRequest(
                    prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5
                )
            )

        assert result.task_id == "task-1"
        # 1 submit + 3 query
        assert mock_client.post.call_count == 4
        fake_download.assert_called_once()

    async def test_failed_status_raises(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                _submit_resp(),
                _make_response(200, {"taskId": "task-1", "status": "FAILED", "errorMessage": "upstream boom"}),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="upstream boom"):
                await b.generate(
                    VideoGenerationRequest(
                        prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5
                    )
                )
        fake_download.assert_not_called()

    async def test_duration_out_of_range_raises(self, tmp_path: Path):
        from lib.video_backends.runninghub import RunningHubVideoBackend

        b = RunningHubVideoBackend(api_key="k")
        with pytest.raises(VideoCapabilityError) as ei:
            await b.generate(
                VideoGenerationRequest(
                    prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=20
                )
            )
        assert ei.value.code == "video_duration_not_supported"

    async def test_upload_missing_image_raises(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            with pytest.raises(VideoCapabilityError) as ei:
                await b.generate(
                    VideoGenerationRequest(
                        prompt="p",
                        output_path=tmp_path / "o.mp4",
                        start_image=tmp_path / "missing.png",
                        aspect_ratio="9:16",
                        duration_seconds=5,
                    )
                )
            assert ei.value.code == "video_start_image_unreadable"

    async def test_upload_nonzero_code_raises(self, tmp_path: Path):
        img = _img(tmp_path)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_response(200, {"code": 401, "message": "bad key", "data": None})
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        p1, p2, p3 = _patches(mock_client, _fake_download())
        with p1, p2, p3, patch("lib.retry._compute_wait", lambda attempt, backoff: 0.0):
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            with pytest.raises(RuntimeError, match="文件上传失败"):
                await b.generate(
                    VideoGenerationRequest(
                        prompt="p",
                        output_path=tmp_path / "o.mp4",
                        start_image=img,
                        aspect_ratio="9:16",
                        duration_seconds=5,
                    )
                )

    async def test_submit_4xx_fails_fast(self, tmp_path: Path):
        bad = _make_response(400, {"error": "bad"})
        bad.raise_for_status = MagicMock(side_effect=_make_http_error(400, "bad request"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=bad)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.retry._compute_wait", lambda attempt, backoff: 0.0),
        ):
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            with pytest.raises(httpx.HTTPStatusError):
                await b.generate(
                    VideoGenerationRequest(
                        prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5
                    )
                )
        assert mock_client.post.call_count == 1


class TestResume:
    async def test_resume_polls_without_submit(self, tmp_path: Path):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_query_resp("SUCCESS", "https://cdn/resumed.mp4"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        fake_download = _fake_download()

        p1, p2, p3 = _patches(mock_client, fake_download)
        with p1, p2, p3:
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            result = await b.resume_video(
                "task-resume",
                VideoGenerationRequest(
                    prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5
                ),
            )

        # resume 只查询(POST /query),body 带 taskId,不重新提交生成端点
        assert mock_client.post.call_count == 1
        query_call = mock_client.post.call_args
        assert query_call.args[0].endswith("/openapi/v2/query")
        assert query_call.kwargs["json"] == {"taskId": "task-resume"}
        assert result.task_id == "task-resume"
        assert result.video_path.read_bytes() == b"mp4-bytes"

    async def test_resume_404_raises_expired_without_retry(self, tmp_path: Path):
        not_found = _make_response(404, {"error": "not found"})
        not_found.raise_for_status = MagicMock(side_effect=_make_http_error(404, "task not found"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=not_found)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("lib.video_backends.runninghub._POLL_INTERVAL_SECONDS", 0.0),
        ):
            from lib.video_backends.runninghub import RunningHubVideoBackend

            b = RunningHubVideoBackend(api_key="k")
            with pytest.raises(ResumeExpiredError) as ei:
                await b.resume_video(
                    "task-404",
                    VideoGenerationRequest(
                        prompt="p", output_path=tmp_path / "o.mp4", aspect_ratio="9:16", duration_seconds=5
                    ),
                )
            assert ei.value.job_id == "task-404"
            assert ei.value.provider == PROVIDER_RUNNINGHUB
            assert mock_client.post.call_count == 1
