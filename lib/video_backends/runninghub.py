"""RunningHubVideoBackend — RunningHub(runninghub.cn) seedance 2.0 视频生成后端。

第三方中转站,与火山原生 ark seedance 协议完全不同:
- 鉴权 ``Authorization: Bearer {api_key}``,base url ``https://www.runninghub.cn``
- 三个不同 URL 端点按输入派发(请求体无 model 字段):
  - 无图 → ``/openapi/v2/rhart-video/sparkvideo-2.0/text-to-video``
  - 有首帧(可选尾帧)→ ``.../image-to-video``(firstFrameUrl / lastFrameUrl)
  - 有参考图 → ``.../multimodal-video``(imageUrls ≤9)
- 图片是 URL:本地图先 multipart 上传 ``/openapi/v2/media/upload/binary`` 取 download_url
  (COS 签名 URL,仅 1 天有效),再填进对应字段
- duration 为 string 枚举("4"~"15");seed=-1 表随机
- 异步:submit 返回 taskId → 轮询 ``POST /openapi/v2/query`` 至 SUCCESS → results[0].url 下载

用户在 UI 填的 model_id 仅用于记账/显示,不下传。
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from lib.logging_utils import format_kwargs_for_log
from lib.providers import PROVIDER_RUNNINGHUB
from lib.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import (
    IMAGE_MIME_TYPES,
    ResumeExpiredError,
    VideoCapabilities,
    VideoCapability,
    VideoCapabilityError,
    VideoGenerationRequest,
    VideoGenerationResult,
    download_video,
    persist_provider_job_id,
    poll_with_retry,
    should_retry_poll,
    should_retry_submit,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "seedance-2.0"
# 生成 + 查询走海外站 .ai(用户可在 UI 覆盖 base_url)
DEFAULT_BASE_URL = "https://www.runninghub.ai"
# 上传接口固定走 .cn 站(与生成/查询不同域名,不跟随 base_url);返回的 COS URL 两站通用
_UPLOAD_BASE_URL = "https://www.runninghub.cn"

_UPLOAD_PATH = "/openapi/v2/media/upload/binary"
_QUERY_PATH = "/openapi/v2/query"
_TEXT_TO_VIDEO_PATH = "/openapi/v2/bytedance/seedance-2.0-global/text-to-video"
_IMAGE_TO_VIDEO_PATH = "/openapi/v2/bytedance/seedance-2.0-global/image-to-video"
_MULTIMODAL_VIDEO_PATH = "/openapi/v2/bytedance/seedance-2.0-global/multimodal-video"

_POLL_INTERVAL_SECONDS = 10.0
_MIN_POLL_TIMEOUT_SECONDS = 600
_POLL_TIMEOUT_PER_SECOND = 60

# duration 合法值域(秒);上游枚举为 "4"~"15"(另含 "-1" 自适应,本项目不用)
_MIN_DURATION = 4
_MAX_DURATION = 15

# 参考图固定上限(multimodal 端点 imageUrls maxItems=9)
_MAX_REFERENCE_IMAGES = 9


def _normalize_base_url(base_url: str | None) -> str:
    """容忍用户填到 host(或带尾斜杠),归一化为不带尾斜杠的 base。空值回落官方域名。"""
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_BASE_URL
    return s if "://" in s else f"https://{s}"


class RunningHubVideoBackend:
    """RunningHub seedance 2.0 视频后端(异步上传→提交→轮询→下载)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("RunningHubVideoBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
            VideoCapability.GENERATE_AUDIO,
            VideoCapability.SEED_CONTROL,
        }

    @property
    def name(self) -> str:
        return PROVIDER_RUNNINGHUB

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @staticmethod
    def video_capabilities_for_model(model: str) -> VideoCapabilities:
        """纯计算参考图等 caps —— 不构造 client(无需 api_key)。

        seedance 2.0 通过 URL 区分端点,所有 model 共享同一套能力(首尾帧 + 参考图 ≤9),
        故恒返回固定上限。instance property 委托至此,保持单一真相源。
        """
        return VideoCapabilities(last_frame=True, reference_images=True, max_reference_images=_MAX_REFERENCE_IMAGES)

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return self.video_capabilities_for_model(self._model)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        duration = self._validate_duration(request.duration_seconds)
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            path, payload = await self._build_request(client, request, duration)
            logger.info(
                "调用 %s 视频 API path=%s body=%s",
                self.name,
                path,
                format_kwargs_for_log(payload),
            )
            task_id = await self._create_task(client, path, payload)
            logger.info("RunningHub 视频任务已创建: task_id=%s model=%s", task_id, self._model)
            if request.task_id is not None:
                await persist_provider_job_id(request.task_id, task_id, provider=PROVIDER_RUNNINGHUB)
            return await self._poll_and_build(client, task_id, request, is_resume=False)

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        """接续已 submit 的 RunningHub task：仅 poll + 下载(ADR 0007)。

        上传 URL 仅 1 天有效,超期上游 task 通常也已过期 → 轮询返回 FAILED / 404 转
        ResumeExpiredError,无需重新上传图。
        """
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            return await self._poll_and_build(client, job_id, request, is_resume=True)

    # ── request building ────────────────────────────────────────────────

    def _validate_duration(self, duration_seconds: int) -> int:
        if not (_MIN_DURATION <= duration_seconds <= _MAX_DURATION):
            raise VideoCapabilityError(
                "video_duration_not_supported",
                model=self._model,
                duration=duration_seconds,
                supported=f"{_MIN_DURATION}-{_MAX_DURATION}",
            )
        return duration_seconds

    async def _build_request(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
        duration: int,
    ) -> tuple[str, dict]:
        """按输入选端点并构造请求体。本地图先上传换 URL。"""
        common: dict = {
            "resolution": request.resolution or "720p",
            "duration": str(duration),
            "generateAudio": request.generate_audio,
            "ratio": request.aspect_ratio,
            "seed": request.seed if request.seed is not None else -1,
        }

        # 参考图 → multimodal 端点(imageUrls)
        # realPersonMode 默认开:小说改编素材常含真人形象,关闭时上游直接拒绝(报错要求设为 true)
        reference_images = [Path(r) for r in (request.reference_images or []) if r]
        if reference_images:
            image_urls = await self._upload_images(client, reference_images, limit=_MAX_REFERENCE_IMAGES)
            payload = {"prompt": request.prompt, "imageUrls": image_urls, "realPersonMode": True, **common}
            return _MULTIMODAL_VIDEO_PATH, payload

        # 首帧(可选尾帧)→ image 端点
        # realPersonMode 默认开:分镜首帧常含真人形象,关闭时上游直接拒绝(报错要求设为 true)
        if request.start_image:
            first_url = await self._upload_image(client, Path(request.start_image))
            payload = {"prompt": request.prompt, "firstFrameUrl": first_url, "realPersonMode": True, **common}
            if request.end_image and Path(request.end_image).is_file():
                payload["lastFrameUrl"] = await self._upload_image(client, Path(request.end_image))
            return _IMAGE_TO_VIDEO_PATH, payload

        # 无图 → text 端点
        payload = {"prompt": request.prompt, **common}
        return _TEXT_TO_VIDEO_PATH, payload

    async def _upload_images(self, client: httpx.AsyncClient, paths: list[Path], *, limit: int) -> list[str]:
        if len(paths) > limit:
            logger.warning("RunningHub 参考图数量 %d 超过上限 %d，截断", len(paths), limit)
            paths = paths[:limit]
        return [await self._upload_image(client, p) for p in paths]

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _upload_image(self, client: httpx.AsyncClient, path: Path) -> str:
        """上传本地图到 RunningHub,返回 download_url(COS 签名 URL,1 天有效)。

        fail-loud:图缺失/不可读 → VideoCapabilityError;上游 code != 0 → RuntimeError。
        """
        if not path.is_file():
            raise VideoCapabilityError("video_start_image_unreadable", model=self._model, name=path.name)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise VideoCapabilityError("video_start_image_unreadable", model=self._model, name=path.name) from exc

        mime = IMAGE_MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        resp = await client.post(
            f"{_UPLOAD_BASE_URL}{_UPLOAD_PATH}",
            files={"file": (path.name, data, mime)},
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"RunningHub 文件上传失败: code={body.get('code')} message={body.get('message')}")
        download_url = (body.get("data") or {}).get("download_url")
        if not download_url:
            raise RuntimeError(f"RunningHub 上传返回体缺少 download_url: {body}")
        return download_url

    # ── HTTP submit / poll / download ───────────────────────────────────

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _create_task(self, client: httpx.AsyncClient, path: str, payload: dict) -> str:
        resp = await client.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
        task_id = body.get("taskId")
        if not task_id:
            raise RuntimeError(f"RunningHub 创建任务返回体缺少 taskId: {body}")
        return task_id

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict:
        resp = await client.post(
            f"{self._base_url}{_QUERY_PATH}",
            json={"taskId": task_id},
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def _poll_and_build(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        request: VideoGenerationRequest,
        *,
        is_resume: bool,
    ) -> VideoGenerationResult:
        # resume 路径下查询 404(task 完全不存在)直接转 ResumeExpiredError,不重试到超时。
        async def _gated_poll() -> dict:
            try:
                return await self._poll_once(client, task_id)
            except httpx.HTTPStatusError as exc:
                if is_resume and exc.response.status_code == 404:
                    raise ResumeExpiredError(job_id=task_id, provider=PROVIDER_RUNNINGHUB) from exc
                raise

        final = await poll_with_retry(
            poll_fn=_gated_poll,
            is_done=lambda state: state.get("status") == "SUCCESS",
            is_failed=_extract_failure,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=self._max_wait(request.duration_seconds),
            retry_if=should_retry_poll,
            label="RunningHub",
            on_progress=lambda v, elapsed: logger.info(
                "RunningHub 视频生成中... status=%s elapsed=%ds", v.get("status"), int(elapsed)
            ),
        )

        results = final.get("results") or []
        video_url = results[0].get("url") if results else None
        if not video_url:
            raise RuntimeError(f"RunningHub 任务完成但缺少结果 url: {final}")

        await self._download_with_retry(video_url, request.output_path)
        logger.info("RunningHub 视频下载完成: %s", request.output_path)

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_RUNNINGHUB,
            model=self._model,
            duration_seconds=request.duration_seconds,
            video_uri=video_url,
            task_id=task_id,
            generate_audio=request.generate_audio,
        )

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retry_if=should_retry_poll,
    )
    async def _download_with_retry(video_url: str, output_path: Path) -> None:
        await download_video(video_url, output_path)

    def _json_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _max_wait(duration_seconds: int) -> float:
        return max(_MIN_POLL_TIMEOUT_SECONDS, duration_seconds * _POLL_TIMEOUT_PER_SECOND)


def _extract_failure(state: dict) -> str | None:
    """FAILED 终态 → 错误信息;其余(QUEUED/RUNNING/SUCCESS)返回 None。"""
    if state.get("status") != "FAILED":
        return None
    msg = state.get("errorMessage") or state.get("errorCode") or "unknown error"
    return f"RunningHub 视频生成失败: {msg}"
