"""TecDoVideoBackend — 钛动能力平台(open-power.tec-do.cn)seedance 2.0 视频生成后端。

第三方中转站,协议与 RunningHub / 火山原生 ark 均不同:
- 鉴权 ``X-App-Secret: {api_key}`` 单 header,三个接口同域 ``https://open-power.tec-do.cn``
- 单 create 端点 ``POST /tecpower/ai/openapi/video/create``,靠 ``content[]`` 数组 + role 派发模式
  (不像 RunningHub 三个 URL):
  - text item ``{type:text, text}`` 必有至少一个
  - 首帧 ``{type:image_url, imageUrl:{url}, role:first_frame}``,可选尾帧 ``role:last_frame``
  - 参考图 ``{type:image_url, imageUrl:{url}, role:reference_image}`` 可多张
- 请求体含 ``model`` 字段(seedance2.0);``duration`` 为 int(4~15);``seed`` 可选
- 图片是 URL:本地图先上传到阿里云 OSS(见 ``lib/oss_uploader``)换签名 URL,再填进
  ``imageUrl.url``。钛动自带的上传接口需单独开通,故不走它。OSS 配置缺失时图生/参考生
  视频 fail-loud(文生视频不受影响)。
- 异步:create 返回 ``data.taskId`` → 轮询 ``GET /tecpower/ai/openapi/video/task?taskId=`` 至
  ``status=COMPLETED`` → ``data.videoUrl`` 下载;``data.actualAmount`` 为实际消耗金额(回报计费)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from lib.logging_utils import format_kwargs_for_log
from lib.oss_uploader import OSSConfig, OSSUploader
from lib.providers import PROVIDER_TECDO
from lib.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import (
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

DEFAULT_MODEL = "seedance2.0"
DEFAULT_BASE_URL = "https://open-power.tec-do.cn"

_CREATE_PATH = "/tecpower/ai/openapi/video/create"
_QUERY_PATH = "/tecpower/ai/openapi/video/task"

# actualAmount 货币:钛动为国内平台,按人民币记账。
_ACTUAL_CURRENCY = "CNY"

_POLL_INTERVAL_SECONDS = 10.0
_MIN_POLL_TIMEOUT_SECONDS = 1200
_POLL_TIMEOUT_PER_SECOND = 60

# duration 合法值域(秒);seedance2.0 上游枚举为 4~15。
_MIN_DURATION = 4
_MAX_DURATION = 15

# 参考图上限(文档仅称"可多张"未给数,沿用 seedance 常见 9 上限)。
_MAX_REFERENCE_IMAGES = 9


def _normalize_base_url(base_url: str | None) -> str:
    """容忍用户填到 host(或带尾斜杠),归一化为不带尾斜杠的 base。空值回落官方域名。"""
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_BASE_URL
    return s if "://" in s else f"https://{s}"


class TecDoVideoBackend:
    """钛动 seedance 2.0 视频后端(异步上传→提交→轮询→下载)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
        oss_config: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("TecDoVideoBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._oss_config = OSSConfig.from_dict(oss_config)
        self._uploader: OSSUploader | None = None
        self._capabilities: set[VideoCapability] = {
            VideoCapability.TEXT_TO_VIDEO,
            VideoCapability.IMAGE_TO_VIDEO,
            VideoCapability.GENERATE_AUDIO,
            VideoCapability.SEED_CONTROL,
        }

    @property
    def name(self) -> str:
        return PROVIDER_TECDO

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[VideoCapability]:
        return self._capabilities

    @staticmethod
    def video_capabilities_for_model(model: str) -> VideoCapabilities:
        """纯计算参考图等 caps —— 不构造 client(无需 api_key)。

        seedance2.0 通过 content[] role 区分模式,所有 model 共享同一套能力
        (首尾帧 + 参考图 ≤9),故恒返回固定上限。
        """
        return VideoCapabilities(last_frame=True, reference_images=True, max_reference_images=_MAX_REFERENCE_IMAGES)

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return self.video_capabilities_for_model(self._model)

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        duration = self._validate_duration(request.duration_seconds)
        payload = await self._build_payload(request, duration)
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            logger.info(
                "调用 %s 视频 API path=%s body=%s",
                self.name,
                _CREATE_PATH,
                format_kwargs_for_log(payload),
            )
            task_id = await self._create_task(client, payload)
            logger.info("钛动视频任务已创建: task_id=%s model=%s", task_id, self._model)
            if request.task_id is not None:
                await persist_provider_job_id(request.task_id, task_id, provider=PROVIDER_TECDO)
            return await self._poll_and_build(client, task_id, request, is_resume=False)

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        """接续已 submit 的钛动 task：仅 poll + 下载(ADR 0007)。

        上传 URL / 结果有时效,超期上游 task 通常也已过期 → 轮询返回 FAILED / 404 转
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

    async def _build_payload(self, request: VideoGenerationRequest, duration: int) -> dict:
        """构造单端点请求体:本地图先上传 OSS 换签名 URL,按首尾帧/参考图拼 content[] role。"""
        content: list[dict] = []

        # 参考图(可多张)→ role=reference_image
        reference_images = [Path(r) for r in (request.reference_images or []) if r]
        if reference_images:
            for url in await self._upload_images(reference_images, limit=_MAX_REFERENCE_IMAGES):
                content.append({"type": "image_url", "imageUrl": {"url": url}, "role": "reference_image"})
        # 否则首帧(可选尾帧)→ role=first_frame / last_frame
        elif request.start_image:
            first_url = await self._upload_image(Path(request.start_image))
            content.append({"type": "image_url", "imageUrl": {"url": first_url}, "role": "first_frame"})
            if request.end_image and Path(request.end_image).is_file():
                last_url = await self._upload_image(Path(request.end_image))
                content.append({"type": "image_url", "imageUrl": {"url": last_url}, "role": "last_frame"})

        # 至少一个 text 内容
        content.append({"type": "text", "text": request.prompt})

        payload: dict = {
            "model": self._model,
            "content": content,
            "resolution": request.resolution or "720p",
            "ratio": request.aspect_ratio,
            "duration": duration,
            "watermark": False,
            "generateAudio": request.generate_audio,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def _get_uploader(self) -> OSSUploader:
        """惰性构造 OSS 上传器。未配置 OSS 时 fail-loud(图生/参考生视频依赖它)。"""
        if self._uploader is None:
            if not self._oss_config.is_complete:
                raise RuntimeError(
                    "钛动图生/参考生视频需要先在系统设置配置阿里云 OSS"
                    "(endpoint / bucket / access_key_id / access_key_secret)"
                )
            self._uploader = OSSUploader(self._oss_config)
        return self._uploader

    async def _upload_images(self, paths: list[Path], *, limit: int) -> list[str]:
        if len(paths) > limit:
            logger.warning("钛动参考图数量 %d 超过上限 %d，截断", len(paths), limit)
            paths = paths[:limit]
        return [await self._upload_image(p) for p in paths]

    async def _upload_image(self, path: Path) -> str:
        """上传本地图到 OSS,返回临时签名 URL。

        fail-loud:图缺失/不可读 → VideoCapabilityError;OSS 未配置 → RuntimeError。
        """
        if not path.is_file():
            raise VideoCapabilityError("video_start_image_unreadable", model=self._model, name=path.name)
        uploader = self._get_uploader()
        try:
            return await asyncio.to_thread(uploader.upload_file, path)
        except OSError as exc:
            raise VideoCapabilityError("video_start_image_unreadable", model=self._model, name=path.name) from exc

    # ── HTTP submit / poll / download ───────────────────────────────────

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _create_task(self, client: httpx.AsyncClient, payload: dict) -> str:
        resp = await client.post(
            f"{self._base_url}{_CREATE_PATH}",
            json=payload,
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
        task_id = (body.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"钛动创建任务返回体缺少 taskId: {body}")
        return task_id

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict:
        resp = await client.get(
            f"{self._base_url}{_QUERY_PATH}",
            params={"taskId": task_id},
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("data") or {}

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
                    raise ResumeExpiredError(job_id=task_id, provider=PROVIDER_TECDO) from exc
                raise

        final = await poll_with_retry(
            poll_fn=_gated_poll,
            # 上游 status 实际为小写(completed/failed/...)，文档写的大写不可信，统一大写后比对。
            is_done=lambda state: (state.get("status") or "").upper() == "COMPLETED",
            is_failed=_extract_failure,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=self._max_wait(request.duration_seconds),
            retry_if=should_retry_poll,
            label="TecDo",
            on_progress=lambda v, elapsed: logger.info(
                "钛动视频生成中... status=%s elapsed=%ds", v.get("status"), int(elapsed)
            ),
        )

        video_url = final.get("videoUrl")
        if not video_url:
            raise RuntimeError(f"钛动任务完成但缺少结果 videoUrl: {final}")

        await self._download_with_retry(video_url, request.output_path)
        logger.info("钛动视频下载完成: %s", request.output_path)

        actual_amount = final.get("actualAmount")
        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_TECDO,
            model=self._model,
            duration_seconds=request.duration_seconds,
            video_uri=video_url,
            task_id=task_id,
            generate_audio=request.generate_audio,
            actual_cost=float(actual_amount) if actual_amount is not None else None,
            # 货币以响应 currency 字段为准(实测为 USD)，缺失时回落默认值。
            actual_currency=(final.get("currency") or _ACTUAL_CURRENCY) if actual_amount is not None else None,
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
        return {"X-App-Secret": self._api_key, "Content-Type": "application/json"}

    @staticmethod
    def _max_wait(duration_seconds: int) -> float:
        return max(_MIN_POLL_TIMEOUT_SECONDS, duration_seconds * _POLL_TIMEOUT_PER_SECOND)


def _extract_failure(state: dict) -> str | None:
    """failed 终态 → 错误信息;其余(pending/processing/completed)返回 None。

    上游 status 实际为小写,大写后比对(文档写大写但实测返回 failed/completed)。
    """
    if (state.get("status") or "").upper() != "FAILED":
        return None
    msg = state.get("error") or "unknown error"
    return f"钛动视频生成失败: {msg}"
