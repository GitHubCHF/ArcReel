"""TecDoVideoBackend — 钛极模型网关(钛动新平台)seedance 视频生成后端。

钛极网关是火山 Ark 原生协议的透传网关(替代旧 open-power.tec-do.cn 自研协议):
- 鉴权 ``Authorization: Bearer {api_key}``,所有接口同域 ``https://api.tcgateway.top``
- 视频任务走 Ark v3 端点 ``POST /api/v3/contents/generations/tasks``,靠 ``content[]``
  数组 + role 派发模式:
  - text item ``{type:text, text}`` 必有至少一个
  - 首帧 ``{type:image_url, image_url:{url}, role:first_frame}``,可选尾帧 ``role:last_frame``
  - 参考图 ``{type:image_url, image_url:{url}, role:reference_image}`` 可多张
- 请求体含 ``model`` 字段(seedance-2-0-260128);``duration`` 为 int(4~15);``seed`` 可选
- 图片是 URL:本地图先上传到阿里云 OSS(见 ``lib/oss_uploader``)换签名 URL。OSS 配置缺失时
  图生/参考生视频 fail-loud(文生视频不受影响)。
- 参考图走网关资产库过审(Ark Action 风格 ``POST /api/ark?Action=CreateAsset&Version=...``),
  且资产必须归属资产组:惰性 ``CreateAssetGroup`` 一次并持久化复用 GroupId
- 异步:create 返回顶层 ``id`` → 轮询 ``GET .../tasks/{task_id}`` 至 ``status=succeeded``
  (兼容 completed/success)→ ``content.video_url`` 下载;计费仅返回 ``usage.total_tokens``
  (旧平台的 actualAmount 已不存在,实际费用无法回报)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from lib.logging_utils import format_kwargs_for_log
from lib.oss_uploader import OSSConfig, OSSUploader
from lib.providers import PROVIDER_TECDO

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker
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

# 缺省 model（未显式指定时下传给网关 API）；大小写敏感，须与 registry key 一致。
DEFAULT_MODEL = "seedance-2-0-260128"
DEFAULT_BASE_URL = "https://api.tcgateway.top"

_TASKS_PATH = "/api/v3/contents/generations/tasks"
# 资产组/资产 CRUD 走 Ark Action 风格端点(Query 传 Action + Version)。
_ARK_ACTION_PATH = "/api/ark"
_ARK_ACTION_VERSION = "2024-01-01"

# 视频任务终态(统一大写后比对);网关另有 queued/running 过程态。
_DONE_STATUSES = {"SUCCEEDED", "COMPLETED", "SUCCESS"}
_FAILED_STATUSES = {"FAILED", "FAILURE", "ERROR"}

_POLL_INTERVAL_SECONDS = 10.0
_MIN_POLL_TIMEOUT_SECONDS = 1200
_POLL_TIMEOUT_PER_SECOND = 60

# 资产库轮询(Pending→Active):图片资产处理通常很快,180s 足够。
_ASSET_POLL_INTERVAL_SECONDS = 3.0
_ASSET_POLL_TIMEOUT_SECONDS = 180.0

# duration 合法值域(秒);seedance2.0 上游枚举为 4~15。
_MIN_DURATION = 4
_MAX_DURATION = 15

# 参考图上限(文档仅称"可多张"未给数,沿用 seedance 常见 9 上限)。
_MAX_REFERENCE_IMAGES = 9

# 资产缓存模式:cached=按内容哈希复用 assetId(默认);recreate=每次请求重新登记资产、
# 不读不写缓存(规避上游资产过期导致 "asset not found" 整条失败)。
ASSET_CACHE_MODE_CACHED = "cached"
ASSET_CACHE_MODE_RECREATE = "recreate"

# 资产组名称与其在 provider_assets 缓存中的哨兵 content_hash(与 sha256 十六进制不可能撞)。
# 资产组是基础设施而非审核结果,不受 asset_cache_mode 影响,恒持久化复用。
_ASSET_GROUP_NAME = "arcreel_assets"
_ASSET_GROUP_SENTINEL_HASH = "__asset_group__"


def _normalize_base_url(base_url: str | None) -> str:
    """容忍用户填到 host(或带尾斜杠),归一化为不带尾斜杠的 base。空值回落官方域名。"""
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_BASE_URL
    return s if "://" in s else f"https://{s}"


class TecDoVideoBackend:
    """钛极模型网关 seedance 视频后端(异步上传→提交→轮询→下载)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        http_timeout: float = 60.0,
        oss_config: dict[str, str] | None = None,
        session_factory: async_sessionmaker | None = None,
        asset_cache_mode: str = ASSET_CACHE_MODE_CACHED,
    ) -> None:
        if not api_key:
            raise ValueError("TecDoVideoBackend 需要 api_key")
        self._api_key = api_key
        # 资产缓存模式:recreate 时每次请求重新登记资产、跳过持久化缓存读写。
        self._asset_cache_mode = (
            ASSET_CACHE_MODE_RECREATE if asset_cache_mode == ASSET_CACHE_MODE_RECREATE else ASSET_CACHE_MODE_CACHED
        )
        # 资产在不同密钥间隔离,缓存按密钥指纹(sha256,不存明文)分层。
        self._key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        self._base_url = _normalize_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._http_timeout = http_timeout
        self._oss_config = OSSConfig.from_dict(oss_config)
        self._uploader: OSSUploader | None = None
        # 资产库 assetId 缓存读写需要 DB；非 worker 路径(测试/直生)可不传,降级为不缓存。
        self._session_factory = session_factory
        # 资产组 GroupId 进程内缓存(DB 哨兵行为二级缓存)。
        self._group_id: str | None = None
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
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            payload = await self._build_payload(client, request, duration)
            logger.info(
                "调用 %s 视频 API path=%s body=%s",
                self.name,
                _TASKS_PATH,
                format_kwargs_for_log(payload),
            )
            task_id = await self._create_task(client, payload)
            logger.info("钛动视频任务已创建: task_id=%s model=%s", task_id, self._model)
            if request.task_id is not None:
                await persist_provider_job_id(request.task_id, task_id, provider=PROVIDER_TECDO)
            return await self._poll_and_build(client, task_id, request, is_resume=False)

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        """接续已 submit 的钛动 task：仅 poll + 下载(ADR 0007)。

        上传 URL / 结果有时效,超期上游 task 通常也已过期 → 轮询返回 failed / 404 转
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

    async def _build_payload(self, client: httpx.AsyncClient, request: VideoGenerationRequest, duration: int) -> dict:
        """构造 Ark v3 请求体,按首尾帧/参考图拼 content[] role。

        - 参考图(角色集)→ 走资产库:OSS 换公网 URL → 登记资产过审 → ``asset://{id}``(按内容哈希缓存复用)
        - 首尾帧 → OSS 直传签名 URL
        """
        # 至少一个 text 内容(网关/Ark 惯例 text 在前)。
        content: list[dict] = [{"type": "text", "text": request.prompt}]

        # 参考图(可多张)→ role=reference_image,走资产库以通过真人/敏感审核
        reference_images = [Path(r) for r in (request.reference_images or []) if r]
        if reference_images:
            for url in await self._resolve_reference_assets(client, reference_images, limit=_MAX_REFERENCE_IMAGES):
                content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
        # 否则首帧(可选尾帧)→ role=first_frame / last_frame,OSS 直传
        elif request.start_image:
            first_url = await self._upload_image(Path(request.start_image))
            content.append({"type": "image_url", "image_url": {"url": first_url}, "role": "first_frame"})
            if request.end_image and Path(request.end_image).is_file():
                last_url = await self._upload_image(Path(request.end_image))
                content.append({"type": "image_url", "image_url": {"url": last_url}, "role": "last_frame"})

        payload: dict = {
            "model": self._model,
            "content": content,
            "resolution": request.resolution or "720p",
            "ratio": request.aspect_ratio,
            "duration": duration,
            "watermark": False,
            "generate_audio": request.generate_audio,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def _get_uploader(self) -> OSSUploader:
        """惰性构造 OSS 上传器(签名 URL,与旧平台跑通时逻辑一致)。未配置 OSS 时 fail-loud。"""
        if self._uploader is None:
            if not self._oss_config.is_complete:
                raise RuntimeError(
                    "钛动图生/参考生视频需要先在系统设置配置阿里云 OSS"
                    "(endpoint / bucket / access_key_id / access_key_secret)"
                )
            self._uploader = OSSUploader(self._oss_config)
        return self._uploader

    # ── 参考图 → 资产库(角色集) ────────────────────────────────────────

    async def _resolve_reference_assets(self, client: httpx.AsyncClient, paths: list[Path], *, limit: int) -> list[str]:
        if len(paths) > limit:
            logger.warning("钛动参考图数量 %d 超过上限 %d，截断", len(paths), limit)
            paths = paths[:limit]
        return [await self._resolve_reference_asset(client, p) for p in paths]

    async def _resolve_reference_asset(self, client: httpx.AsyncClient, path: Path) -> str:
        """参考图(角色集)→ ``asset://{assetId}``。

        cached 模式:按图片内容哈希持久化复用,命中且 Active 直接返回跳过所有网络调用;
        未命中则 OSS 换公网 URL → CreateAsset 登记 → 轮询 Active → 写缓存。
        recreate 模式:跳过缓存读写,每次都重新登记资产(规避上游资产过期)。
        """
        if not path.is_file():
            raise VideoCapabilityError("video_start_image_unreadable", model=self._model, name=path.name)

        cache_enabled = self._asset_cache_mode != ASSET_CACHE_MODE_RECREATE
        content_hash: str | None = None
        if cache_enabled:
            content_hash = await asyncio.to_thread(_sha256_file, path)
            cached = await self._get_cached_asset(content_hash)
            if cached is not None:
                logger.info("钛动资产缓存命中: hash=%s asset_id=%s", content_hash[:12], cached)
                return f"asset://{cached}"

        oss_url = await self._upload_image(path)
        group_id = await self._get_or_create_group(client)
        # 打实际下发 URL:网关侧会剥 query 后回显/重拉,排障时可与 GetAsset 回显对比。
        logger.info("钛动登记资产: name=%s url=%s", path.name, oss_url)
        asset_id = await self._create_asset(client, oss_url, group_id=group_id, name=path.name)
        logger.info("钛动资产已创建,等待过审: asset_id=%s", asset_id)
        await self._wait_asset_active(client, asset_id)
        if cache_enabled and content_hash is not None:
            await self._save_cached_asset(content_hash, asset_id)
        return f"asset://{asset_id}"

    async def _get_or_create_group(self, client: httpx.AsyncClient) -> str:
        """资产组 GroupId:进程内缓存 → DB 哨兵行 → CreateAssetGroup 新建并回写。"""
        if self._group_id is not None:
            return self._group_id
        cached = await self._get_cached_asset(_ASSET_GROUP_SENTINEL_HASH)
        if cached is not None:
            self._group_id = cached
            return cached
        group_id = await self._create_group(client)
        logger.info("钛动资产组已创建: group_id=%s", group_id)
        await self._save_cached_asset(_ASSET_GROUP_SENTINEL_HASH, group_id)
        self._group_id = group_id
        return group_id

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _create_group(self, client: httpx.AsyncClient) -> str:
        body = await self._ark_action(
            client,
            "CreateAssetGroup",
            {"Name": _ASSET_GROUP_NAME, "Description": "ArcReel 参考图资产组"},
        )
        group_id = (body.get("Result") or {}).get("Id")
        if not group_id:
            raise RuntimeError(f"钛动资产组创建失败或返回缺少 Result.Id: {body}")
        return group_id

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _create_asset(self, client: httpx.AsyncClient, url: str, *, group_id: str, name: str) -> str:
        body = await self._ark_action(
            client,
            "CreateAsset",
            {"GroupId": group_id, "Name": name, "AssetType": "Image", "URL": url},
        )
        asset_id = (body.get("Result") or {}).get("Id")
        if not asset_id:
            raise RuntimeError(f"钛动资产创建失败或返回缺少 Result.Id: {body}")
        return asset_id

    async def _wait_asset_active(self, client: httpx.AsyncClient, asset_id: str) -> None:
        async def _poll() -> dict:
            body = await self._ark_action(client, "GetAsset", {"Id": asset_id})
            return body.get("Result") or {}

        await poll_with_retry(
            poll_fn=_poll,
            is_done=lambda s: (s.get("Status") or "").upper() == "ACTIVE",
            is_failed=lambda s: (
                f"钛动资产审核失败(可能含真人/敏感内容): asset_id={asset_id}"
                if (s.get("Status") or "").upper() == "FAILED"
                else None
            ),
            poll_interval=_ASSET_POLL_INTERVAL_SECONDS,
            max_wait=_ASSET_POLL_TIMEOUT_SECONDS,
            retry_if=should_retry_poll,
            label="TecDo-Asset",
        )

    async def _ark_action(self, client: httpx.AsyncClient, action: str, payload: dict) -> dict:
        """Ark Action 风格调用:``POST /api/ark?Action=X&Version=...``,返回响应 JSON。

        请求/响应全量打日志:新网关行为与文档常有出入,排障靠原始报文。
        """
        logger.info("钛动 Action=%s 请求: %s", action, payload)
        resp = await client.post(
            f"{self._base_url}{_ARK_ACTION_PATH}",
            params={"Action": action, "Version": _ARK_ACTION_VERSION},
            json=payload,
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
        logger.info("钛动 Action=%s 响应: %s", action, body)
        return body

    async def _get_cached_asset(self, content_hash: str) -> str | None:
        """查持久化缓存:返回 Active 资产的 assetId,否则 None(含无 DB 时)。"""
        if self._session_factory is None:
            return None
        from lib.db.repositories.provider_asset_repo import ProviderAssetRepository

        async with self._session_factory() as session:
            row = await ProviderAssetRepository(session).get(PROVIDER_TECDO, self._key_hash, content_hash)
            if row is not None and row.status.upper() == "ACTIVE":
                return row.asset_id
        return None

    async def _save_cached_asset(self, content_hash: str, asset_id: str) -> None:
        if self._session_factory is None:
            return
        from lib.db.repositories.provider_asset_repo import ProviderAssetRepository

        async with self._session_factory() as session:
            await ProviderAssetRepository(session).upsert(
                provider=PROVIDER_TECDO,
                key_hash=self._key_hash,
                content_hash=content_hash,
                asset_id=asset_id,
                status="Active",
            )
            await session.commit()

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
            f"{self._base_url}{_TASKS_PATH}",
            json=payload,
            headers=self._json_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
        logger.info("钛动创建任务响应: %s", body)
        task_id = body.get("id")
        if not task_id:
            raise RuntimeError(f"钛动创建任务返回体缺少 id: {body}")
        return task_id

    async def _poll_once(self, client: httpx.AsyncClient, task_id: str) -> dict:
        resp = await client.get(
            f"{self._base_url}{_TASKS_PATH}/{task_id}",
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
                    raise ResumeExpiredError(job_id=task_id, provider=PROVIDER_TECDO) from exc
                raise

        final = await poll_with_retry(
            poll_fn=_gated_poll,
            # 统一大写后比对:标准终态 succeeded/failed,另兼容 completed/success/failure/error。
            is_done=lambda state: (state.get("status") or "").upper() in _DONE_STATUSES,
            is_failed=_extract_failure,
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=self._max_wait(request.duration_seconds),
            retry_if=should_retry_poll,
            label="TecDo",
            # 打完整轮询响应(不只 status):网关返回结构与文档常有出入,排障靠原始报文。
            on_progress=lambda v, elapsed: logger.info("钛动视频生成中... elapsed=%ds 响应=%s", int(elapsed), v),
        )
        logger.info("钛动任务终态响应: %s", final)

        video_url = (final.get("content") or {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"钛动任务完成但缺少结果 content.video_url: {final}")

        await self._download_with_retry(video_url, request.output_path)
        logger.info("钛动视频下载完成: %s", request.output_path)

        # 网关只回报 token 用量(按其计费策略在平台侧结算),无金额字段可透传。
        total_tokens = (final.get("usage") or {}).get("total_tokens")
        if total_tokens is not None:
            logger.info("钛动视频任务用量: task_id=%s total_tokens=%s", task_id, total_tokens)
        return VideoGenerationResult(
            video_path=request.output_path,
            provider=PROVIDER_TECDO,
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
    """failed/failure/error 终态 → 错误信息;其余(queued/running/succeeded)返回 None。

    ``error`` 字段可能是对象(``{message}``)或字符串,两种形态都兼容。
    """
    if (state.get("status") or "").upper() not in _FAILED_STATUSES:
        return None
    err = state.get("error")
    msg = (err.get("message") if isinstance(err, dict) else err) or "unknown error"
    return f"钛动视频生成失败: {msg}"


def _sha256_file(path: Path) -> str:
    """计算文件 sha256(分块读,适配大图)。同步阻塞,异步调用方用 to_thread 包裹。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
