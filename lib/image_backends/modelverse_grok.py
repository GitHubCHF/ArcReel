"""ModelVerseGrokImageBackend — ModelVerse(api.modelverse.cn)Grok Imagine 图像后端。

Grok Imagine 在 ModelVerse 上是**标准 OpenAI 图像协议**(同步,一次返回),但有 Grok 专属参数,
不能直接套通用 OpenAIImageBackend(它发像素 ``size`` + ``quality``,会被 Grok 的
``additionalProperties:false`` + ``size`` 枚举拒):

- 文生图:``POST /v1/images/generations`` body ``{model,prompt,n,size,aspect_ratio}``
- 图生图/风格迁移:``POST /v1/images/edits``,单图 ``image`` / 多图 ``images``(URI,本项目传 base64 data URI)
- ``size`` 仅枚举 ``1k``/``2k``;``aspect_ratio`` 独立枚举;响应 ``data[].url``(或 ``b64_json``)

支持模型:``grok-imagine-image`` / ``grok-imagine-image-quality``。鉴权 ``Authorization: Bearer``。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import httpx

from lib.image_backends.base import (
    ImageCapability,
    ImageCapabilityError,
    ImageGenerationRequest,
    ImageGenerationResult,
    download_image_to_path,
    image_to_base64_data_uri,
)
from lib.logging_utils import format_kwargs_for_log
from lib.retry import with_retry_async
from lib.video_backends.base import should_retry_submit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-imagine-image"
DEFAULT_BASE_URL = "https://api.modelverse.cn"
_DEFAULT_PROVIDER_NAME = "modelverse"

_GENERATIONS_PATH = "/v1/images/generations"
_EDITS_PATH = "/v1/images/edits"

# Grok Imagine 支持的宽高比枚举(与 ModelVerse 文档一致);不在集合内则退回 auto(避免 400)。
_SUPPORTED_ASPECT_RATIOS = {
    "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2",
    "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto",
}  # fmt: skip

# 单次编辑参考图上限(经验值;文档未限死,给个稳妥上限避免超长请求)
_MAX_REFERENCE_IMAGES = 8

# 单张图 HTTP 超时;Grok Imagine 同步返回但 2k 出图仍可能几十秒。
_HTTP_TIMEOUT = 180.0

# 让 ModelVerse 把图以 base64 内联返回:Grok 结果 url 落在 xAI imgen CDN(imgen.x.ai),服务端
# 直取被挡 403(不止查 UA)。请求 b64_json 从已鉴权的 ModelVerse 响应直接拿字节,彻底不碰 CDN。
_RESPONSE_FORMAT = "b64_json"

# url 兜底下载(relay 忽略 response_format 仍只回 url 时)带浏览器 UA 尽力一试。
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def _normalize_base_url(base_url: str | None) -> str:
    """容忍 host-only / 尾斜杠 / 误带 ``/v1``,统一为不带路径的 base;由本后端补 ``/v1/images/*``。"""
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_BASE_URL
    if "://" not in s:
        s = f"https://{s}"
    if s.endswith("/v1"):
        s = s[: -len("/v1")].rstrip("/")
    return s


def _map_size(image_size: str | None) -> str:
    """把请求清晰度映射到 Grok 的 ``1k``/``2k`` 枚举:除非显式 1k,一律 2k(供视频帧用,优先清晰度)。"""
    return "1k" if "1k" in (image_size or "").strip().lower() else "2k"


def _validate_aspect_ratio(aspect_ratio: str) -> str:
    if aspect_ratio in _SUPPORTED_ASPECT_RATIOS:
        return aspect_ratio
    logger.warning("Grok Imagine 不支持 aspect_ratio=%s,退回 auto", aspect_ratio)
    return "auto"


class ModelVerseGrokImageBackend:
    """ModelVerse Grok Imagine 图像后端(同步 OpenAI 风格 REST,支持 T2I / I2I)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        http_timeout: float = _HTTP_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("ModelVerseGrokImageBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._provider_name = provider_name
        self._http_timeout = http_timeout
        self._capabilities: set[ImageCapability] = {
            ImageCapability.TEXT_TO_IMAGE,
            ImageCapability.IMAGE_TO_IMAGE,
        }

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[ImageCapability]:
        return self._capabilities

    @with_retry_async(retry_if=should_retry_submit)
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        has_refs = bool(request.reference_images)
        path, payload = self._build_edit(request) if has_refs else self._build_generate(request)
        logger.info(
            "调用 ModelVerse Grok 图片 API path=%s body=%s",
            path,
            format_kwargs_for_log(_safe_payload_for_log(payload)),
        )
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            resp = await client.post(f"{self._base_url}{path}", json=payload, headers=self._headers())
            if resp.status_code >= 400:
                logger.warning("ModelVerse Grok 图片接口返回 %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            body = resp.json()

        data = body.get("data") or []
        if not data:
            raise RuntimeError(f"ModelVerse Grok 图片响应 data 为空 (model={self._model}),可能触发内容审核或上游异常")
        await self._save_item(data[0], request.output_path)
        logger.info("ModelVerse Grok 图片生成完成: %s", request.output_path)

        return ImageGenerationResult(
            image_path=request.output_path,
            provider=self._provider_name,
            model=self._model,
            image_uri=data[0].get("url"),
        )

    def _build_generate(self, request: ImageGenerationRequest) -> tuple[str, dict]:
        payload = {
            "model": self._model,
            "prompt": request.prompt,
            "n": 1,
            "size": _map_size(request.image_size),
            "aspect_ratio": _validate_aspect_ratio(request.aspect_ratio),
            "response_format": _RESPONSE_FORMAT,
        }
        return _GENERATIONS_PATH, payload

    def _build_edit(self, request: ImageGenerationRequest) -> tuple[str, dict]:
        # fail-loud:声明了参考图但全部不可读 → 报错,不静默降级到 T2I
        data_uris: list[str] = []
        for ref in request.reference_images[:_MAX_REFERENCE_IMAGES]:
            p = Path(ref.path) if ref.path else None
            if p is not None and p.is_file():
                data_uris.append(image_to_base64_data_uri(p))
        if not data_uris:
            raise ImageCapabilityError("image_endpoint_mismatch_no_i2i", model=self._model)

        payload: dict = {
            "model": self._model,
            "prompt": request.prompt,
            "n": 1,
            "size": _map_size(request.image_size),
            "aspect_ratio": _validate_aspect_ratio(request.aspect_ratio),
            "response_format": _RESPONSE_FORMAT,
        }
        # 单图用 image、多图用 images(与文档一致,二者互斥)
        if len(data_uris) == 1:
            payload["image"] = data_uris[0]
        else:
            payload["images"] = data_uris
        return _EDITS_PATH, payload

    async def _save_item(self, item: dict, output_path: Path) -> None:
        """从 ``data[0]`` 落盘:优先 b64_json 内联字节(我们请求的就是它,绕开 CDN),其次 url 兜底下载。"""
        b64 = item.get("b64_json")
        if b64:

            def _decode_and_save() -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(base64.b64decode(b64))

            await asyncio.to_thread(_decode_and_save)
            return
        url = item.get("url")
        if url:
            # relay 忽略 response_format 只回 url 时的兜底:带浏览器 UA 尽力下载
            await download_image_to_path(url, output_path, headers=_BROWSER_HEADERS)
            return
        raise RuntimeError(f"ModelVerse Grok 图片响应项既无 b64_json 也无 url: {item}")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}


def _safe_payload_for_log(payload: dict) -> dict:
    """日志里把 base64 参考图折叠成占位,避免刷屏。"""
    safe = dict(payload)
    if "image" in safe:
        safe["image"] = "<data-uri>"
    if "images" in safe:
        safe["images"] = f"<{len(payload['images'])} data-uri>"
    return safe
