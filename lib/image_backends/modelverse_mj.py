"""ModelVerseMidjourneyImageBackend — ModelVerse(api.modelverse.cn)Midjourney 图像后端。

Midjourney 在 ModelVerse 上不是 OpenAI 协议,而是自有的**异步任务 API**(提交 + 轮询):
- 提交:``POST /v1/tasks/submit`` body ``{"model","input":{"prompt"},"parameters"?}`` → ``output.task_id``
- 轮询:``GET /v1/tasks/status?task_id=`` → ``output.task_status``(Pending/Running/Success/Failure)
  + ``output.urls[]``(拼合图/单图)+ ``output.buttons[]``(后续操作 custom_id)

``midjourney-fast-imagine`` 文生图返回一张 **2×2 四宫格拼图**。ArcReel 需要单图,故默认再走一次
``midjourney-fast-upscale``(U1,带上一步 task_id + 按钮 custom_id)放大出左上角单图。找不到放大按钮
时回退直接用四宫格图,保证有产出。

鉴权与 base_url 同 OpenAI 风格(``Authorization: Bearer``);仅覆盖 imagine(文生图),不支持参考图。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from lib.image_backends.base import (
    ImageCapability,
    ImageCapabilityError,
    ImageGenerationRequest,
    ImageGenerationResult,
    download_image_to_path,
)
from lib.logging_utils import format_kwargs_for_log
from lib.retry import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DOWNLOAD_BACKOFF_SECONDS,
    DOWNLOAD_MAX_ATTEMPTS,
    with_retry_async,
)
from lib.video_backends.base import poll_with_retry, should_retry_poll, should_retry_submit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "midjourney-fast-imagine"
DEFAULT_BASE_URL = "https://api.modelverse.cn"
_DEFAULT_PROVIDER_NAME = "modelverse"

_SUBMIT_PATH = "/v1/tasks/submit"
_STATUS_PATH = "/v1/tasks/status"

_POLL_INTERVAL_SECONDS = 5.0
# imagine 与 upscale 各自的单任务超时;MJ 高峰排队可能较久,给足余量。
_POLL_TIMEOUT_SECONDS = 600.0

_DONE_STATUS = "SUCCESS"
_FAILED_STATUS = "FAILURE"


def _normalize_base_url(base_url: str | None) -> str:
    """容忍 host-only / 带尾斜杠 / 误带 ``/v1`` 尾段,统一为不带路径的 base;由本后端补 ``/v1/tasks/*``。"""
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_BASE_URL
    if "://" not in s:
        s = f"https://{s}"
    if s.endswith("/v1"):
        s = s[: -len("/v1")].rstrip("/")
    return s


def _find_upscale_custom_id(buttons: Sequence[dict]) -> str | None:
    """从状态响应的 buttons 里找放大到单图的 custom_id:优先 U1,退而求其次任意 upsample。"""
    for b in buttons:
        if (b.get("label") or "").strip().upper() == "U1":
            return b.get("custom_id")
    for b in buttons:
        if "upsample::1" in (b.get("custom_id") or ""):
            return b.get("custom_id")
    for b in buttons:
        if "upsample" in (b.get("custom_id") or ""):
            return b.get("custom_id")
    return None


class ModelVerseMidjourneyImageBackend:
    """ModelVerse Midjourney 图像后端(异步提交→轮询→自动 U1 放大→下载单图)。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        provider_name: str = _DEFAULT_PROVIDER_NAME,
        http_timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("ModelVerseMidjourneyImageBackend 需要 api_key")
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._model = model or DEFAULT_MODEL
        self._provider_name = provider_name
        self._http_timeout = http_timeout
        # imagine 仅文生图;参考图(I2I)不在此 API 覆盖内。
        self._capabilities: set[ImageCapability] = {ImageCapability.TEXT_TO_IMAGE}

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[ImageCapability]:
        return self._capabilities

    def _upscale_model(self) -> str:
        """由 imagine 模型 id 推放大模型 id(``*-imagine`` → ``*-upscale``),覆盖 fast/relax/turbo 档。"""
        if "imagine" in self._model:
            return self._model.replace("imagine", "upscale")
        return "midjourney-fast-upscale"

    def _build_prompt(self, request: ImageGenerationRequest) -> str:
        """把请求比例作为 Midjourney ``--ar`` 参数拼进 prompt(用户已写则不覆盖)。"""
        prompt = request.prompt
        if request.aspect_ratio and "--ar" not in prompt:
            prompt = f"{prompt} --ar {request.aspect_ratio}"
        return prompt

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if request.reference_images:
            # imagine 不接受参考图 → 报能力不匹配(与其它 image 后端一致的稳定 code)
            raise ImageCapabilityError("image_endpoint_mismatch_no_i2i", model=self._model)

        prompt = self._build_prompt(request)
        async with httpx.AsyncClient(timeout=self._http_timeout) as client:
            imagine_task_id = await self._submit(client, {"model": self._model, "input": {"prompt": prompt}})
            logger.info("ModelVerse MJ imagine 已提交: task_id=%s model=%s", imagine_task_id, self._model)
            imagine_out = await self._poll(client, imagine_task_id, label="MJ-imagine")

            grid_urls = imagine_out.get("urls") or []
            buttons = imagine_out.get("buttons") or []
            final_url = await self._resolve_single_image(client, imagine_task_id, grid_urls, buttons)

            await self._download_with_retry(final_url, request.output_path)
            logger.info("ModelVerse MJ 图片下载完成: %s", request.output_path)

        return ImageGenerationResult(
            image_path=request.output_path,
            provider=self._provider_name,
            model=self._model,
            image_uri=final_url,
        )

    async def _resolve_single_image(
        self,
        client: httpx.AsyncClient,
        imagine_task_id: str,
        grid_urls: list[str],
        buttons: Sequence[dict],
    ) -> str:
        """默认出单图:走 U1 放大;找不到放大按钮或放大无产出时回退四宫格拼图。"""
        custom_id = _find_upscale_custom_id(buttons)
        if custom_id is None:
            if not grid_urls:
                raise RuntimeError("ModelVerse MJ imagine 完成但既无放大按钮也无 urls,无法取图")
            logger.warning("ModelVerse MJ: 未找到 U1 放大按钮,回退四宫格拼图 url=%s", grid_urls[0])
            return grid_urls[0]

        up_task_id = await self._submit(
            client,
            {
                "model": self._upscale_model(),
                "input": {},
                "parameters": {"mj_task_id": imagine_task_id, "mj_custom_id": custom_id},
            },
        )
        logger.info("ModelVerse MJ upscale(U1) 已提交: task_id=%s", up_task_id)
        up_out = await self._poll(client, up_task_id, label="MJ-upscale")
        urls = up_out.get("urls") or []
        if urls:
            return urls[0]
        if grid_urls:
            logger.warning("ModelVerse MJ: 放大任务无 urls,回退四宫格拼图")
            return grid_urls[0]
        raise RuntimeError("ModelVerse MJ 放大任务完成但缺少结果 url")

    @with_retry_async(
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        retry_if=should_retry_submit,
    )
    async def _submit(self, client: httpx.AsyncClient, payload: dict) -> str:
        logger.info("ModelVerse MJ 提交任务: %s", format_kwargs_for_log(payload))
        resp = await client.post(f"{self._base_url}{_SUBMIT_PATH}", json=payload, headers=self._headers())
        if resp.status_code >= 400:
            logger.warning("ModelVerse MJ 提交返回 %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        body = resp.json()
        task_id = (body.get("output") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"ModelVerse MJ 提交返回缺少 output.task_id: {body}")
        return task_id

    async def _poll(self, client: httpx.AsyncClient, task_id: str, *, label: str) -> dict:
        async def _poll_once() -> dict:
            resp = await client.get(
                f"{self._base_url}{_STATUS_PATH}", params={"task_id": task_id}, headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json().get("output") or {}

        return await poll_with_retry(
            poll_fn=_poll_once,
            is_done=lambda o: (o.get("task_status") or "").upper() == _DONE_STATUS,
            is_failed=lambda o: (
                f"ModelVerse MJ 生成失败: {o.get('error_message') or 'unknown error'}"
                if (o.get("task_status") or "").upper() == _FAILED_STATUS
                else None
            ),
            poll_interval=_POLL_INTERVAL_SECONDS,
            max_wait=_POLL_TIMEOUT_SECONDS,
            retry_if=should_retry_poll,
            label=label,
            on_progress=lambda o, elapsed: logger.info(
                "%s 生成中... status=%s elapsed=%ds", label, o.get("task_status"), int(elapsed)
            ),
        )

    @staticmethod
    @with_retry_async(
        max_attempts=DOWNLOAD_MAX_ATTEMPTS,
        backoff_seconds=DOWNLOAD_BACKOFF_SECONDS,
        retry_if=should_retry_poll,
    )
    async def _download_with_retry(url: str, output_path: Path) -> None:
        await download_image_to_path(url, output_path)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
