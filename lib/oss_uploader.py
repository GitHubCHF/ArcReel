"""阿里云 OSS 上传工具。

供需要"本地图 → 公网可访问 URL"的第三方视频后端复用(如钛动 tecdo:其图片入参要求
URL,而平台自带的上传接口需单独开通)。默认上传到私有桶后返回带签名的临时 URL
(默认 2 小时有效),无需把桶设为公共读。

oss2 的网络调用是同步阻塞的,异步调用方需用 ``asyncio.to_thread`` 包裹 ``upload_file``。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import oss2

logger = logging.getLogger(__name__)

# 签名 URL 默认有效期(秒):覆盖一次视频生成的拉取窗口即可。
DEFAULT_SIGN_EXPIRES_SECONDS = 2 * 60 * 60

# oss2 默认 connect timeout 是 60s：endpoint 填错(如误填 -internal 内网域名)会干等一分钟。
# 收紧到 15s，让 misconfig 快速 fail 而非长时间卡住。
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15


@dataclass
class OSSConfig:
    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str
    upload_prefix: str = ""

    @property
    def is_complete(self) -> bool:
        return all((self.endpoint, self.bucket, self.access_key_id, self.access_key_secret))

    @classmethod
    def from_dict(cls, data: dict[str, str] | None) -> OSSConfig:
        data = data or {}
        return cls(
            endpoint=(data.get("endpoint") or "").strip(),
            bucket=(data.get("bucket") or "").strip(),
            access_key_id=(data.get("access_key_id") or "").strip(),
            access_key_secret=(data.get("access_key_secret") or "").strip(),
            upload_prefix=(data.get("upload_prefix") or "").strip(),
        )


class OSSUploader:
    """上传本地文件到阿里云 OSS 并返回临时签名 URL。"""

    def __init__(
        self,
        config: OSSConfig,
        *,
        sign_expires_seconds: int = DEFAULT_SIGN_EXPIRES_SECONDS,
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if not config.is_complete:
            raise ValueError("OSS 配置不完整:需要 endpoint / bucket / access_key_id / access_key_secret")
        self._config = config
        self._sign_expires = sign_expires_seconds
        self._endpoint = config.endpoint if "://" in config.endpoint else f"https://{config.endpoint}"
        auth = oss2.Auth(config.access_key_id, config.access_key_secret)
        self._bucket = oss2.Bucket(auth, self._endpoint, config.bucket, connect_timeout=connect_timeout_seconds)

    def upload_file(self, path: Path, *, key: str | None = None) -> str:
        """上传文件,返回 GET 签名 URL。同步阻塞,异步调用方请用 asyncio.to_thread 包裹。"""
        object_key = key or self._build_key(path)
        logger.info("OSS 上传开始: endpoint=%s bucket=%s key=%s", self._endpoint, self._config.bucket, object_key)
        self._bucket.put_object_from_file(object_key, str(path))
        url: str = self._bucket.sign_url("GET", object_key, self._sign_expires, slash_safe=True)
        logger.info("OSS 上传完成: key=%s", object_key)
        return url

    def _build_key(self, path: Path) -> str:
        name = f"{uuid.uuid4().hex}{path.suffix.lower()}"
        prefix = self._config.upload_prefix.strip().strip("/")
        return f"{prefix}/{name}" if prefix else name
