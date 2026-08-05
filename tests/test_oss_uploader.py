"""OSSUploader 单元测试（mock oss2.Bucket，不触网）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.oss_uploader import OSSConfig, OSSUploader


def _cfg() -> OSSConfig:
    return OSSConfig(
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        bucket="bkt",
        access_key_id="id",
        access_key_secret="sec",
        upload_prefix="arcreel",
    )


def _file(tmp_path: Path) -> Path:
    p = tmp_path / "a.png"
    p.write_bytes(b"png-bytes")
    return p


class TestUploadFile:
    def test_default_mode_returns_signed_url(self, tmp_path: Path):
        bucket = MagicMock()
        bucket.sign_url.return_value = "https://bkt.oss-cn-hangzhou.aliyuncs.com/arcreel/x.png?Signature=sig"
        with patch("lib.oss_uploader.oss2.Bucket", return_value=bucket):
            uploader = OSSUploader(_cfg())
            url = uploader.upload_file(_file(tmp_path))

        assert "Signature=" in url
        # 默认模式不带 ACL header
        _, kwargs = bucket.put_object_from_file.call_args
        assert "headers" not in kwargs
        bucket.sign_url.assert_called_once()

    def test_public_read_mode_returns_plain_url_with_acl(self, tmp_path: Path):
        bucket = MagicMock()
        with patch("lib.oss_uploader.oss2.Bucket", return_value=bucket):
            uploader = OSSUploader(_cfg(), public_read=True)
            url = uploader.upload_file(_file(tmp_path))

        # 裸 URL:https://{bucket}.{endpoint_host}/{prefix}/{uuid}.png,无签名 query
        assert url.startswith("https://bkt.oss-cn-hangzhou.aliyuncs.com/arcreel/")
        assert url.endswith(".png")
        assert "?" not in url
        # 上传带对象级 public-read ACL,且不走签名
        _, kwargs = bucket.put_object_from_file.call_args
        assert kwargs["headers"] == {"x-oss-object-acl": "public-read"}
        bucket.sign_url.assert_not_called()

    def test_incomplete_config_raises(self):
        with pytest.raises(ValueError, match="OSS 配置不完整"):
            OSSUploader(OSSConfig(endpoint="", bucket="b", access_key_id="i", access_key_secret="s"))
