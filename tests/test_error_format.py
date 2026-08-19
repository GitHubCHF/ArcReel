"""describe_exception 单元测试。"""

from __future__ import annotations

import httpx

from lib.error_format import describe_exception


def _http_status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/x")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"error {status}", request=request, response=response)


def test_http_error_with_body_prefers_body():
    exc = _http_status_error(400, '{"error":{"message":"invalid model"}}')
    msg = describe_exception(exc)
    assert msg == 'HTTP 400: {"error":{"message":"invalid model"}}'


def test_http_error_without_body_falls_back_to_str():
    exc = _http_status_error(500, "")
    msg = describe_exception(exc)
    assert "500" in msg
    assert not msg.startswith("HTTP 500: ")  # 无 body → 退回 str(exc)


def test_http_error_body_truncated():
    exc = _http_status_error(413, "x" * 5000)
    msg = describe_exception(exc)
    assert msg.startswith("HTTP 413: ")
    assert msg.endswith("…")
    assert len(msg) < 2000


def test_non_http_exception_uses_str():
    assert describe_exception(ValueError("boom")) == "boom"


def test_http_error_unread_stream_body_falls_back():
    """流式响应未读取时访问 .text 抛错,应吞掉并退回 str(exc)。"""
    request = httpx.Request("GET", "https://api.example.com/stream")
    response = httpx.Response(502, request=request, stream=httpx.ByteStream(b"later"))
    exc = httpx.HTTPStatusError("error 502", request=request, response=response)
    msg = describe_exception(exc)
    assert "502" in msg
