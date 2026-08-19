"""异常 → 面向用户错误信息的格式化。

任务失败信息(工作区通知)默认取 ``str(exc)``,但 ``httpx.HTTPStatusError`` 的 str 只有
"Client error '400 Bad Request' for url ..." —— 上游真正的原因(``{"error":{"message":...}}``
之类)在 response body 里。有 body 就优先展示 body。
"""

from __future__ import annotations

import httpx

# body 截断上限(task.error_message 落库前还会再 [:2000],此处先收口避免超长网关 HTML 刷屏)。
_MAX_BODY_CHARS = 1500


def describe_exception(exc: BaseException) -> str:
    """把异常转成给用户看的错误信息。

    httpx.HTTPStatusError:有 response body 时返回 ``HTTP <status>: <body>``,否则退回 str(exc)。
    其它异常:原样 str(exc)。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = (exc.response.text or "").strip()
        except Exception:
            # 流式响应未读取等场景访问 .text 会抛,吞掉退回 str
            body = ""
        if body:
            if len(body) > _MAX_BODY_CHARS:
                body = body[:_MAX_BODY_CHARS] + "…"
            return f"HTTP {exc.response.status_code}: {body}"
    return str(exc)
