"""视频生成确认门 PreToolUse hook 的单元测试。

hook 拦截 mcp__arcreel__generate_video*，复用 AskUserQuestion 通道弹网页确认：
- 用户点"确认生成" → continue_（放行，交 allow 规则批准）
- 点"取消" / 自定义输入 / 缺失答案 → deny
- 会话中断（future 抛异常）→ deny
- 非视频工具 → 直接放行
- 无会话（managed 为 None）→ fail-closed deny
"""

from __future__ import annotations

import asyncio

import pytest

from server.agent_runtime.session_manager import (
    _VIDEO_CONFIRM_TEXT,
    PendingQuestion,
)


class _FakeManaged:
    """只实现 hook 用到的两个方法。"""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.pending: dict[str, PendingQuestion] = {}

    def add_pending_question(self, payload: dict) -> PendingQuestion:
        future: asyncio.Future[dict[str, str]] = asyncio.get_running_loop().create_future()
        pending = PendingQuestion(
            question_id=payload["question_id"],
            payload=payload,
            answer_future=future,
        )
        self.pending[pending.question_id] = pending
        return pending

    def add_message(self, message: dict) -> None:
        self.messages.append(message)


async def _drive(hook, tool_name: str):
    """启动 hook，等它注册问题并挂起，返回 (task, pending)。"""
    task = asyncio.create_task(hook({"tool_name": tool_name, "tool_input": {}}, "tid", None))
    # 让 hook 跑到 await answer_future 处。
    for _ in range(10):
        await asyncio.sleep(0)
    return task


_VIDEO_TOOL = "mcp__arcreel__generate_video_episode"


class TestVideoConfirmHook:
    async def test_confirm_allows(self, session_manager) -> None:
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        task = await _drive(hook, _VIDEO_TOOL)

        assert len(fake.pending) == 1
        # 弹出的卡片已 broadcast 给前端。
        assert fake.messages and fake.messages[0]["type"] == "ask_user_question"

        pending = next(iter(fake.pending.values()))
        question = _VIDEO_CONFIRM_TEXT["zh"]["question"]
        pending.answer_future.set_result({question: _VIDEO_CONFIRM_TEXT["zh"]["confirm"]})

        result = await task
        assert result == {"continue_": True}

    async def test_cancel_denies(self, session_manager) -> None:
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        task = await _drive(hook, _VIDEO_TOOL)

        pending = next(iter(fake.pending.values()))
        question = _VIDEO_CONFIRM_TEXT["zh"]["question"]
        pending.answer_future.set_result({question: _VIDEO_CONFIRM_TEXT["zh"]["cancel"]})

        result = await task
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert result["hookSpecificOutput"]["permissionDecisionReason"] == _VIDEO_CONFIRM_TEXT["zh"]["deny_reason"]

    async def test_custom_answer_denies(self, session_manager) -> None:
        """非"确认"label（如用户走"其他"自定义输入）一律安全默认拒绝。"""
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        task = await _drive(hook, _VIDEO_TOOL)

        pending = next(iter(fake.pending.values()))
        question = _VIDEO_CONFIRM_TEXT["zh"]["question"]
        pending.answer_future.set_result({question: "随便写点别的"})

        result = await task
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_interrupt_denies(self, session_manager) -> None:
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        task = await _drive(hook, _VIDEO_TOOL)

        pending = next(iter(fake.pending.values()))
        pending.answer_future.set_exception(RuntimeError("session interrupted"))

        result = await task
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert result["hookSpecificOutput"]["permissionDecisionReason"] == _VIDEO_CONFIRM_TEXT["zh"]["interrupt_reason"]

    @pytest.mark.parametrize("tool_name", ["mcp__arcreel__generate_storyboards", "Read", "Bash"])
    async def test_non_video_tool_passthrough(self, session_manager, tool_name: str) -> None:
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        result = await hook({"tool_name": tool_name, "tool_input": {}}, "tid", None)
        assert result == {"continue_": True}
        assert fake.pending == {}

    async def test_no_session_fails_closed(self, session_manager) -> None:
        hook = session_manager._build_video_confirm_hook([None], "zh")
        result = await hook({"tool_name": _VIDEO_TOOL, "tool_input": {}}, "tid", None)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.parametrize(
        "tool_name",
        [
            "mcp__arcreel__generate_video_episode",
            "mcp__arcreel__generate_video_scene",
            "mcp__arcreel__generate_video_all",
            "mcp__arcreel__generate_video_selected",
        ],
    )
    async def test_all_four_video_tools_gated(self, session_manager, tool_name: str) -> None:
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "zh")
        task = await _drive(hook, tool_name)
        assert len(fake.pending) == 1
        # 收尾：放行避免悬挂 task。
        pending = next(iter(fake.pending.values()))
        question = _VIDEO_CONFIRM_TEXT["zh"]["question"]
        pending.answer_future.set_result({question: _VIDEO_CONFIRM_TEXT["zh"]["confirm"]})
        await task

    async def test_locale_fallback_to_zh(self, session_manager) -> None:
        """未知 locale 回退中文，不报错。"""
        fake = _FakeManaged()
        hook = session_manager._build_video_confirm_hook([fake], "fr")
        task = await _drive(hook, _VIDEO_TOOL)
        payload = fake.messages[0]
        assert payload["questions"][0]["question"] == _VIDEO_CONFIRM_TEXT["zh"]["question"]
        pending = next(iter(fake.pending.values()))
        pending.answer_future.set_result({_VIDEO_CONFIRM_TEXT["zh"]["question"]: _VIDEO_CONFIRM_TEXT["zh"]["confirm"]})
        await task
