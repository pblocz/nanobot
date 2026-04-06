"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import AgentDefaults

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


def _make_loop(*, exec_config=None):
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace, exec_config=exec_config)
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    def test_exec_tool_not_registered_when_disabled(self):
        from nanobot.config.schema import ExecToolConfig

        loop, _bus = _make_loop(exec_config=ExecToolConfig(enable=False))

        assert loop.tools.get("exec") is None

    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_dispatch_streaming_preserves_message_metadata(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(
            channel="matrix",
            sender_id="u1",
            chat_id="!room:matrix.org",
            content="hello",
            metadata={
                "_wants_stream": True,
                "thread_root_event_id": "$root1",
                "thread_reply_to_event_id": "$reply1",
            },
        )

        async def fake_process(_msg, *, on_stream=None, on_stream_end=None, **kwargs):
            assert on_stream is not None
            assert on_stream_end is not None
            await on_stream("hi")
            await on_stream_end(resuming=False)
            return None

        loop._process_message = fake_process

        await loop._dispatch(msg)
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        second = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert first.metadata["thread_root_event_id"] == "$root1"
        assert first.metadata["thread_reply_to_event_id"] == "$reply1"
        assert first.metadata["_stream_delta"] is True
        assert second.metadata["thread_root_event_id"] == "$root1"
        assert second.metadata["thread_reply_to_event_id"] == "$reply1"
        assert second.metadata["_stream_end"] is True

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]


class TestSubagentCancellation:
    @pytest.mark.asyncio
    async def test_cancel_by_session(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(
            provider=provider,
            workspace=MagicMock(),
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        count = await mgr.cancel_by_session("test:c1")
        assert count == 1
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_by_session_no_tasks(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(
            provider=provider,
            workspace=MagicMock(),
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )
        assert await mgr.cancel_by_session("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_subagent_preserves_reasoning_fields_in_tool_turn(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        captured_second_call: list[dict] = []

        call_count = {"n": 0}

        async def scripted_chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="thinking",
                    tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                    reasoning_content="hidden reasoning",
                    thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                )
            captured_second_call[:] = messages
            return LLMResponse(content="done", tool_calls=[])
        provider.chat_with_retry = scripted_chat_with_retry
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )

        async def fake_execute(self, **kwargs):
            return "tool result"

        monkeypatch.setattr("nanobot.agent.tools.filesystem.ListDirTool.execute", fake_execute)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        assistant_messages = [
            msg for msg in captured_second_call
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
        assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]

    @pytest.mark.asyncio
    async def test_subagent_exec_tool_not_registered_when_disabled(self, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.config.schema import ExecToolConfig

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            exec_config=ExecToolConfig(enable=False),
        )
        mgr._announce_result = AsyncMock()

        async def fake_run(spec):
            assert spec.tools.get("exec") is None
            return SimpleNamespace(
                stop_reason="done",
                final_content="done",
                error=None,
                tool_events=[],
            )

        mgr.runner.run = AsyncMock(side_effect=fake_run)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        mgr.runner.run.assert_awaited_once()
        mgr._announce_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subagent_announces_error_when_tool_execution_fails(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="thinking",
            tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
        ))
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )
        mgr._announce_result = AsyncMock()

        calls = {"n": 0}

        async def fake_execute(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "first result"
            raise RuntimeError("boom")

        monkeypatch.setattr("nanobot.agent.tools.filesystem.ListDirTool.execute", fake_execute)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        mgr._announce_result.assert_awaited_once()
        args = mgr._announce_result.await_args.args
        assert "Completed steps:" in args[3]
        assert "- list_dir: first result" in args[3]
        assert "Failure:" in args[3]
        assert "- list_dir: boom" in args[3]
        assert args[5] == "error"

    @pytest.mark.asyncio
    async def test_cancel_by_session_cancels_running_subagent_tool(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="thinking",
            tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
        ))
        mgr = SubagentManager(
            provider=provider,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        )
        mgr._announce_result = AsyncMock()

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_execute(self, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr("nanobot.agent.tools.filesystem.ListDirTool.execute", fake_execute)

        task = asyncio.create_task(
            mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})
        )
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        await asyncio.wait_for(started.wait(), timeout=1.0)

        count = await mgr.cancel_by_session("test:c1")

        assert count == 1
        assert cancelled.is_set()
        assert task.cancelled()
        mgr._announce_result.assert_not_awaited()


from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_real_loop(tmp_path):
    """AgentLoop backed by a real SessionManager for session-persistence tests."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import GenerationSettings

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "mock")

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


class TestStopPreservesSession:
    """Verify that /stop saves the user message to the session."""

    @pytest.mark.asyncio
    async def test_stop_during_tool_execution_leaves_no_orphan(self, tmp_path):
        """Cancel while the FIRST tool is executing must not leave an orphaned tool call.

        AgentRunner.run() starts with messages = list(spec.initial_messages), so
        initial_messages is never mutated regardless of what happens inside the runner.
        The _partial checkpoint therefore always holds a clean user-message-only snapshot.
        """
        loop = _make_real_loop(tmp_path)

        tool_started = asyncio.Event()

        async def scripted_chat(**kwargs):
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="slow_tool", arguments={})],
            )

        loop.runner.provider.chat_with_retry = scripted_chat
        loop.runner.provider.chat_stream_with_retry = scripted_chat

        async def slow_tool(name, arguments):
            tool_started.set()
            await asyncio.sleep(3600)

        loop.tools.execute = slow_tool

        from nanobot.bus.events import InboundMessage

        msg = InboundMessage(
            channel="cli", sender_id="user", chat_id="direct",
            content="What is the capital of France?",
        )

        task = asyncio.create_task(loop._dispatch(msg))
        await asyncio.wait_for(tool_started.wait(), timeout=2.0)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        session = loop.sessions.get_or_create(msg.session_key)
        roles = [m["role"] for m in session.messages]
        contents = [m.get("content", "") for m in session.messages]

        assert "user" in roles, "user message was not saved after /stop"
        assert any("France" in str(c) for c in contents), "user message content not saved"

        # No orphaned assistant tool_calls (assistant with tool_calls but no matching result)
        saved = session.messages
        tool_result_ids = {m.get("tool_call_id") for m in saved if m.get("role") == "tool"}
        orphan = [
            m for m in saved
            if m.get("role") == "assistant"
            and m.get("tool_calls")
            and not all(
                tc.get("id") in tool_result_ids
                for tc in (m.get("tool_calls") or [])
                if isinstance(tc, dict)
            )
        ]
        assert not orphan, (
            "orphaned assistant tool_call in session after /stop — "
            "would cause 'tool call result does not follow tool call' on next request"
        )

    @pytest.mark.asyncio
    async def test_stop_preserves_user_message(self, tmp_path):
        """Cancelling an in-flight task (during LLM call) must still save the user's message."""
        loop = _make_real_loop(tmp_path)

        ready = asyncio.Event()

        async def hanging_chat(**kwargs):
            ready.set()
            await asyncio.sleep(3600)

        loop.runner.provider.chat_with_retry = hanging_chat
        loop.runner.provider.chat_stream_with_retry = hanging_chat

        from nanobot.bus.events import InboundMessage

        msg = InboundMessage(
            channel="cli", sender_id="user", chat_id="direct",
            content="What is the capital of France?",
        )

        task = asyncio.create_task(loop._dispatch(msg))
        await asyncio.wait_for(ready.wait(), timeout=2.0)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        session = loop.sessions.get_or_create(msg.session_key)
        roles = [m["role"] for m in session.messages]
        contents = [m.get("content", "") for m in session.messages]
        assert "user" in roles, "user message was not saved after /stop"
        assert any("France" in str(c) for c in contents), "user message content not saved"
