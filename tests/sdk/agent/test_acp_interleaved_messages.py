"""Text emitted between ACP tool calls must surface as interleaved MessageEvents.

Regression: the ACP bridge accumulated all assistant text for a turn and only
emitted it once at turn end (as ``FinishAction.message``), so narration the agent
produced *between* tool calls was collapsed to the end instead of rendering in
order. The bridge now flushes pending assistant text as a ``MessageEvent`` right
before each tool call, leaving only the trailing text for the ``FinishAction``.
"""

from __future__ import annotations

import pytest
from acp.schema import AgentMessageChunk, TextContentBlock, ToolCallStart

from openhands.sdk.agent.acp_agent import _OpenHandsACPBridge
from openhands.sdk.event import ACPToolCallEvent, MessageEvent


def _text(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=TextContentBlock(type="text", text=text),
    )


def _tool(tool_call_id: str, title: str) -> ToolCallStart:
    return ToolCallStart(
        session_update="tool_call", tool_call_id=tool_call_id, title=title
    )


def _msg_text(event: MessageEvent) -> str:
    return "".join(
        block.text
        for block in event.llm_message.content
        if getattr(block, "text", None) is not None
    )


@pytest.mark.asyncio
async def test_text_between_tool_calls_emits_interleaved_message_events() -> None:
    bridge = _OpenHandsACPBridge()
    events: list = []
    bridge.on_event = events.append

    await bridge.session_update("s", _text("Let me read the file."))
    await bridge.session_update("s", _tool("t1", "read"))
    await bridge.session_update("s", _text("Now I'll edit it."))
    await bridge.session_update("s", _tool("t2", "edit"))
    await bridge.session_update("s", _text("Done."))  # trailing -> FinishAction

    msgs = [e for e in events if isinstance(e, MessageEvent)]
    assert [_msg_text(m) for m in msgs] == [
        "Let me read the file.",
        "Now I'll edit it.",
    ], "text between tool calls must be emitted as interleaved MessageEvents"

    # Each text MessageEvent must precede the tool call it introduces.
    tools = [e for e in events if isinstance(e, ACPToolCallEvent)]
    assert len(tools) == 2
    assert events.index(msgs[0]) < events.index(tools[0])
    assert events.index(msgs[1]) < events.index(tools[1])

    # Trailing text (after the last tool call) is NOT emitted as a MessageEvent;
    # it is reserved for the end-of-turn FinishAction.
    assert "Done." not in [_msg_text(m) for m in msgs]


@pytest.mark.asyncio
async def test_take_unflushed_text_returns_only_trailing_segment() -> None:
    bridge = _OpenHandsACPBridge()
    bridge.on_event = lambda e: None

    await bridge.session_update("s", _text("alpha "))
    await bridge.session_update("s", _tool("t1", "read"))
    await bridge.session_update("s", _text("beta"))

    # 'alpha ' was flushed before t1; only 'beta' remains for the FinishAction.
    assert bridge.take_unflushed_text() == "beta"
    # Full accumulated text remains available for cost / token accounting.
    assert "".join(bridge.accumulated_text) == "alpha beta"


@pytest.mark.asyncio
async def test_no_tool_calls_keeps_all_text_for_finish() -> None:
    """A pure-text turn (no tool calls) emits no interleaved MessageEvents;
    all text stays unflushed for the FinishAction (unchanged behavior)."""
    bridge = _OpenHandsACPBridge()
    events: list = []
    bridge.on_event = events.append

    await bridge.session_update("s", _text("Just an answer, no tools."))

    assert [e for e in events if isinstance(e, MessageEvent)] == []
    assert bridge.take_unflushed_text() == "Just an answer, no tools."
