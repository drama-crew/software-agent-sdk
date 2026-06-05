"""Tests for the per-turn ACP output soft-fuse (P0-2).

The bridge meters accumulated output bytes per turn and, when the configured
``max_turn_output_bytes`` is exceeded, fires ``on_output_cap_exceeded`` exactly
once so the agent can request ``session/cancel`` and stop a runaway turn before
it OOMs the sandbox.
"""

import pytest
from acp.schema import AgentMessageChunk, TextContentBlock

from openhands.sdk.agent.acp_agent import _OpenHandsACPBridge


def _chunk(n: int) -> AgentMessageChunk:
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=TextContentBlock(type="text", text="x" * n),
    )


@pytest.mark.asyncio
async def test_output_cap_triggers_cancel_callback_once():
    bridge = _OpenHandsACPBridge()
    bridge.max_turn_output_bytes = 100
    fired = []
    bridge.on_output_cap_exceeded = lambda: fired.append(True)
    await bridge.session_update("s1", _chunk(60))
    assert fired == []  # 60 < 100
    await bridge.session_update("s1", _chunk(60))
    assert fired == [True]  # 120 > 100
    await bridge.session_update("s1", _chunk(60))
    assert fired == [True]  # not re-triggered


@pytest.mark.asyncio
async def test_output_cap_disabled_when_zero():
    bridge = _OpenHandsACPBridge()
    bridge.max_turn_output_bytes = 0
    fired = []
    bridge.on_output_cap_exceeded = lambda: fired.append(True)
    await bridge.session_update("s1", _chunk(10_000))
    assert fired == []


@pytest.mark.asyncio
async def test_reset_clears_output_counter():
    bridge = _OpenHandsACPBridge()
    bridge.max_turn_output_bytes = 100
    fired = []
    bridge.on_output_cap_exceeded = lambda: fired.append(True)
    await bridge.session_update("s1", _chunk(80))
    bridge.reset()
    await bridge.session_update("s1", _chunk(80))  # re-counts after reset
    assert fired == []
