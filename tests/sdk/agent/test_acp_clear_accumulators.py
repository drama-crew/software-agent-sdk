"""Tests for clearing ACP accumulation lists after a turn (P2).

``_clear_accumulators`` drops the per-turn text/thought/tool-call buffers so a
runaway turn's data does not linger into the next turn, while leaving the live
callbacks wired (unlike ``reset()``).
"""

from openhands.sdk.agent.acp_agent import _OpenHandsACPBridge


def test_clear_accumulators_clears_lists_only():
    bridge = _OpenHandsACPBridge()
    bridge.accumulated_text.append("a")
    bridge.accumulated_thoughts.append("t")
    bridge.accumulated_tool_calls.append({"tool_call_id": "1"})
    sentinel = object()
    bridge.on_event = sentinel
    bridge._clear_accumulators()
    assert bridge.accumulated_text == []
    assert bridge.accumulated_thoughts == []
    assert bridge.accumulated_tool_calls == []
    assert bridge.on_event is sentinel  # callbacks preserved (not reset())
