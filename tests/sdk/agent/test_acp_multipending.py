import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from acp.schema import AllowedOutcome

from openhands.sdk.agent.acp_agent import ACPAgent


def _agent():
    return ACPAgent(acp_command=["echo", "x"], acp_approval_mode="ask_always")


def _opts():
    return [
        SimpleNamespace(option_id="allow-1", kind="allow_once"),
        SimpleNamespace(option_id="reject-1", kind="reject_once"),
    ]


@pytest.mark.asyncio
async def test_two_concurrent_pending_do_not_cancel_each_other():
    agent = _agent()
    state = MagicMock()
    events = []
    tc_a = SimpleNamespace(kind="edit", title="A", tool_call_id="A")
    tc_b = SimpleNamespace(kind="execute", title="B", tool_call_id="B")

    fa = asyncio.create_task(
        agent._handle_permission_request(_opts(), "s", tc_a, state, events.append)
    )
    fb = asyncio.create_task(
        agent._handle_permission_request(_opts(), "s", tc_b, state, events.append)
    )
    await asyncio.sleep(0.05)

    # both pending, neither resolved/cancelled
    assert set(agent._pending_acp_permissions.keys()) == {"A", "B"}
    assert not fa.done() and not fb.done()

    # resolve just A
    agent.respond_to_pending_acp_permission(accept=True, tool_call_id="A")
    ra = await asyncio.wait_for(fa, 1)
    assert isinstance(ra.outcome, AllowedOutcome)
    assert not fb.done()
    assert set(agent._pending_acp_permissions.keys()) == {"B"}

    agent.respond_to_pending_acp_permission(accept=True, tool_call_id="B")
    await asyncio.wait_for(fb, 1)
    assert agent._pending_acp_permissions == {}
