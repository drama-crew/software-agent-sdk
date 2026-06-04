"""Tests for ACP plan / available-commands events reaching the event stream.

These ACP session updates were previously dropped in the bridge's
``session_update`` else-branch; they now surface as ``AgentPlanEvent`` /
``AvailableCommandsEvent`` so the frontend can render plans and command lists.
"""

from __future__ import annotations

import pytest
from acp.schema import (
    AgentPlanUpdate,
    AvailableCommand,
    AvailableCommandsUpdate,
    PlanEntry,
)

from openhands.sdk.agent.acp_agent import _OpenHandsACPBridge
from openhands.sdk.event import (
    AgentPlanEvent,
    AvailableCommandsEvent,
)


@pytest.mark.asyncio
async def test_session_update_emits_agent_plan_event():
    client = _OpenHandsACPBridge()
    events: list = []
    client.on_event = events.append

    update = AgentPlanUpdate(
        session_update="plan",
        entries=[
            PlanEntry(content="Write the outline", priority="high", status="pending"),
            PlanEntry(
                content="Draft scene one", priority="medium", status="in_progress"
            ),
        ],
    )

    await client.session_update("session-1", update)

    plan_events = [e for e in events if isinstance(e, AgentPlanEvent)]
    assert len(plan_events) == 1
    plan = plan_events[0]
    assert [entry.content for entry in plan.entries] == [
        "Write the outline",
        "Draft scene one",
    ]
    assert plan.entries[0].priority == "high"
    assert plan.entries[1].status == "in_progress"


@pytest.mark.asyncio
async def test_session_update_emits_available_commands_event():
    client = _OpenHandsACPBridge()
    events: list = []
    client.on_event = events.append

    update = AvailableCommandsUpdate(
        session_update="available_commands_update",
        available_commands=[
            AvailableCommand(name="create_plan", description="Create a plan"),
            AvailableCommand(name="research", description="Research the codebase"),
        ],
    )

    await client.session_update("session-1", update)

    cmd_events = [e for e in events if isinstance(e, AvailableCommandsEvent)]
    assert len(cmd_events) == 1
    cmds = cmd_events[0]
    assert [c.name for c in cmds.commands] == ["create_plan", "research"]
    assert cmds.commands[0].description == "Create a plan"


def test_agent_plan_event_json_roundtrip():
    from openhands.sdk.event import AgentPlanEntry

    event = AgentPlanEvent(
        entries=[AgentPlanEntry(content="x", priority="low", status="completed")]
    )

    dumped = event.model_dump(mode="json")
    assert dumped["kind"] == "AgentPlanEvent"

    restored = AgentPlanEvent.model_validate(dumped)
    assert restored.entries[0].content == "x"
    assert restored.entries[0].status == "completed"
