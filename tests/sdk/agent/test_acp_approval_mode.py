"""Tests for the configurable ACP approval mode.

``ACPAgent.acp_approval_mode`` controls how the per-action "Continue?" ACP
permission request is answered: prompt the user (default), auto-approve all,
auto-approve only edit/read kinds, or reject all.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _auto_permission_decision,
    _select_permission_option_id,
)


def _make_agent(**kwargs) -> ACPAgent:
    return ACPAgent(acp_command=["echo", "test"], **kwargs)


class _Opt:
    def __init__(self, option_id: str, kind: str):
        self.option_id = option_id
        self.kind = kind


def _options():
    return [
        _Opt("allow-1", "allow_once"),
        _Opt("reject-1", "reject_once"),
    ]


def test_ask_always_defers_to_ui():
    assert _auto_permission_decision("ask_always", "execute") is None
    assert _auto_permission_decision("ask_always", "edit") is None


def test_auto_approve_all_approves_every_kind():
    assert _auto_permission_decision("auto_approve_all", "execute") is True
    assert _auto_permission_decision("auto_approve_all", "edit") is True
    assert _auto_permission_decision("auto_approve_all", None) is True


def test_reject_all_rejects_every_kind():
    assert _auto_permission_decision("reject_all", "execute") is False
    assert _auto_permission_decision("reject_all", "edit") is False


def test_edits_only_approves_edit_and_read_but_asks_for_execute():
    assert _auto_permission_decision("auto_approve_edits_only", "edit") is True
    assert _auto_permission_decision("auto_approve_edits_only", "read") is True
    assert _auto_permission_decision("auto_approve_edits_only", "execute") is None
    assert _auto_permission_decision("auto_approve_edits_only", "fetch") is None


def test_unknown_mode_defers_to_ui():
    assert _auto_permission_decision("nonsense", "edit") is None


def test_decision_maps_to_allow_and_reject_option_ids():
    opts = _options()
    assert _select_permission_option_id(opts, accept=True) == "allow-1"
    assert _select_permission_option_id(opts, accept=False) == "reject-1"


def test_outcome_response_shapes():
    # Sanity that the schema outcome types we rely on construct cleanly.
    allowed = AllowedOutcome(outcome="selected", option_id="allow-1")
    denied = DeniedOutcome(outcome="cancelled")
    assert allowed.option_id == "allow-1"
    assert denied.outcome == "cancelled"


def test_default_approval_mode_is_ask_always():
    assert _make_agent().acp_approval_mode == "ask_always"


@pytest.mark.asyncio
async def test_handle_permission_request_auto_approves_without_ui():
    agent = _make_agent(acp_approval_mode="auto_approve_all")
    events: list = []
    state = MagicMock()
    tool_call = SimpleNamespace(kind="execute", title="bash", tool_call_id="t1")

    response = await agent._handle_permission_request(
        _options(),
        "session-1",
        tool_call,
        state,
        events.append,
    )

    assert isinstance(response.outcome, AllowedOutcome)
    assert response.outcome.option_id == "allow-1"
    # No pending event emitted, and WAITING state never set (state untouched).
    assert events == []
    state.__enter__.assert_not_called()


@pytest.mark.asyncio
async def test_handle_permission_request_rejects_under_reject_all():
    agent = _make_agent(acp_approval_mode="reject_all")
    events: list = []
    tool_call = SimpleNamespace(kind="edit", title="edit", tool_call_id="t2")

    response = await agent._handle_permission_request(
        _options(),
        "session-1",
        tool_call,
        MagicMock(),
        events.append,
    )

    # reject_all selects the reject *option* — ACP signals refusal by choosing
    # the reject_once option, not by a transport-level cancel.
    assert isinstance(response.outcome, AllowedOutcome)
    assert response.outcome.option_id == "reject-1"
    assert events == []
