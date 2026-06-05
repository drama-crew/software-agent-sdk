"""Tests for the per-turn ACP output soft-fuse (P0-2).

The bridge meters accumulated output bytes per turn and, when the configured
``max_turn_output_bytes`` is exceeded, fires ``on_output_cap_exceeded`` exactly
once so the agent can request ``session/cancel`` and stop a runaway turn before
it OOMs the sandbox.
"""

import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import AgentMessageChunk, TextContentBlock

from openhands.sdk.agent.acp_agent import ACPAgent, _OpenHandsACPBridge
from openhands.sdk.utils.async_executor import AsyncExecutor


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


def _make_cancel_agent(executor: AsyncExecutor) -> tuple[ACPAgent, MagicMock]:
    """Wire an ACPAgent with a mock ``_conn`` whose ``cancel`` is observable.

    Mirrors the production wiring of ``_request_session_cancel``: a live
    ``_executor`` (real portal), a connection with an awaitable ``cancel``,
    and a session id. Returns the agent and its ``cancel`` mock so callers
    can assert it actually ran.
    """
    agent = ACPAgent(acp_command=["echo", "test"])
    conn = MagicMock()
    conn.cancel = AsyncMock(return_value=None)
    agent._conn = conn
    agent._executor = executor
    agent._session_id = "test-session"
    return agent, conn.cancel


def test_request_session_cancel_from_event_loop_thread_fires_cancel():
    """The real OOM bug: the output-cap callback runs ``_request_session_cancel``
    on the **portal event-loop thread** (``session_update`` is dispatched there,
    and ``_account_output_bytes`` calls the callback inline). With the buggy
    ``portal.start_task_soon(_cancel)`` body, anyio raises
    ``RuntimeError("This method cannot be called from the event loop thread")``,
    which the surrounding ``except Exception`` swallows — so ``conn.cancel`` is
    NEVER sent and the runaway turn keeps streaming until OOM.

    This test schedules ``_request_session_cancel`` to execute *on* the portal
    loop thread (exactly the cap-callback context) and asserts ``conn.cancel``
    actually fires. It FAILS pre-fix (cancel never called) and passes once the
    scheduler detects a running loop and uses ``loop.create_task``.
    """
    executor = AsyncExecutor()
    try:
        agent, cancel_mock = _make_cancel_agent(executor)

        # Run _request_session_cancel ON the portal loop thread — this is the
        # production cap-callback path (session_update dispatched by that loop).
        executor.portal.start_task_soon(_async_wrap(agent._request_session_cancel))

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cancel_mock.await_count > 0 or cancel_mock.call_count > 0:
                break
            time.sleep(0.01)

        assert cancel_mock.call_count == 1, (
            "conn.cancel was never sent: _request_session_cancel scheduled the "
            "cancel via portal.start_task_soon from the event-loop thread, where "
            "anyio raises RuntimeError that the bare except swallows."
        )
        cancel_mock.assert_awaited_once_with("test-session")
    finally:
        executor.close()


def _async_wrap(fn):
    """Wrap a sync callable as a coroutine fn for ``start_task_soon``.

    ``portal.start_task_soon`` schedules a coroutine *on the loop thread*; the
    body invokes ``fn`` there, reproducing the cap-callback execution context.
    """

    async def _runner() -> None:
        fn()

    return _runner


def test_request_session_cancel_from_caller_thread_fires_cancel():
    """The fallback path must keep working: ``_request_session_cancel`` also has
    a legitimate non-loop caller (sync ``step``'s ``TimeoutError`` branch runs
    on the caller thread, not the portal loop). From there ``asyncio`` has no
    running loop, so the fix falls back to ``portal.start_task_soon`` — which is
    the correct cross-thread hand-off. Assert ``conn.cancel`` still fires.
    """
    executor = AsyncExecutor()
    try:
        # Force the portal loop to exist (and to be a *different* thread).
        loop_thread_id = executor.run_async(_current_thread_id)
        assert loop_thread_id != threading.get_ident()

        agent, cancel_mock = _make_cancel_agent(executor)

        # Called directly from this (caller) thread — no running event loop here.
        agent._request_session_cancel()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cancel_mock.call_count > 0:
                break
            time.sleep(0.01)

        assert cancel_mock.call_count == 1
        cancel_mock.assert_awaited_once_with("test-session")
    finally:
        executor.close()


async def _current_thread_id() -> int:
    return threading.get_ident()
