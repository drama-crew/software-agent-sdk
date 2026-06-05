import asyncio

import pytest
from acp.task import RpcTask, RpcTaskKind
from acp.task.queue import InMemoryMessageQueue
from acp.task.state import InMemoryMessageStateStore
from acp.task.supervisor import TaskSupervisor

from openhands.sdk.agent.acp_backpressure import (
    BoundedMessageDispatcher,
    make_bounded_dispatcher_factory,
)


@pytest.mark.asyncio
async def test_notifications_are_bounded_not_fire_and_forget():
    queue = InMemoryMessageQueue(maxsize=0)
    supervisor = TaskSupervisor(source="test")
    store = InMemoryMessageStateStore()
    started = 0
    release = asyncio.Event()

    async def slow_notification(message):
        nonlocal started
        started += 1
        await release.wait()

    async def noop_request(message):
        return {}

    disp = BoundedMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=noop_request,
        notification_runner=slow_notification,
        max_concurrent_notifications=2,
    )
    disp.start()
    for _ in range(5):
        await queue.publish(RpcTask(RpcTaskKind.NOTIFICATION, {"method": "x"}))
    await asyncio.sleep(0.05)
    assert started == 2, f"expected 2 concurrent, got {started}"
    release.set()
    await asyncio.sleep(0.05)
    assert started == 5
    await disp.stop()


@pytest.mark.asyncio
async def test_requests_not_blocked_by_slow_notifications():
    queue = InMemoryMessageQueue(maxsize=0)
    supervisor = TaskSupervisor(source="test")
    store = InMemoryMessageStateStore()
    release = asyncio.Event()
    req_done = asyncio.Event()

    async def slow_notification(message):
        await release.wait()

    async def fast_request(message):
        req_done.set()
        return {"ok": True}

    disp = BoundedMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=fast_request,
        notification_runner=slow_notification,
        max_concurrent_notifications=1,
    )
    disp.start()
    await queue.publish(RpcTask(RpcTaskKind.NOTIFICATION, {"method": "n"}))
    await asyncio.sleep(0.01)
    await queue.publish(RpcTask(RpcTaskKind.REQUEST, {"method": "r", "id": 1}))
    await asyncio.wait_for(req_done.wait(), timeout=1.0)
    release.set()
    await disp.stop()


@pytest.mark.asyncio
async def test_semaphore_released_when_notification_raises():
    """A notification raising must still release its slot, else slots leak and
    the dispatcher would wedge after one error."""
    queue = InMemoryMessageQueue(maxsize=0)
    supervisor = TaskSupervisor(source="test")
    store = InMemoryMessageStateStore()
    calls = 0
    second_started = asyncio.Event()

    async def notification(message):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")  # first one fails
        second_started.set()  # second running proves the slot was released

    async def noop_request(message):
        return {}

    disp = BoundedMessageDispatcher(
        queue=queue,
        supervisor=supervisor,
        store=store,
        request_runner=noop_request,
        notification_runner=notification,
        max_concurrent_notifications=1,  # single slot: leak would wedge
    )
    disp.start()
    await queue.publish(RpcTask(RpcTaskKind.NOTIFICATION, {"method": "a"}))
    await queue.publish(RpcTask(RpcTaskKind.NOTIFICATION, {"method": "b"}))
    await asyncio.wait_for(second_started.wait(), timeout=1.0)
    await disp.stop()


def test_factory_is_callable():
    assert callable(make_bounded_dispatcher_factory(max_concurrent_notifications=4))
