"""Bounded ACP message dispatcher: cap notification concurrency to apply
backpressure on a runaway agent stream, preventing unbounded memory growth.

The default acp dispatcher fires a new fire-and-forget task per notification
with no concurrency cap, so a degenerate chunk flood (~360/s) piles up tasks +
message copies until OOM. We cap notification concurrency with a semaphore
acquired *inside the dispatch loop*: when the cap is hit the loop blocks, the
(bounded) queue fills, and publish() blocks the receive loop -- backpressure all
the way to the subprocess stdout. Requests are not gated by this semaphore, so
once dispatched they run concurrently; under sustained notification saturation a
request may queue behind in-flight notifications until a slot frees, but cannot
deadlock.
"""

from __future__ import annotations

import asyncio
from typing import Any

from acp.task.dispatcher import (
    DefaultMessageDispatcher,
    MessageDispatcher,
    NotificationRunner,
    RequestRunner,
)
from acp.task.queue import MessageQueue
from acp.task.state import MessageStateStore
from acp.task.supervisor import TaskSupervisor


DEFAULT_MAX_CONCURRENT_NOTIFICATIONS = 8


class BoundedMessageDispatcher(DefaultMessageDispatcher):
    """DefaultMessageDispatcher with a semaphore-bounded notification path."""

    def __init__(
        self,
        *,
        queue: MessageQueue,
        supervisor: TaskSupervisor,
        store: MessageStateStore,
        request_runner: RequestRunner,
        notification_runner: NotificationRunner,
        max_concurrent_notifications: int = DEFAULT_MAX_CONCURRENT_NOTIFICATIONS,
    ) -> None:
        super().__init__(
            queue=queue,
            supervisor=supervisor,
            store=store,
            request_runner=request_runner,
            notification_runner=notification_runner,
        )
        self._notif_sem = asyncio.Semaphore(max(1, max_concurrent_notifications))

    async def _dispatch_notification(self, message: dict[str, Any]) -> None:
        # Acquire *inside* the dispatch loop: when the cap is hit this blocks
        # the loop, so the (bounded) queue fills and publish() blocks the
        # receive loop -- propagating backpressure to the subprocess stdout.
        await self._notif_sem.acquire()

        async def runner() -> None:
            try:
                await self._notification_runner(message)
            finally:
                self._notif_sem.release()

        self._supervisor.create(runner(), name="acp.Dispatcher.notification")


def make_bounded_dispatcher_factory(
    *, max_concurrent_notifications: int = DEFAULT_MAX_CONCURRENT_NOTIFICATIONS
):
    def factory(
        queue: MessageQueue,
        supervisor: TaskSupervisor,
        store: MessageStateStore,
        request_runner: RequestRunner,
        notification_runner: NotificationRunner,
    ) -> MessageDispatcher:
        return BoundedMessageDispatcher(
            queue=queue,
            supervisor=supervisor,
            store=store,
            request_runner=request_runner,
            notification_runner=notification_runner,
            max_concurrent_notifications=max_concurrent_notifications,
        )

    return factory
