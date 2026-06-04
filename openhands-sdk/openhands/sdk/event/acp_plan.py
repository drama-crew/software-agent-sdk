"""ACP plan / available-commands events.

OpenCode and other ACP servers emit ``AgentPlanUpdate`` (a plan/todo list) and
``AvailableCommandsUpdate`` (slash-style commands the agent can run) as session
updates. These mirror ``ACPToolCallEvent`` so they reach the OpenHands event
stream and the frontend chat instead of being dropped. Neither participates in
LLM message conversion, so both subclass ``Event`` (not ``LLMConvertibleEvent``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class AgentPlanEntry(BaseModel):
    """A single task in an agent plan/todo list (ACP ``PlanEntry``)."""

    content: str
    priority: str | None = None
    status: str | None = None


class AgentPlanEvent(Event):
    """Event representing an agent plan/todo list (ACP ``AgentPlanUpdate``).

    The agent re-emits the full plan on every change, so consumers should treat
    the latest ``AgentPlanEvent`` as authoritative and replace any prior plan.
    """

    source: SourceType = "agent"
    entries: list[AgentPlanEntry] = Field(default_factory=list)

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Plan", style="bold")
        for entry in self.entries:
            marker = {
                "completed": "[x]",
                "in_progress": "[~]",
                "pending": "[ ]",
            }.get(entry.status or "", "[ ]")
            content.append(f"\n{marker} {entry.content}")
            if entry.priority:
                content.append(f" ({entry.priority})", style="dim")
        return content

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ({self.source}): "
            f"{len(self.entries)} task(s)"
        )


class AvailableCommandInfo(BaseModel):
    """A command the agent can run (ACP ``AvailableCommand``)."""

    name: str
    description: str | None = None


class AvailableCommandsEvent(Event):
    """Event listing the commands an ACP agent exposes.

    Re-emitted whenever the command set changes; consumers should replace any
    prior list with the latest event.
    """

    source: SourceType = "agent"
    commands: list[AvailableCommandInfo] = Field(default_factory=list)

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Available commands", style="bold")
        for cmd in self.commands:
            content.append(f"\n/{cmd.name}", style="bold")
            if cmd.description:
                content.append(f" — {cmd.description}", style="dim")
        return content

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} ({self.source}): "
            f"{len(self.commands)} command(s)"
        )
