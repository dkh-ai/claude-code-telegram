"""Base protocol for assistant plugins."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class PluginResponse:
    """Response from a plugin handler."""

    content: str
    model: str = ""
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AssistantPlugin(Protocol):
    """Protocol for assistant plugins."""

    name: str
    description: str
    patterns: list[re.Pattern[str]]
    model: str

    async def can_handle(self, message: str, context: dict[str, Any]) -> float:
        """Return confidence 0.0-1.0 that this plugin handles the message."""
        ...

    async def handle(
        self,
        message: str,
        context: dict[str, Any],
        chat_provider: Any,
    ) -> PluginResponse:
        """Process the message and return response."""
        ...
