"""LLM provider interface and shared types.

Defines the Protocol for LLM providers and the unified response dataclass.
This abstraction decouples the TaskManager from any specific LLM SDK,
allowing future swapping to an LLM Gateway or other provider.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


@dataclass
class LLMResponse:
    """Unified response from any LLM provider.

    Maps provider-specific responses into a common format
    consumed by TaskManager and other components.
    """

    content: str
    session_id: Optional[str]
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool
    error_message: Optional[str] = None


class LLMProvider(Protocol):
    """Protocol defining the LLM provider interface.

    Any LLM backend (Claude SDK, LLM Gateway, etc.) must implement
    this protocol to be used by the background task system.
    """

    async def execute(
        self,
        prompt: str,
        working_dir: Path,
        user_id: int,
        session_id: Optional[str] = None,
        stream_callback: Optional[Callable[..., Any]] = None,
        force_new: bool = False,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a prompt against the LLM provider.

        Args:
            prompt: The user prompt to send.
            working_dir: Working directory for the execution context.
            user_id: Telegram user ID for session tracking.
            session_id: Optional session ID to resume a conversation.
            stream_callback: Optional callback for streaming updates.
            force_new: If True, force a new session instead of resuming.
            model: Optional model ID override (e.g., "claude-opus-4-6").

        Returns:
            LLMResponse with the execution result.
        """
        ...

    async def healthcheck(self) -> bool:
        """Check if the provider is healthy and reachable.

        Returns:
            True if the provider is operational, False otherwise.
        """
        ...
