"""Claude SDK LLM provider implementation.

Wraps the existing ClaudeIntegration facade to conform to the
LLMProvider protocol, mapping between ClaudeResponse and LLMResponse.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import structlog

from .interface import LLMResponse

logger = structlog.get_logger()


class ClaudeProvider:
    """LLM provider backed by ClaudeIntegration (Claude Code SDK).

    Adapts ClaudeIntegration.run_command() to the LLMProvider protocol,
    translating between ClaudeResponse and LLMResponse.
    """

    def __init__(self, claude_integration: Any) -> None:
        """Initialize with an existing ClaudeIntegration instance.

        Args:
            claude_integration: A ClaudeIntegration facade instance.
        """
        self._claude_integration = claude_integration

    async def execute(
        self,
        prompt: str,
        working_dir: Path,
        user_id: int,
        session_id: Optional[str] = None,
        stream_callback: Optional[Callable[..., Any]] = None,
        force_new: bool = False,
    ) -> LLMResponse:
        """Execute a prompt via ClaudeIntegration.

        Maps the LLMProvider interface to ClaudeIntegration.run_command(),
        translating the ClaudeResponse back into an LLMResponse.

        On exception, returns an LLMResponse with is_error=True instead
        of propagating the error, so callers get a uniform result type.
        """
        try:
            response = await self._claude_integration.run_command(
                prompt=prompt,
                working_directory=working_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=stream_callback,
                force_new=force_new,
            )

            return LLMResponse(
                content=response.content,
                session_id=response.session_id,
                cost=response.cost,
                duration_ms=response.duration_ms,
                num_turns=response.num_turns,
                is_error=response.is_error,
            )

        except Exception as exc:
            logger.error(
                "ClaudeProvider execution failed",
                error=str(exc),
                user_id=user_id,
                working_dir=str(working_dir),
            )
            return LLMResponse(
                content="",
                session_id=None,
                cost=0.0,
                duration_ms=0,
                num_turns=0,
                is_error=True,
                error_message=str(exc),
            )

    async def healthcheck(self) -> bool:
        """Check provider health.

        The Claude SDK does not expose a dedicated health endpoint,
        so this returns True as a simple liveness indicator.
        """
        return True
