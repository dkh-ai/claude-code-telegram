"""Assistant dispatcher — routes messages to plugins."""

from typing import Any, Optional

import structlog

from ..llm.chat_pool import ChatProviderPool
from .base import PluginResponse
from .registry import PluginRegistry

logger = structlog.get_logger()


class AssistantDispatcher:
    """Route assistant-mode messages to the appropriate plugin."""

    def __init__(
        self,
        registry: PluginRegistry,
        chat_pool: ChatProviderPool,
    ) -> None:
        self._registry = registry
        self._chat_pool = chat_pool

    async def dispatch(
        self,
        message: str,
        context: dict[str, Any],
    ) -> Optional[PluginResponse]:
        """Dispatch a message to a matching plugin.

        Returns None if no plugin can handle the message (fallback to chat).
        """
        result = self._registry.find_handler(message, context)
        if not result:
            return None

        plugin, confidence = result
        if confidence < 0.5:
            return None

        provider = self._chat_pool.get_for_model(plugin.model)
        if not provider:
            logger.warning(
                "No provider for plugin model",
                plugin=plugin.name,
                model=plugin.model,
            )
            return None

        try:
            response = await plugin.handle(message, context, provider)
            logger.info(
                "Plugin handled message",
                plugin=plugin.name,
                model=plugin.model,
                confidence=confidence,
            )
            return response
        except Exception as exc:
            logger.error(
                "Plugin handler failed",
                plugin=plugin.name,
                error=str(exc),
            )
            return None
