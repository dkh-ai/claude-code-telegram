"""Plugin discovery and management."""

from typing import Any, Optional

import structlog

from .base import AssistantPlugin

logger = structlog.get_logger()


class PluginRegistry:
    """Registry of assistant plugins."""

    def __init__(self) -> None:
        self._plugins: list[AssistantPlugin] = []

    def register(self, plugin: AssistantPlugin) -> None:
        """Register a plugin."""
        self._plugins.append(plugin)
        logger.info("Plugin registered", name=plugin.name)

    def find_handler(
        self,
        message: str,
        context: dict[str, Any],
    ) -> Optional[tuple[AssistantPlugin, float]]:
        """Find the best plugin handler for a message.

        Returns (plugin, confidence) or None if no plugin matches.
        """
        best: Optional[tuple[AssistantPlugin, float]] = None

        for plugin in self._plugins:
            # Quick regex check first
            for pattern in plugin.patterns:
                if pattern.search(message):
                    # Pattern matched → confidence from pattern
                    confidence = 0.8
                    if best is None or confidence > best[1]:
                        best = (plugin, confidence)
                    break

        return best

    def list_plugins(self) -> list[AssistantPlugin]:
        """List all registered plugins."""
        return list(self._plugins)
