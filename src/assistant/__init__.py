"""Assistant plugin system for structured task handling."""

from .base import AssistantPlugin, PluginResponse
from .dispatcher import AssistantDispatcher
from .registry import PluginRegistry

__all__ = [
    "AssistantPlugin",
    "AssistantDispatcher",
    "PluginRegistry",
    "PluginResponse",
]
