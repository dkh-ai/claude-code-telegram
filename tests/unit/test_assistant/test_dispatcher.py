"""Test assistant dispatcher — routing messages to plugins."""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.assistant.base import PluginResponse
from src.assistant.dispatcher import AssistantDispatcher
from src.assistant.registry import PluginRegistry


class MockPlugin:
    """Mock plugin for testing dispatcher."""

    def __init__(self, name: str, patterns: list[re.Pattern[str]], model: str = "mock-model") -> None:
        self.name = name
        self.description = f"Mock plugin {name}"
        self.patterns = patterns
        self.model = model
        self._handle_response: PluginResponse | None = None
        self._should_raise: bool = False

    def set_handle_response(self, response: PluginResponse) -> None:
        """Set the response to return from handle()."""
        self._handle_response = response

    def set_should_raise(self, should_raise: bool = True) -> None:
        """Configure whether handle() should raise an exception."""
        self._should_raise = should_raise

    async def can_handle(self, message: str, context: dict) -> float:
        return 0.9

    async def handle(self, message: str, context: dict, provider) -> PluginResponse:
        if self._should_raise:
            raise RuntimeError("Plugin handler failed")
        if self._handle_response:
            return self._handle_response
        return PluginResponse(content=f"Response from {self.name}", model=self.model)


class TestAssistantDispatcher:
    """Test AssistantDispatcher routing logic."""

    def setup_method(self) -> None:
        self.registry = PluginRegistry()
        self.chat_pool = MagicMock()
        self.dispatcher = AssistantDispatcher(self.registry, self.chat_pool)

    async def test_dispatch_returns_none_when_no_handler_found(self) -> None:
        """Test dispatch() returns None when registry finds no matching plugin."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)])
        self.registry.register(plugin)

        # Message that doesn't match the pattern
        result = await self.dispatcher.dispatch("what is the weather", {})
        assert result is None

    async def test_dispatch_returns_none_when_confidence_too_low(self) -> None:
        """Test dispatch() returns None when confidence < 0.5."""
        # This test verifies the logic, though current registry always returns 0.8
        # We'll mock the registry to return low confidence
        plugin = MockPlugin("test", [re.compile(r"test", re.I)])
        self.registry.register(plugin)

        # Patch find_handler to return low confidence
        original_find = self.registry.find_handler
        self.registry.find_handler = lambda msg, ctx: (plugin, 0.3)

        result = await self.dispatcher.dispatch("test message", {})
        assert result is None

        # Restore original method
        self.registry.find_handler = original_find

    async def test_dispatch_returns_none_when_no_provider_for_model(self) -> None:
        """Test dispatch() returns None when chat_pool has no provider for plugin model."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)], model="gpt-4o-mini")
        self.registry.register(plugin)

        # Configure chat_pool to return None for the model
        self.chat_pool.get_for_model = MagicMock(return_value=None)

        result = await self.dispatcher.dispatch("remind me in 10 minutes", {})
        assert result is None

    async def test_dispatch_returns_plugin_response_on_success(self) -> None:
        """Test dispatch() returns PluginResponse when plugin handles successfully."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)], model="gpt-4o-mini")
        expected_response = PluginResponse(
            content="Reminder set for 10 minutes",
            model="gpt-4o-mini",
            metadata={"delay_minutes": 10},
        )
        plugin.set_handle_response(expected_response)
        self.registry.register(plugin)

        # Configure chat_pool to return a mock provider
        mock_provider = MagicMock()
        self.chat_pool.get_for_model = MagicMock(return_value=mock_provider)

        result = await self.dispatcher.dispatch("remind me in 10 minutes", {})
        assert result is not None
        assert result.content == "Reminder set for 10 minutes"
        assert result.model == "gpt-4o-mini"
        assert result.metadata == {"delay_minutes": 10}

    async def test_dispatch_returns_none_when_plugin_raises_exception(self) -> None:
        """Test dispatch() returns None when plugin.handle() raises an exception."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)])
        plugin.set_should_raise(True)
        self.registry.register(plugin)

        # Configure chat_pool to return a mock provider
        mock_provider = MagicMock()
        self.chat_pool.get_for_model = MagicMock(return_value=mock_provider)

        result = await self.dispatcher.dispatch("remind me in 10 minutes", {})
        assert result is None

    async def test_dispatch_passes_context_to_plugin(self) -> None:
        """Test dispatch() passes context dict to plugin.handle()."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)])
        self.registry.register(plugin)

        mock_provider = MagicMock()
        self.chat_pool.get_for_model = MagicMock(return_value=mock_provider)

        context = {"user_id": 12345, "chat_id": 67890}
        # Patch plugin.handle to capture the context
        original_handle = plugin.handle
        captured_context = None

        async def capture_handle(message: str, ctx: dict, provider) -> PluginResponse:
            nonlocal captured_context
            captured_context = ctx
            return await original_handle(message, ctx, provider)

        plugin.handle = capture_handle

        await self.dispatcher.dispatch("remind me", context)
        assert captured_context == context

    async def test_dispatch_passes_provider_to_plugin(self) -> None:
        """Test dispatch() passes chat provider to plugin.handle()."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)])
        self.registry.register(plugin)

        mock_provider = MagicMock()
        self.chat_pool.get_for_model = MagicMock(return_value=mock_provider)

        # Patch plugin.handle to capture the provider
        original_handle = plugin.handle
        captured_provider = None

        async def capture_handle(message: str, ctx: dict, provider) -> PluginResponse:
            nonlocal captured_provider
            captured_provider = provider
            return await original_handle(message, ctx, provider)

        plugin.handle = capture_handle

        await self.dispatcher.dispatch("remind me", {})
        assert captured_provider is mock_provider
