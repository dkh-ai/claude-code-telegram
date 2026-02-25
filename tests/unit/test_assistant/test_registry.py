"""Test plugin registry — registration and pattern matching."""

import re

import pytest

from src.assistant.base import PluginResponse
from src.assistant.registry import PluginRegistry


class MockPlugin:
    """Mock plugin for testing."""

    def __init__(self, name: str, patterns: list[re.Pattern[str]]) -> None:
        self.name = name
        self.description = f"Mock plugin {name}"
        self.patterns = patterns
        self.model = "mock-model"

    async def can_handle(self, message: str, context: dict) -> float:
        return 0.9

    async def handle(self, message: str, context: dict, provider) -> PluginResponse:
        return PluginResponse(content=f"Mock response from {self.name}", model=self.model)


class TestPluginRegistry:
    """Test PluginRegistry registration and handler discovery."""

    def setup_method(self) -> None:
        self.registry = PluginRegistry()

    def test_register_adds_plugin(self) -> None:
        """Test that register() adds a plugin to the registry."""
        plugin = MockPlugin("test", [re.compile(r"test", re.I)])
        self.registry.register(plugin)

        plugins = self.registry.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test"

    def test_find_handler_matches_pattern(self) -> None:
        """Test that find_handler() returns plugin when pattern matches."""
        plugin = MockPlugin("reminder", [re.compile(r"remind|напомни", re.I)])
        self.registry.register(plugin)

        result = self.registry.find_handler("remind me in 10 minutes", {})
        assert result is not None
        found_plugin, confidence = result
        assert found_plugin.name == "reminder"
        assert confidence == 0.8

    def test_find_handler_matches_russian_pattern(self) -> None:
        """Test that find_handler() matches Russian patterns."""
        plugin = MockPlugin("reminder", [re.compile(r"remind|напомни", re.I)])
        self.registry.register(plugin)

        result = self.registry.find_handler("напомни через час", {})
        assert result is not None
        found_plugin, confidence = result
        assert found_plugin.name == "reminder"
        assert confidence == 0.8

    def test_find_handler_no_match_returns_none(self) -> None:
        """Test that find_handler() returns None when no patterns match."""
        plugin = MockPlugin("reminder", [re.compile(r"remind", re.I)])
        self.registry.register(plugin)

        result = self.registry.find_handler("what is the weather", {})
        assert result is None

    def test_find_handler_returns_first_matching_plugin(self) -> None:
        """Test that find_handler() returns the first matching plugin."""
        plugin1 = MockPlugin("plugin1", [re.compile(r"test", re.I)])
        plugin2 = MockPlugin("plugin2", [re.compile(r"test", re.I)])
        self.registry.register(plugin1)
        self.registry.register(plugin2)

        result = self.registry.find_handler("test message", {})
        assert result is not None
        found_plugin, confidence = result
        # Should return the first registered plugin
        assert found_plugin.name == "plugin1"

    def test_list_plugins_returns_copy(self) -> None:
        """Test that list_plugins() returns a copy, not the internal list."""
        plugin = MockPlugin("test", [re.compile(r"test", re.I)])
        self.registry.register(plugin)

        plugins1 = self.registry.list_plugins()
        plugins2 = self.registry.list_plugins()

        # Should be equal but not the same object
        assert plugins1 == plugins2
        assert plugins1 is not plugins2

    def test_multiple_patterns_in_plugin(self) -> None:
        """Test plugin with multiple patterns."""
        plugin = MockPlugin(
            "reminder",
            [
                re.compile(r"remind", re.I),
                re.compile(r"timer", re.I),
                re.compile(r"alarm", re.I),
            ],
        )
        self.registry.register(plugin)

        # Test each pattern matches
        for message in ["remind me", "set a timer", "wake up alarm"]:
            result = self.registry.find_handler(message, {})
            assert result is not None
            found_plugin, _ = result
            assert found_plugin.name == "reminder"

    def test_empty_registry_returns_none(self) -> None:
        """Test find_handler() on empty registry returns None."""
        result = self.registry.find_handler("any message", {})
        assert result is None
