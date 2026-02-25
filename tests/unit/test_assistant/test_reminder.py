"""Test reminder plugin — LLM-based reminder extraction."""

import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.assistant.base import PluginResponse
from src.assistant.plugins.reminder import ReminderPlugin


class TestReminderPlugin:
    """Test ReminderPlugin pattern matching and LLM extraction."""

    def setup_method(self) -> None:
        self.plugin = ReminderPlugin()

    async def test_can_handle_matches_english_reminder_patterns(self) -> None:
        """Test can_handle() matches English reminder keywords."""
        test_cases = [
            "remind me in 10 minutes",
            "set a timer for 30 seconds",
            "alarm in 1 hour",
            "set alarm for tomorrow",
        ]

        for message in test_cases:
            confidence = await self.plugin.can_handle(message, {})
            assert confidence == 0.9, f"Failed to match: {message}"

    async def test_can_handle_matches_russian_reminder_patterns(self) -> None:
        """Test can_handle() matches Russian reminder keywords."""
        test_cases = [
            "напомни через 30 минут",
            "поставь таймер на 5 минут",
            "будильник через час",
            "напоминание через 10 мин",
            "запланируй встречу",
        ]

        for message in test_cases:
            confidence = await self.plugin.can_handle(message, {})
            assert confidence == 0.9, f"Failed to match: {message}"

    async def test_can_handle_returns_zero_for_non_matching(self) -> None:
        """Test can_handle() returns 0.0 for non-matching messages."""
        test_cases = [
            "what is the weather",
            "tell me a joke",
            "how are you",
            "что такое python",
        ]

        for message in test_cases:
            confidence = await self.plugin.can_handle(message, {})
            assert confidence == 0.0, f"Incorrectly matched: {message}"

    async def test_patterns_match_time_expressions(self) -> None:
        """Test patterns match various time expressions."""
        # Test the actual regex patterns
        patterns = self.plugin.patterns
        test_cases = [
            "через 10 мин позвонить",
            "через 2 часа встреча",
            "через 30 сек проверить",
        ]

        for message in test_cases:
            matched = any(pattern.search(message) for pattern in patterns)
            assert matched, f"Pattern failed to match: {message}"

    async def test_handle_parses_valid_json_response(self) -> None:
        """Test handle() successfully parses valid JSON from provider."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "call mom", "delay_minutes": 30, "recurring": null}'
        )

        context = {"user_id": 12345}
        response = await self.plugin.handle("remind me to call mom in 30 minutes", context, mock_provider)

        assert isinstance(response, PluginResponse)
        assert "call mom" in response.content
        assert "30" in response.content
        assert response.model == "gpt-4o-mini"

    async def test_handle_formats_time_under_60_minutes(self) -> None:
        """Test handle() formats time correctly when under 60 minutes."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "check tests", "delay_minutes": 45, "recurring": null}'
        )

        response = await self.plugin.handle("remind in 45 min", {}, mock_provider)

        assert "45 мин" in response.content

    async def test_handle_formats_time_over_60_minutes(self) -> None:
        """Test handle() formats time correctly when over 60 minutes."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "meeting", "delay_minutes": 90, "recurring": null}'
        )

        response = await self.plugin.handle("remind in 90 minutes", {}, mock_provider)

        assert "1ч 30мин" in response.content

    async def test_handle_formats_time_exactly_60_minutes(self) -> None:
        """Test handle() formats time correctly for exactly 60 minutes."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "lunch", "delay_minutes": 60, "recurring": null}'
        )

        response = await self.plugin.handle("remind in 1 hour", {}, mock_provider)

        assert "1ч" in response.content
        # Should not have minutes part
        assert "0мин" not in response.content

    async def test_handle_includes_recurring_info(self) -> None:
        """Test handle() includes recurring information when present."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "standup", "delay_minutes": 1440, "recurring": "daily"}'
        )

        response = await self.plugin.handle("remind me daily", {}, mock_provider)

        assert "daily" in response.content or "повтор" in response.content

    async def test_handle_returns_error_when_provider_returns_error(self) -> None:
        """Test handle() returns error response when provider returns error JSON."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"error": "Could not parse time expression"}'
        )

        response = await self.plugin.handle("remind me sometime", {}, mock_provider)

        assert isinstance(response, PluginResponse)
        assert "Could not parse time expression" in response.content or "Не удалось распознать" in response.content

    async def test_handle_handles_json_decode_error(self) -> None:
        """Test handle() handles JSON parse errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='invalid json {'
        )

        response = await self.plugin.handle("remind me", {}, mock_provider)

        assert isinstance(response, PluginResponse)
        assert "Не удалось распознать" in response.content

    async def test_handle_handles_missing_fields(self) -> None:
        """Test handle() handles missing JSON fields with defaults."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "task"}'
        )

        response = await self.plugin.handle("remind me", {}, mock_provider)

        # Should use default delay (30 minutes)
        assert isinstance(response, PluginResponse)
        assert "task" in response.content

    async def test_handle_handles_value_error(self) -> None:
        """Test handle() handles ValueError when parsing delay_minutes."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "task", "delay_minutes": "invalid", "recurring": null}'
        )

        response = await self.plugin.handle("remind me", {}, mock_provider)

        assert isinstance(response, PluginResponse)
        assert "Не удалось распознать" in response.content

    async def test_handle_stores_reminder_in_memory(self) -> None:
        """Test handle() stores reminder in internal _pending list."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "test reminder", "delay_minutes": 15, "recurring": null}'
        )

        context = {"user_id": 12345}
        initial_count = len(self.plugin._pending)

        await self.plugin.handle("remind me in 15 min", context, mock_provider)

        assert len(self.plugin._pending) == initial_count + 1
        reminder = self.plugin._pending[-1]
        assert reminder.text == "test reminder"
        assert reminder.user_id == 12345

    async def test_handle_includes_metadata_with_timestamp(self) -> None:
        """Test handle() includes metadata with reminder timestamp."""
        mock_provider = MagicMock()
        mock_provider.classify = AsyncMock(
            return_value='{"text": "task", "delay_minutes": 20, "recurring": null}'
        )

        response = await self.plugin.handle("remind me", {}, mock_provider)

        assert "reminder_at" in response.metadata
        assert isinstance(response.metadata["reminder_at"], str)

    async def test_plugin_attributes(self) -> None:
        """Test plugin has required attributes."""
        assert self.plugin.name == "reminder"
        assert self.plugin.description == "Set and manage reminders"
        assert self.plugin.model == "gpt-4o-mini"
        assert len(self.plugin.patterns) > 0
        assert all(isinstance(p, re.Pattern) for p in self.plugin.patterns)

    async def test_plugin_response_is_importable(self) -> None:
        """Test that PluginResponse base class is importable from correct location."""
        # This test verifies the import structure
        from src.assistant.base import PluginResponse as ImportedResponse
        assert ImportedResponse is PluginResponse
