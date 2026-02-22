"""Tests for LLM provider factory."""

from unittest.mock import MagicMock

import pytest

from src.llm.claude_provider import ClaudeProvider
from src.llm.factory import create_llm_provider


@pytest.fixture
def mock_settings():
    """Create mock settings with llm_provider field."""
    settings = MagicMock()
    settings.llm_provider = "claude_sdk"
    return settings


@pytest.fixture
def mock_claude_integration():
    """Create mock ClaudeIntegration for factory."""
    return MagicMock()


class TestCreateLLMProvider:
    """Tests for create_llm_provider factory function."""

    def test_claude_sdk_returns_claude_provider(
        self, mock_settings, mock_claude_integration
    ):
        """Test that 'claude_sdk' creates a ClaudeProvider."""
        provider = create_llm_provider(
            mock_settings, claude_integration=mock_claude_integration
        )

        assert isinstance(provider, ClaudeProvider)

    def test_claude_sdk_passes_integration(
        self, mock_settings, mock_claude_integration
    ):
        """Test that ClaudeIntegration is passed to ClaudeProvider."""
        provider = create_llm_provider(
            mock_settings, claude_integration=mock_claude_integration
        )

        assert provider._claude_integration is mock_claude_integration

    def test_unknown_provider_raises_value_error(self, mock_settings):
        """Test that unknown provider name raises ValueError."""
        mock_settings.llm_provider = "openai_gateway"

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider(mock_settings)

    def test_default_provider_is_claude_sdk(self, mock_claude_integration):
        """Test that missing llm_provider attribute defaults to claude_sdk."""
        settings = MagicMock(spec=[])  # No attributes by default
        settings.llm_provider = "claude_sdk"

        provider = create_llm_provider(
            settings, claude_integration=mock_claude_integration
        )

        assert isinstance(provider, ClaudeProvider)

    def test_claude_sdk_without_integration_raises(self, mock_settings):
        """Test that claude_sdk without claude_integration kwarg raises."""
        with pytest.raises(KeyError):
            create_llm_provider(mock_settings)
