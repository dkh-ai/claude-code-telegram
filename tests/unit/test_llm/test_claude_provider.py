"""Tests for ClaudeProvider LLM implementation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.claude_provider import ClaudeProvider
from src.llm.interface import LLMResponse


@pytest.fixture
def mock_claude_response():
    """Create a mock ClaudeResponse."""
    response = MagicMock()
    response.content = "Hello from Claude"
    response.session_id = "sess-abc123"
    response.cost = 0.03
    response.duration_ms = 2000
    response.num_turns = 4
    response.is_error = False
    return response


@pytest.fixture
def mock_claude_integration(mock_claude_response):
    """Create a mock ClaudeIntegration."""
    integration = MagicMock()
    integration.run_command = AsyncMock(return_value=mock_claude_response)
    return integration


@pytest.fixture
def provider(mock_claude_integration):
    """Create a ClaudeProvider with mocked integration."""
    return ClaudeProvider(claude_integration=mock_claude_integration)


class TestClaudeProviderExecute:
    """Tests for ClaudeProvider.execute()."""

    async def test_execute_returns_llm_response(self, provider, mock_claude_integration):
        """Test that execute() returns a properly mapped LLMResponse."""
        result = await provider.execute(
            prompt="Hello",
            working_dir=Path("/tmp/project"),
            user_id=42,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Claude"
        assert result.session_id == "sess-abc123"
        assert result.cost == 0.03
        assert result.duration_ms == 2000
        assert result.num_turns == 4
        assert result.is_error is False
        assert result.error_message is None

    async def test_execute_passes_session_id(
        self, provider, mock_claude_integration
    ):
        """Test that session_id is forwarded to ClaudeIntegration."""
        await provider.execute(
            prompt="Continue",
            working_dir=Path("/tmp/project"),
            user_id=42,
            session_id="existing-session",
        )

        mock_claude_integration.run_command.assert_called_once()
        call_kwargs = mock_claude_integration.run_command.call_args[1]
        assert call_kwargs["session_id"] == "existing-session"

    async def test_execute_passes_stream_callback(
        self, provider, mock_claude_integration
    ):
        """Test that stream_callback is forwarded as on_stream."""
        callback = AsyncMock()

        await provider.execute(
            prompt="Hello",
            working_dir=Path("/tmp/project"),
            user_id=42,
            stream_callback=callback,
        )

        call_kwargs = mock_claude_integration.run_command.call_args[1]
        assert call_kwargs["on_stream"] is callback

    async def test_execute_passes_force_new(
        self, provider, mock_claude_integration
    ):
        """Test that force_new is forwarded to ClaudeIntegration."""
        await provider.execute(
            prompt="New session",
            working_dir=Path("/tmp/project"),
            user_id=42,
            force_new=True,
        )

        call_kwargs = mock_claude_integration.run_command.call_args[1]
        assert call_kwargs["force_new"] is True

    async def test_execute_maps_error_response(
        self, provider, mock_claude_integration
    ):
        """Test that an error ClaudeResponse is mapped correctly."""
        error_response = MagicMock()
        error_response.content = "Something went wrong"
        error_response.session_id = "sess-err"
        error_response.cost = 0.01
        error_response.duration_ms = 500
        error_response.num_turns = 1
        error_response.is_error = True
        mock_claude_integration.run_command = AsyncMock(return_value=error_response)

        result = await provider.execute(
            prompt="Fail",
            working_dir=Path("/tmp/project"),
            user_id=42,
        )

        assert result.is_error is True
        assert result.content == "Something went wrong"

    async def test_execute_handles_exception(
        self, provider, mock_claude_integration
    ):
        """Test that exceptions are caught and returned as error LLMResponse."""
        mock_claude_integration.run_command = AsyncMock(
            side_effect=RuntimeError("SDK connection lost")
        )

        result = await provider.execute(
            prompt="Hello",
            working_dir=Path("/tmp/project"),
            user_id=42,
        )

        assert result.is_error is True
        assert "SDK connection lost" in result.error_message
        assert result.content == ""
        assert result.cost == 0.0
        assert result.duration_ms == 0
        assert result.num_turns == 0
        assert result.session_id is None

    async def test_execute_default_args(
        self, provider, mock_claude_integration
    ):
        """Test that optional args default correctly when not provided."""
        await provider.execute(
            prompt="Hello",
            working_dir=Path("/tmp/project"),
            user_id=42,
        )

        call_kwargs = mock_claude_integration.run_command.call_args[1]
        assert call_kwargs["session_id"] is None
        assert call_kwargs["on_stream"] is None
        assert call_kwargs["force_new"] is False


class TestClaudeProviderHealthcheck:
    """Tests for ClaudeProvider.healthcheck()."""

    async def test_healthcheck_returns_true(self, provider):
        """Test that healthcheck always returns True."""
        result = await provider.healthcheck()
        assert result is True
