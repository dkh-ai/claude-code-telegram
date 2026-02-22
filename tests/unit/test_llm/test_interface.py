"""Tests for LLM provider interface and response dataclass."""

from src.llm.interface import LLMResponse


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_success_response(self):
        """Test creating a successful LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            session_id="sess-123",
            cost=0.05,
            duration_ms=1500,
            num_turns=3,
            is_error=False,
        )

        assert response.content == "Hello, world!"
        assert response.session_id == "sess-123"
        assert response.cost == 0.05
        assert response.duration_ms == 1500
        assert response.num_turns == 3
        assert response.is_error is False
        assert response.error_message is None

    def test_error_response(self):
        """Test creating an error LLMResponse."""
        response = LLMResponse(
            content="",
            session_id=None,
            cost=0.0,
            duration_ms=100,
            num_turns=0,
            is_error=True,
            error_message="Connection timeout",
        )

        assert response.content == ""
        assert response.session_id is None
        assert response.cost == 0.0
        assert response.duration_ms == 100
        assert response.num_turns == 0
        assert response.is_error is True
        assert response.error_message == "Connection timeout"

    def test_default_error_message_is_none(self):
        """Test that error_message defaults to None."""
        response = LLMResponse(
            content="result",
            session_id="s1",
            cost=0.01,
            duration_ms=500,
            num_turns=1,
            is_error=False,
        )

        assert response.error_message is None

    def test_response_with_no_session_id(self):
        """Test LLMResponse with session_id=None (new session)."""
        response = LLMResponse(
            content="result",
            session_id=None,
            cost=0.02,
            duration_ms=800,
            num_turns=2,
            is_error=False,
        )

        assert response.session_id is None
