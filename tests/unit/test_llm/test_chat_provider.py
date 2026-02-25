"""Unit tests for chat_provider module."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from src.llm.chat_provider import (
    _estimate_cost,
    ChatResponse,
    ChatProvider,
    MODEL_PRICING,
)


# Test fixtures and helpers

def create_mock_response(
    content: str = "test response",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.model = model

    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens

    return mock_response


@pytest.fixture
def mock_openai_client():
    """Create a mock AsyncOpenAI client."""
    with patch("src.llm.chat_provider.AsyncOpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


# Test _estimate_cost function

class TestEstimateCost:
    def test_known_model_deepseek_chat(self):
        """Test cost estimation for known model deepseek-chat."""
        cost = _estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.14 + 500 * 0.28) / 1_000_000
        assert cost == expected
        assert cost == pytest.approx(0.00028, abs=1e-8)

    def test_known_model_gpt_4o(self):
        """Test cost estimation for known model gpt-4o."""
        cost = _estimate_cost("gpt-4o", input_tokens=2000, output_tokens=1000)
        expected = (2000 * 2.50 + 1000 * 10.00) / 1_000_000
        assert cost == expected
        assert cost == pytest.approx(0.015, abs=1e-8)

    def test_unknown_model_fallback(self):
        """Test cost estimation falls back to default pricing for unknown models."""
        cost = _estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        expected = (1000 * 1.0 + 500 * 3.0) / 1_000_000
        assert cost == expected
        assert cost == pytest.approx(0.0025, abs=1e-8)

    def test_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = _estimate_cost("gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_zero_input_tokens(self):
        """Test cost estimation with only output tokens."""
        cost = _estimate_cost("gpt-4o-mini", input_tokens=0, output_tokens=1000)
        expected = (0 * 0.15 + 1000 * 0.60) / 1_000_000
        assert cost == expected

    def test_zero_output_tokens(self):
        """Test cost estimation with only input tokens."""
        cost = _estimate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=0)
        expected = (1000 * 0.15 + 0 * 0.60) / 1_000_000
        assert cost == expected


# Test ChatResponse dataclass

class TestChatResponse:
    def test_dataclass_construction(self):
        """Test ChatResponse can be constructed with all fields."""
        response = ChatResponse(
            content="Hello world",
            model="gpt-4o-mini",
            cost=0.001,
            input_tokens=100,
            output_tokens=50,
            duration_ms=250,
        )
        assert response.content == "Hello world"
        assert response.model == "gpt-4o-mini"
        assert response.cost == 0.001
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.duration_ms == 250

    def test_dataclass_is_dataclass(self):
        """Test ChatResponse is a proper dataclass."""
        response = ChatResponse(
            content="test",
            model="test-model",
            cost=0.0,
            input_tokens=0,
            output_tokens=0,
            duration_ms=0,
        )
        assert asdict(response) == {
            "content": "test",
            "model": "test-model",
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_ms": 0,
        }


# Test ChatProvider class

class TestChatProviderInit:
    def test_init_stores_model(self, mock_openai_client):
        """Test ChatProvider stores the model name."""
        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        assert provider.model == "gpt-4o-mini"

    def test_init_creates_client_with_api_key(self):
        """Test ChatProvider creates AsyncOpenAI client with API key."""
        with patch("src.llm.chat_provider.AsyncOpenAI") as mock_client_class:
            ChatProvider(model="gpt-4o-mini", api_key="test-api-key")
            mock_client_class.assert_called_once_with(
                api_key="test-api-key",
                base_url=None,
            )

    def test_init_creates_client_with_base_url(self):
        """Test ChatProvider creates AsyncOpenAI client with custom base URL."""
        with patch("src.llm.chat_provider.AsyncOpenAI") as mock_client_class:
            ChatProvider(
                model="deepseek-chat",
                api_key="test-key",
                base_url="https://api.deepseek.com",
            )
            mock_client_class.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.deepseek.com",
            )


class TestChatProviderChat:
    async def test_chat_successful_call(self, mock_openai_client):
        """Test successful chat call returns ChatResponse."""
        mock_response = create_mock_response(
            content="Hello there",
            model="gpt-4o-mini",
            prompt_tokens=150,
            completion_tokens=75,
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "Hi"}]

        response = await provider.chat(messages)

        assert response.content == "Hello there"
        assert response.model == "gpt-4o-mini"
        assert response.input_tokens == 150
        assert response.output_tokens == 75
        assert response.cost > 0
        assert response.duration_ms >= 0

    async def test_chat_uses_default_model(self, mock_openai_client):
        """Test chat uses provider's default model when not overridden."""
        mock_response = create_mock_response()
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="deepseek-chat", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        await provider.chat(messages)

        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"

    async def test_chat_model_override(self, mock_openai_client):
        """Test chat can override the default model."""
        mock_response = create_mock_response(model="gpt-4o")
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        response = await provider.chat(messages, model="gpt-4o")

        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert response.model == "gpt-4o"

    async def test_chat_with_custom_parameters(self, mock_openai_client):
        """Test chat passes max_tokens and temperature parameters."""
        mock_response = create_mock_response()
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        await provider.chat(messages, max_tokens=2048, temperature=0.5)

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5

    async def test_chat_empty_usage(self, mock_openai_client):
        """Test chat handles missing usage information gracefully."""
        mock_response = create_mock_response()
        mock_response.usage = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        response = await provider.chat(messages)

        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.cost == 0.0

    async def test_chat_empty_content(self, mock_openai_client):
        """Test chat handles None content gracefully."""
        mock_response = create_mock_response()
        mock_response.choices[0].message.content = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        response = await provider.chat(messages)

        assert response.content == ""

    async def test_chat_model_fallback_in_response(self, mock_openai_client):
        """Test chat uses requested model when response.model is None."""
        mock_response = create_mock_response()
        mock_response.model = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        response = await provider.chat(messages)

        assert response.model == "gpt-4o-mini"

    async def test_chat_passes_messages_correctly(self, mock_openai_client):
        """Test chat passes messages list to API correctly."""
        mock_response = create_mock_response()
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        await provider.chat(messages)

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == messages

    async def test_chat_duration_calculated(self, mock_openai_client):
        """Test chat calculates duration_ms correctly."""
        mock_response = create_mock_response()
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")
        messages = [{"role": "user", "content": "test"}]

        response = await provider.chat(messages)

        # Duration should be >= 0 and typically small for mocked call
        assert isinstance(response.duration_ms, int)
        assert response.duration_ms >= 0


class TestChatProviderClassify:
    async def test_classify_delegates_to_chat(self, mock_openai_client):
        """Test classify delegates to chat with correct parameters."""
        mock_response = create_mock_response(content="positive")
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")

        result = await provider.classify(
            prompt="This is great!",
            system="Classify sentiment as positive or negative",
        )

        assert result == "positive"

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "Classify sentiment as positive or negative"},
            {"role": "user", "content": "This is great!"},
        ]
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.0

    async def test_classify_uses_default_model(self, mock_openai_client):
        """Test classify uses provider's default model."""
        mock_response = create_mock_response(content="yes")
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="deepseek-chat", api_key="test-key")

        await provider.classify(prompt="Is this spam?", system="Answer yes or no")

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"

    async def test_classify_with_model_override(self, mock_openai_client):
        """Test classify can override model."""
        mock_response = create_mock_response(content="no")
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")

        await provider.classify(
            prompt="Is this urgent?",
            system="Answer yes or no",
            model="gpt-4o",
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    async def test_classify_returns_content_string(self, mock_openai_client):
        """Test classify returns the content string from response."""
        mock_response = create_mock_response(content="classification result")
        mock_openai_client.chat.completions.create.return_value = mock_response

        provider = ChatProvider(model="gpt-4o-mini", api_key="test-key")

        result = await provider.classify(prompt="test", system="classify this")

        assert result == "classification result"
        assert isinstance(result, str)
