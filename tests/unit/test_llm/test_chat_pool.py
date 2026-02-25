"""Tests for chat provider pool."""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.llm.chat_pool import ChatProviderPool


@pytest.fixture
def mock_settings():
    """Create mock settings with API keys."""
    settings = MagicMock(spec=Settings)
    settings.deepseek_api_key_str = "sk-deepseek-test"
    settings.openai_api_key_str = "sk-openai-test"
    settings.anthropic_api_key_str = "sk-anthropic-test"
    settings.model_router_llm = "deepseek-chat"
    return settings


@pytest.fixture
def mock_settings_openai_only():
    """Create mock settings with only OpenAI key."""
    settings = MagicMock(spec=Settings)
    settings.deepseek_api_key_str = None
    settings.openai_api_key_str = "sk-openai-test"
    settings.anthropic_api_key_str = None
    settings.model_router_llm = "gpt-4o-mini"
    return settings


@pytest.fixture
def mock_settings_no_keys():
    """Create mock settings with no API keys."""
    settings = MagicMock(spec=Settings)
    settings.deepseek_api_key_str = None
    settings.openai_api_key_str = None
    settings.anthropic_api_key_str = None
    settings.model_router_llm = "gpt-4o-mini"
    return settings


class TestResolveVendor:
    """Tests for _resolve_vendor method."""

    def test_deepseek_model_resolves_to_deepseek_api(self, mock_settings):
        """Test that deepseek model resolves to DeepSeek API and key."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("deepseek-chat")

        assert base_url == "https://api.deepseek.com"
        assert api_key == "sk-deepseek-test"

    def test_deepseek_reasoner_resolves_to_deepseek_api(self, mock_settings):
        """Test that deepseek-reasoner model resolves to DeepSeek API."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("deepseek-reasoner")

        assert base_url == "https://api.deepseek.com"
        assert api_key == "sk-deepseek-test"

    def test_gpt_model_resolves_to_openai(self, mock_settings):
        """Test that gpt model resolves to OpenAI with no base_url."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("gpt-4o-mini")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_gpt_turbo_model_resolves_to_openai(self, mock_settings):
        """Test that gpt-3.5-turbo model resolves to OpenAI."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("gpt-3.5-turbo")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_o1_model_resolves_to_openai(self, mock_settings):
        """Test that o1 model resolves to OpenAI."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("o1-preview")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_o3_model_resolves_to_openai(self, mock_settings):
        """Test that o3 model resolves to OpenAI."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("o3-mini")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_o4_model_resolves_to_openai(self, mock_settings):
        """Test that o4 model resolves to OpenAI."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("o4-preview")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_unknown_model_falls_back_to_openai(self, mock_settings):
        """Test that unknown model falls back to OpenAI key with no base_url."""
        pool = ChatProviderPool(mock_settings)
        base_url, api_key = pool._resolve_vendor("claude-sonnet-4-5")

        assert base_url is None
        assert api_key == "sk-openai-test"

    def test_unknown_model_with_no_openai_key_returns_none(self, mock_settings_no_keys):
        """Test that unknown model returns None when no OpenAI key available."""
        pool = ChatProviderPool(mock_settings_no_keys)
        base_url, api_key = pool._resolve_vendor("unknown-model")

        assert base_url is None
        assert api_key is None

    def test_deepseek_with_no_key_returns_none(self, mock_settings_no_keys):
        """Test that deepseek model returns None when no deepseek key available."""
        pool = ChatProviderPool(mock_settings_no_keys)
        base_url, api_key = pool._resolve_vendor("deepseek-chat")

        assert base_url == "https://api.deepseek.com"
        assert api_key is None


class TestGetForModel:
    """Tests for get_for_model method."""

    @patch("src.llm.chat_pool.ChatProvider")
    def test_creates_provider_for_new_model(self, mock_chat_provider_cls, mock_settings):
        """Test that get_for_model creates a new provider for unseen model."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings)

        result = pool.get_for_model("deepseek-chat")

        assert result is mock_provider
        mock_chat_provider_cls.assert_called_once_with(
            model="deepseek-chat",
            api_key="sk-deepseek-test",
            base_url="https://api.deepseek.com",
        )

    @patch("src.llm.chat_pool.ChatProvider")
    def test_caches_provider_for_same_model(self, mock_chat_provider_cls, mock_settings):
        """Test that get_for_model returns cached provider on second call."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings)

        result1 = pool.get_for_model("deepseek-chat")
        result2 = pool.get_for_model("deepseek-chat")

        assert result1 is result2
        mock_chat_provider_cls.assert_called_once()

    @patch("src.llm.chat_pool.ChatProvider")
    def test_creates_separate_providers_for_different_models(
        self, mock_chat_provider_cls, mock_settings
    ):
        """Test that different models get separate provider instances."""
        mock_provider1 = MagicMock()
        mock_provider2 = MagicMock()
        mock_chat_provider_cls.side_effect = [mock_provider1, mock_provider2]
        pool = ChatProviderPool(mock_settings)

        result1 = pool.get_for_model("deepseek-chat")
        result2 = pool.get_for_model("gpt-4o-mini")

        assert result1 is mock_provider1
        assert result2 is mock_provider2
        assert mock_chat_provider_cls.call_count == 2

    @patch("src.llm.chat_pool.ChatProvider")
    def test_returns_none_when_no_api_key_available(
        self, mock_chat_provider_cls, mock_settings_no_keys
    ):
        """Test that get_for_model returns None when no API key found."""
        pool = ChatProviderPool(mock_settings_no_keys)

        result = pool.get_for_model("gpt-4o-mini")

        assert result is None
        mock_chat_provider_cls.assert_not_called()

    @patch("src.llm.chat_pool.ChatProvider")
    def test_creates_openai_provider_for_gpt_model(
        self, mock_chat_provider_cls, mock_settings
    ):
        """Test that GPT model creates provider with OpenAI key and no base_url."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings)

        result = pool.get_for_model("gpt-4o-mini")

        assert result is mock_provider
        mock_chat_provider_cls.assert_called_once_with(
            model="gpt-4o-mini",
            api_key="sk-openai-test",
            base_url=None,
        )

    @patch("src.llm.chat_pool.ChatProvider")
    def test_returns_none_for_deepseek_when_key_missing(
        self, mock_chat_provider_cls, mock_settings_openai_only
    ):
        """Test that deepseek model returns None when deepseek key not available."""
        pool = ChatProviderPool(mock_settings_openai_only)

        result = pool.get_for_model("deepseek-chat")

        assert result is None
        mock_chat_provider_cls.assert_not_called()

    @patch("src.llm.chat_pool.ChatProvider")
    def test_uses_openai_for_o1_family(
        self, mock_chat_provider_cls, mock_settings_openai_only
    ):
        """Test that o1 family models use OpenAI key."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings_openai_only)

        result = pool.get_for_model("o1-mini")

        assert result is mock_provider
        mock_chat_provider_cls.assert_called_once_with(
            model="o1-mini",
            api_key="sk-openai-test",
            base_url=None,
        )


class TestGetRouterProvider:
    """Tests for get_router_provider method."""

    @patch("src.llm.chat_pool.ChatProvider")
    def test_returns_provider_for_router_model(
        self, mock_chat_provider_cls, mock_settings
    ):
        """Test that get_router_provider delegates to get_for_model with router model."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings)

        result = pool.get_router_provider()

        assert result is mock_provider
        mock_chat_provider_cls.assert_called_once_with(
            model="deepseek-chat",
            api_key="sk-deepseek-test",
            base_url="https://api.deepseek.com",
        )

    @patch("src.llm.chat_pool.ChatProvider")
    def test_returns_none_when_router_model_has_no_key(
        self, mock_chat_provider_cls, mock_settings_no_keys
    ):
        """Test that get_router_provider returns None when router model key missing."""
        pool = ChatProviderPool(mock_settings_no_keys)

        result = pool.get_router_provider()

        assert result is None
        mock_chat_provider_cls.assert_not_called()

    @patch("src.llm.chat_pool.ChatProvider")
    def test_caches_router_provider(self, mock_chat_provider_cls, mock_settings):
        """Test that get_router_provider uses cached provider on subsequent calls."""
        mock_provider = MagicMock()
        mock_chat_provider_cls.return_value = mock_provider
        pool = ChatProviderPool(mock_settings)

        result1 = pool.get_router_provider()
        result2 = pool.get_router_provider()

        assert result1 is result2
        mock_chat_provider_cls.assert_called_once()


class TestAvailableVendors:
    """Tests for available_vendors method."""

    def test_returns_all_vendors_when_all_keys_present(self, mock_settings):
        """Test that available_vendors lists all vendors when all keys configured."""
        pool = ChatProviderPool(mock_settings)

        vendors = pool.available_vendors()

        assert vendors == ["deepseek", "openai", "anthropic"]

    def test_returns_only_openai_when_only_openai_key(self, mock_settings_openai_only):
        """Test that available_vendors lists only openai when only OpenAI key present."""
        pool = ChatProviderPool(mock_settings_openai_only)

        vendors = pool.available_vendors()

        assert vendors == ["openai"]

    def test_returns_empty_list_when_no_keys(self, mock_settings_no_keys):
        """Test that available_vendors returns empty list when no keys configured."""
        pool = ChatProviderPool(mock_settings_no_keys)

        vendors = pool.available_vendors()

        assert vendors == []

    def test_returns_deepseek_and_openai_when_anthropic_missing(self):
        """Test that available_vendors lists deepseek and openai when anthropic key missing."""
        settings = MagicMock(spec=Settings)
        settings.deepseek_api_key_str = "sk-deepseek-test"
        settings.openai_api_key_str = "sk-openai-test"
        settings.anthropic_api_key_str = None
        settings.model_router_llm = "gpt-4o-mini"
        pool = ChatProviderPool(settings)

        vendors = pool.available_vendors()

        assert vendors == ["deepseek", "openai"]

    def test_returns_only_anthropic_when_only_anthropic_key(self):
        """Test that available_vendors lists only anthropic when only Anthropic key present."""
        settings = MagicMock(spec=Settings)
        settings.deepseek_api_key_str = None
        settings.openai_api_key_str = None
        settings.anthropic_api_key_str = "sk-anthropic-test"
        settings.model_router_llm = "claude-sonnet-4-5"
        pool = ChatProviderPool(settings)

        vendors = pool.available_vendors()

        assert vendors == ["anthropic"]
