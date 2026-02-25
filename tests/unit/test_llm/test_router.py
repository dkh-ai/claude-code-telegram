"""Unit tests for src/llm/router.py."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.router import (
    AGENT_CHAIN,
    AGENT_PATTERNS,
    ASSISTANT_PATTERNS,
    CHAT_CHAIN,
    IntentRouter,
    RoutingDecision,
)


# Test fixtures


@pytest.fixture
def mock_settings():
    """Mock settings with routing configuration."""
    settings = MagicMock()
    settings.model_agent_default = "claude-sonnet-4-5"
    settings.model_agent_heavy = "claude-opus-4-6"
    settings.model_chat_default = "deepseek-chat"
    settings.model_chat_fallback = "gpt-4o-mini"
    settings.auto_route_enabled = True
    return settings


@pytest.fixture
def mock_chat_pool():
    """Mock chat provider pool."""
    pool = MagicMock()
    provider = MagicMock()
    provider.classify = AsyncMock()
    pool.get_router_provider.return_value = provider
    return pool


@pytest.fixture
def router(mock_settings, mock_chat_pool):
    """Create IntentRouter instance."""
    return IntentRouter(mock_settings, mock_chat_pool)


@pytest.fixture
def router_no_pool(mock_settings):
    """Create IntentRouter without chat pool."""
    return IntentRouter(mock_settings, chat_pool=None)


# TestRoutingDecision


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_construction(self):
        """Test basic dataclass construction."""
        decision = RoutingDecision(
            mode="agent",
            model="claude-sonnet-4-5",
            confidence=0.85,
            method="regex",
        )
        assert decision.mode == "agent"
        assert decision.model == "claude-sonnet-4-5"
        assert decision.confidence == 0.85
        assert decision.method == "regex"

    def test_all_modes(self):
        """Test all valid mode types."""
        modes = ["agent", "assistant", "chat"]
        for mode in modes:
            decision = RoutingDecision(
                mode=mode,
                model="test-model",
                confidence=1.0,
                method="test",
            )
            assert decision.mode == mode

    def test_all_methods(self):
        """Test all valid method types."""
        methods = ["regex", "llm", "override", "context"]
        for method in methods:
            decision = RoutingDecision(
                mode="chat",
                model="test-model",
                confidence=0.5,
                method=method,
            )
            assert decision.method == method


# TestMatchRegex


class TestMatchRegex:
    """Test regex pattern matching."""

    def test_agent_pattern_english_edit(self, router):
        """Test agent pattern matches English 'edit' keyword."""
        result = router._match_regex("edit the config file")
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"
        assert result.confidence == 0.85
        assert result.method == "regex"

    def test_agent_pattern_english_create(self, router):
        """Test agent pattern matches English 'create' keyword."""
        result = router._match_regex("create a new script.py")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_english_git(self, router):
        """Test agent pattern matches git commands."""
        result = router._match_regex("git commit -m 'test'")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_russian_sozdai(self, router):
        """Test agent pattern matches Russian 'создай' keyword."""
        result = router._match_regex("создай новый файл")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_russian_otredaktiruy(self, router):
        """Test agent pattern matches Russian 'отредактируй' keyword."""
        result = router._match_regex("отредактируй конфиг")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_file_extension_py(self, router):
        """Test agent pattern matches .py file extension."""
        result = router._match_regex("check test.py for errors")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_file_extension_js(self, router):
        """Test agent pattern matches .js file extension."""
        result = router._match_regex("review index.js")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_file_extension_yaml(self, router):
        """Test agent pattern matches .yaml file extension."""
        result = router._match_regex("update config.yaml")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_code_fence(self, router):
        """Test agent pattern matches code fence marker."""
        result = router._match_regex("here is the code:\n```python\nprint('hi')\n```")
        assert result is not None
        assert result.mode == "agent"

    def test_agent_pattern_case_insensitive(self, router):
        """Test agent patterns are case insensitive."""
        result = router._match_regex("EDIT THE FILE")
        assert result is not None
        assert result.mode == "agent"

    def test_assistant_pattern_reminder(self, router):
        """Test assistant pattern matches 'reminder' keyword."""
        result = router._match_regex("set a reminder for tomorrow")
        assert result is not None
        assert result.mode == "assistant"
        assert result.model == "deepseek-chat"
        assert result.confidence == 0.85
        assert result.method == "regex"

    def test_assistant_pattern_napomni(self, router):
        """Test assistant pattern matches Russian 'напомни' keyword."""
        result = router._match_regex("напомни мне завтра")
        assert result is not None
        assert result.mode == "assistant"

    def test_assistant_pattern_timer(self, router):
        """Test assistant pattern matches 'timer' keyword."""
        result = router._match_regex("start a 5 minute timer")
        assert result is not None
        assert result.mode == "assistant"

    def test_assistant_pattern_kb(self, router):
        """Test assistant pattern matches knowledge base keywords."""
        result = router._match_regex("search the knowledge base")
        assert result is not None
        assert result.mode == "assistant"

    def test_assistant_pattern_weather(self, router):
        """Test assistant pattern matches weather keyword."""
        result = router._match_regex("what's the weather today")
        assert result is not None
        assert result.mode == "assistant"

    def test_no_match_general_question(self, router):
        """Test general question returns None."""
        result = router._match_regex("what is machine learning?")
        assert result is None

    def test_no_match_greeting(self, router):
        """Test greeting returns None."""
        result = router._match_regex("hello there!")
        assert result is None

    def test_no_match_empty_string(self, router):
        """Test empty string returns None."""
        result = router._match_regex("")
        assert result is None

    def test_agent_priority_over_assistant(self, router):
        """Test agent patterns checked before assistant patterns."""
        # "edit" is both agent and could be assistant context
        result = router._match_regex("edit reminder settings")
        assert result is not None
        assert result.mode == "agent"  # Agent pattern matches first


# TestRoute


class TestRoute:
    """Test main route() method."""

    async def test_override_model(self, router):
        """Test user model override bypasses all routing logic."""
        context = {"model_override": "claude-opus-4-6"}
        result = await router.route("anything here", context)
        assert result.mode == "agent"
        assert result.model == "claude-opus-4-6"
        assert result.confidence == 1.0
        assert result.method == "override"

    async def test_override_non_claude_model(self, router):
        """Test override with non-Claude model sets chat mode."""
        context = {"model_override": "gpt-4o"}
        result = await router.route("test", context)
        assert result.mode == "chat"
        assert result.model == "gpt-4o"
        assert result.confidence == 1.0
        assert result.method == "override"

    async def test_active_session_stays_agent(self, router):
        """Test active Claude session stays in agent mode."""
        context = {"claude_session_id": "sess-123"}
        result = await router.route("continue working", context)
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"
        assert result.confidence == 0.8
        assert result.method == "context"

    async def test_active_session_force_new_bypasses(self, router):
        """Test force_new_session bypasses active session routing."""
        context = {"claude_session_id": "sess-123", "force_new_session": True}
        result = await router.route("edit file.py", context)
        # Should route via regex instead of context
        assert result.method in ["regex", "llm"]  # Not "context"

    async def test_auto_route_disabled_defaults_agent(self, router, mock_settings):
        """Test auto_route disabled defaults to agent mode."""
        mock_settings.auto_route_enabled = False
        result = await router.route("hello world", {})
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"
        assert result.confidence == 1.0
        assert result.method == "override"

    async def test_regex_match_returns_immediately(self, router):
        """Test regex match returns without LLM call."""
        result = await router.route("edit config.yaml", {})
        assert result.mode == "agent"
        assert result.method == "regex"
        # Verify LLM was not called
        router._chat_pool.get_router_provider().classify.assert_not_called()

    async def test_llm_fallback_on_no_regex_match(self, router, mock_chat_pool):
        """Test LLM classification used when regex doesn't match."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "chat",
            "confidence": 0.75,
        })
        result = await router.route("what is Python?", {})
        assert result.mode == "chat"
        assert result.method == "llm"
        assert result.confidence == 0.75
        mock_chat_pool.get_router_provider().classify.assert_called_once()

    async def test_default_chat_on_llm_failure(self, router, mock_chat_pool):
        """Test default to chat mode when both regex and LLM fail."""
        mock_chat_pool.get_router_provider().classify.side_effect = Exception("LLM error")
        result = await router.route("ambiguous message", {})
        assert result.mode == "chat"
        assert result.model == "deepseek-chat"
        assert result.confidence == 0.5
        assert result.method == "regex"  # Default still reports "regex" method

    async def test_default_chat_on_no_provider(self, router_no_pool):
        """Test default to chat when no chat pool available."""
        result = await router_no_pool.route("some question", {})
        assert result.mode == "chat"
        assert result.model == "deepseek-chat"
        assert result.confidence == 0.5
        assert result.method == "regex"

    async def test_routing_with_empty_context(self, router):
        """Test routing works with None context."""
        result = await router.route("edit test.py", None)
        assert result.mode == "agent"
        assert result.method == "regex"


# TestClassifyWithLLM


class TestClassifyWithLLM:
    """Test LLM classification."""

    async def test_successful_classification_agent(self, router, mock_chat_pool):
        """Test successful LLM classification to agent mode."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "agent",
            "confidence": 0.9,
        })
        result = await router._classify_with_llm("deploy to production")
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"
        assert result.confidence == 0.9
        assert result.method == "llm"

    async def test_successful_classification_assistant(self, router, mock_chat_pool):
        """Test successful LLM classification to assistant mode."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "assistant",
            "confidence": 0.85,
        })
        result = await router._classify_with_llm("remind me about meeting")
        assert result is not None
        assert result.mode == "assistant"
        assert result.model == "deepseek-chat"
        assert result.confidence == 0.85

    async def test_successful_classification_chat(self, router, mock_chat_pool):
        """Test successful LLM classification to chat mode."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "chat",
            "confidence": 0.7,
        })
        result = await router._classify_with_llm("explain quantum physics")
        assert result is not None
        assert result.mode == "chat"
        assert result.model == "deepseek-chat"

    async def test_invalid_mode_defaults_to_chat(self, router, mock_chat_pool):
        """Test invalid mode from LLM defaults to chat."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "invalid_mode",
            "confidence": 0.8,
        })
        result = await router._classify_with_llm("test")
        assert result is not None
        assert result.mode == "chat"

    async def test_missing_confidence_defaults(self, router, mock_chat_pool):
        """Test missing confidence field defaults to 0.5."""
        mock_chat_pool.get_router_provider().classify.return_value = json.dumps({
            "mode": "chat",
        })
        result = await router._classify_with_llm("test")
        assert result is not None
        assert result.confidence == 0.5

    async def test_json_parse_error_returns_none(self, router, mock_chat_pool):
        """Test JSON parse error returns None."""
        mock_chat_pool.get_router_provider().classify.return_value = "invalid json {"
        result = await router._classify_with_llm("test")
        assert result is None

    async def test_llm_exception_returns_none(self, router, mock_chat_pool):
        """Test LLM exception returns None."""
        mock_chat_pool.get_router_provider().classify.side_effect = Exception("API error")
        result = await router._classify_with_llm("test")
        assert result is None

    async def test_no_provider_returns_none(self, router, mock_chat_pool):
        """Test no provider available returns None."""
        mock_chat_pool.get_router_provider.return_value = None
        result = await router._classify_with_llm("test")
        assert result is None

    async def test_no_chat_pool_returns_none(self, router_no_pool):
        """Test no chat pool returns None."""
        result = await router_no_pool._classify_with_llm("test")
        assert result is None


# TestEscalate


class TestEscalate:
    """Test escalation logic."""

    def test_escalate_agent_to_heavy(self, router):
        """Test escalate from agent default to heavy model."""
        current = RoutingDecision(
            mode="agent",
            model="claude-sonnet-4-5",
            confidence=0.85,
            method="regex",
        )
        result = router.escalate(current)
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-opus-4-6"
        assert result.confidence == 0.85
        assert result.method == "escalation"

    def test_escalate_agent_heavy_returns_none(self, router):
        """Test escalate from heavy agent model returns None."""
        current = RoutingDecision(
            mode="agent",
            model="claude-opus-4-6",
            confidence=0.85,
            method="regex",
        )
        result = router.escalate(current)
        assert result is None

    def test_escalate_chat_default_to_fallback(self, router):
        """Test escalate from chat default to fallback."""
        current = RoutingDecision(
            mode="chat",
            model="deepseek-chat",
            confidence=0.7,
            method="llm",
        )
        result = router.escalate(current)
        assert result is not None
        assert result.mode == "chat"
        assert result.model == "gpt-4o-mini"
        assert result.method == "escalation"

    def test_escalate_chat_fallback_to_agent(self, router):
        """Test escalate from chat fallback to agent cross-mode."""
        current = RoutingDecision(
            mode="chat",
            model="gpt-4o-mini",
            confidence=0.6,
            method="llm",
        )
        result = router.escalate(current)
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"
        assert result.method == "escalation"

    def test_escalate_chat_agent_returns_none(self, router):
        """Test escalate from chat's agent fallback returns None."""
        # Chat chain ends at agent default
        current = RoutingDecision(
            mode="agent",
            model="claude-sonnet-4-5",
            confidence=0.5,
            method="escalation",
        )
        result = router.escalate(current)
        # Should escalate to agent heavy
        assert result is not None
        assert result.model == "claude-opus-4-6"

    def test_escalate_unknown_model_cross_mode(self, router):
        """Test escalate with unknown chat model triggers cross-mode."""
        current = RoutingDecision(
            mode="chat",
            model="unknown-model",
            confidence=0.5,
            method="llm",
        )
        result = router.escalate(current)
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"

    def test_escalate_preserves_confidence(self, router):
        """Test escalation preserves original confidence."""
        current = RoutingDecision(
            mode="agent",
            model="claude-sonnet-4-5",
            confidence=0.92,
            method="regex",
        )
        result = router.escalate(current)
        assert result.confidence == 0.92

    def test_escalate_with_missing_model(self, router, mock_settings):
        """Test escalation skips missing models in chain."""
        # Simulate missing fallback model
        mock_settings.model_chat_fallback = None
        current = RoutingDecision(
            mode="chat",
            model="deepseek-chat",
            confidence=0.7,
            method="llm",
        )
        result = router.escalate(current)
        # Should skip fallback and go to agent
        assert result is not None
        assert result.mode == "agent"
        assert result.model == "claude-sonnet-4-5"


# TestModeForModel


class TestModeForModel:
    """Test model to mode mapping."""

    def test_claude_sonnet_is_agent(self, router):
        """Test Claude Sonnet model maps to agent mode."""
        result = router._mode_for_model("claude-sonnet-4-5")
        assert result == "agent"

    def test_claude_opus_is_agent(self, router):
        """Test Claude Opus model maps to agent mode."""
        result = router._mode_for_model("claude-opus-4-6")
        assert result == "agent"

    def test_claude_haiku_is_agent(self, router):
        """Test Claude Haiku model maps to agent mode."""
        result = router._mode_for_model("claude-haiku-3-5")
        assert result == "agent"

    def test_gpt_is_chat(self, router):
        """Test GPT model maps to chat mode."""
        result = router._mode_for_model("gpt-4o")
        assert result == "chat"

    def test_deepseek_is_chat(self, router):
        """Test DeepSeek model maps to chat mode."""
        result = router._mode_for_model("deepseek-chat")
        assert result == "chat"

    def test_unknown_model_is_chat(self, router):
        """Test unknown model defaults to chat mode."""
        result = router._mode_for_model("some-random-model")
        assert result == "chat"

    def test_empty_string_is_chat(self, router):
        """Test empty string defaults to chat mode."""
        result = router._mode_for_model("")
        assert result == "chat"


# TestLogClassification


class TestLogClassification:
    """Test classification logging."""

    def test_appends_entry(self, router):
        """Test log appends classification entry."""
        decision = RoutingDecision(
            mode="agent",
            model="claude-sonnet-4-5",
            confidence=0.85,
            method="regex",
        )
        router._log_classification("test message", decision)
        assert len(router._intent_log) == 1
        entry = router._intent_log[0]
        assert "message_hash" in entry
        assert entry["mode"] == "agent"
        assert entry["confidence"] == 0.85
        assert entry["method"] == "regex"

    def test_message_hash_consistent(self, router):
        """Test same message produces same hash."""
        decision = RoutingDecision(
            mode="chat",
            model="deepseek-chat",
            confidence=0.7,
            method="llm",
        )
        router._log_classification("hello world", decision)
        router._log_classification("hello world", decision)
        hash1 = router._intent_log[0]["message_hash"]
        hash2 = router._intent_log[1]["message_hash"]
        assert hash1 == hash2

    def test_different_messages_different_hash(self, router):
        """Test different messages produce different hashes."""
        decision = RoutingDecision(
            mode="chat",
            model="deepseek-chat",
            confidence=0.7,
            method="llm",
        )
        router._log_classification("message one", decision)
        router._log_classification("message two", decision)
        hash1 = router._intent_log[0]["message_hash"]
        hash2 = router._intent_log[1]["message_hash"]
        assert hash1 != hash2

    def test_trims_at_1000(self, router):
        """Test log trims to 500 when reaching 1000 entries."""
        decision = RoutingDecision(
            mode="chat",
            model="test",
            confidence=0.5,
            method="test",
        )
        # Add 1001 entries
        for i in range(1001):
            router._log_classification(f"message {i}", decision)
        # Should be trimmed to last 500
        assert len(router._intent_log) == 500
        # First entry should be from message 501 onwards
        first_hash = router._intent_log[0]["message_hash"]
        # Verify it's not from the very first messages
        early_router = IntentRouter(router._settings, router._chat_pool)
        early_router._log_classification("message 0", decision)
        assert first_hash != early_router._intent_log[0]["message_hash"]

    def test_log_structure(self, router):
        """Test log entry has correct structure."""
        decision = RoutingDecision(
            mode="assistant",
            model="deepseek-chat",
            confidence=0.9,
            method="regex",
        )
        router._log_classification("test", decision)
        entry = router._intent_log[0]
        assert isinstance(entry["message_hash"], str)
        assert len(entry["message_hash"]) == 16
        assert entry["mode"] == "assistant"
        assert entry["confidence"] == 0.9
        assert entry["method"] == "regex"


# TestResolveModel


class TestResolveModel:
    """Test model resolution for modes."""

    def test_agent_mode_resolves_agent_default(self, router):
        """Test agent mode resolves to agent default model."""
        model = router._resolve_model("agent")
        assert model == "claude-sonnet-4-5"

    def test_assistant_mode_resolves_chat_default(self, router):
        """Test assistant mode resolves to chat default model."""
        model = router._resolve_model("assistant")
        assert model == "deepseek-chat"

    def test_chat_mode_resolves_chat_default(self, router):
        """Test chat mode resolves to chat default model."""
        model = router._resolve_model("chat")
        assert model == "deepseek-chat"

    def test_unknown_mode_resolves_chat_default(self, router):
        """Test unknown mode defaults to chat default model."""
        model = router._resolve_model("unknown")
        assert model == "deepseek-chat"


# TestChainConstants


class TestChainConstants:
    """Test escalation chain constants."""

    def test_agent_chain_structure(self):
        """Test AGENT_CHAIN has correct structure."""
        assert len(AGENT_CHAIN) == 2
        assert AGENT_CHAIN[0] == ("agent", "model_agent_default")
        assert AGENT_CHAIN[1] == ("agent", "model_agent_heavy")

    def test_chat_chain_structure(self):
        """Test CHAT_CHAIN has correct structure."""
        assert len(CHAT_CHAIN) == 3
        assert CHAT_CHAIN[0] == ("chat", "model_chat_default")
        assert CHAT_CHAIN[1] == ("chat", "model_chat_fallback")
        assert CHAT_CHAIN[2] == ("agent", "model_agent_default")

    def test_chat_chain_cross_mode_fallback(self):
        """Test CHAT_CHAIN includes agent as final fallback."""
        final_mode, _ = CHAT_CHAIN[-1]
        assert final_mode == "agent"


# TestPatternConstants


class TestPatternConstants:
    """Test regex pattern constants."""

    def test_agent_patterns_count(self):
        """Test AGENT_PATTERNS has expected number of patterns."""
        assert len(AGENT_PATTERNS) == 4

    def test_assistant_patterns_count(self):
        """Test ASSISTANT_PATTERNS has expected number of patterns."""
        assert len(ASSISTANT_PATTERNS) == 2

    def test_agent_pattern_english_matches(self):
        """Test agent pattern matches key English terms."""
        pattern = AGENT_PATTERNS[0]
        assert pattern.search("edit file")
        assert pattern.search("create script")
        assert pattern.search("run test")
        assert pattern.search("git commit")

    def test_agent_pattern_russian_matches(self):
        """Test agent pattern matches key Russian terms."""
        pattern = AGENT_PATTERNS[1]
        assert pattern.search("отредактируй файл")
        assert pattern.search("создай скрипт")
        assert pattern.search("запусти тест")

    def test_agent_pattern_file_extensions(self):
        """Test agent pattern matches file extensions."""
        pattern = AGENT_PATTERNS[2]
        assert pattern.search("test.py")
        assert pattern.search("index.js")
        assert pattern.search("config.yaml")

    def test_agent_pattern_code_fence(self):
        """Test agent pattern matches code fence."""
        pattern = AGENT_PATTERNS[3]
        assert pattern.search("```python")
        assert pattern.search("```")

    def test_assistant_pattern_reminder_matches(self):
        """Test assistant pattern matches reminder terms."""
        pattern = ASSISTANT_PATTERNS[0]
        assert pattern.search("remind me")
        assert pattern.search("напомни завтра")
        assert pattern.search("через 5 мин")

    def test_assistant_pattern_kb_matches(self):
        """Test assistant pattern matches knowledge base terms."""
        pattern = ASSISTANT_PATTERNS[1]
        assert pattern.search("knowledge base")
        assert pattern.search("найди ресторан")
        assert pattern.search("weather")
