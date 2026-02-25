"""Test FactExtractor — LLM-based fact extraction and summarization."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.extractor import EXTRACT_FACTS_SYSTEM, SUMMARIZE_SYSTEM, FactExtractor
from src.memory.models import UserFact


class TestFactExtractor:
    """Test FactExtractor for fact extraction and conversation summarization."""

    def test_construction_without_provider(self) -> None:
        """Test FactExtractor can be created without a provider."""
        extractor = FactExtractor()
        assert extractor._provider is None

    def test_construction_with_provider(self) -> None:
        """Test FactExtractor can be created with a chat provider."""
        provider = MagicMock()
        extractor = FactExtractor(chat_provider=provider)
        assert extractor._provider is provider

    async def test_extract_returns_empty_when_no_provider(self) -> None:
        """Test extract() returns empty list when no provider is configured."""
        extractor = FactExtractor()
        facts = await extractor.extract(
            user_message="I love Python programming",
            bot_response="That's great! Python is very popular.",
            user_id=123,
        )
        assert facts == []

    async def test_extract_returns_empty_when_messages_too_short(self) -> None:
        """Test extract() returns empty list when both messages are too short."""
        provider = MagicMock()
        extractor = FactExtractor(chat_provider=provider)

        # Both messages < 10 chars
        facts = await extractor.extract(
            user_message="Hi",
            bot_response="Hello",
            user_id=123,
        )
        assert facts == []

        # User message >= 10 but bot response < 10
        facts = await extractor.extract(
            user_message="Hello there!",
            bot_response="Hi",
            user_id=123,
        )
        assert facts == []

        # Bot response >= 10 but user message < 10
        facts = await extractor.extract(
            user_message="Hi",
            bot_response="Hello there!",
            user_id=123,
        )
        assert facts == []

    async def test_extract_parses_valid_json_array(self) -> None:
        """Test extract() successfully parses valid JSON array from provider."""
        provider = MagicMock()
        provider.classify = AsyncMock(
            return_value=json.dumps(
                [
                    {"category": "preference", "fact": "likes Python"},
                    {"category": "work", "fact": "works as software engineer"},
                ]
            )
        )

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I love Python and I work as a software engineer",
            bot_response="That's wonderful! Python is great for software development.",
            user_id=456,
        )

        assert len(facts) == 2
        assert facts[0].user_id == 456
        assert facts[0].category == "preference"
        assert facts[0].fact == "likes Python"
        assert facts[0].source == "auto_extract"
        assert facts[1].category == "work"
        assert facts[1].fact == "works as software engineer"
        assert facts[1].source == "auto_extract"

        # Verify provider was called with correct arguments
        provider.classify.assert_called_once()
        call_args = provider.classify.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == EXTRACT_FACTS_SYSTEM

    async def test_extract_handles_invalid_json(self) -> None:
        """Test extract() handles invalid JSON gracefully."""
        provider = MagicMock()
        provider.classify = AsyncMock(return_value="not valid json {")

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I like to code in Python",
            bot_response="Python is a great language!",
            user_id=123,
        )

        assert facts == []

    async def test_extract_handles_non_list_json(self) -> None:
        """Test extract() returns empty list when JSON is not a list."""
        provider = MagicMock()
        provider.classify = AsyncMock(
            return_value=json.dumps({"category": "preference", "fact": "likes Python"})
        )

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I like Python",
            bot_response="Great choice!",
            user_id=789,
        )

        assert facts == []

    async def test_extract_handles_items_missing_category(self) -> None:
        """Test extract() skips items missing category field."""
        provider = MagicMock()
        provider.classify = AsyncMock(
            return_value=json.dumps(
                [
                    {"category": "preference", "fact": "likes Python"},
                    {"fact": "missing category field"},  # no category
                    {"category": "work", "fact": "software engineer"},
                ]
            )
        )

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I love Python and work as an engineer",
            bot_response="Excellent!",
            user_id=100,
        )

        # Should only extract the items with both category and fact
        assert len(facts) == 2
        assert facts[0].fact == "likes Python"
        assert facts[1].fact == "software engineer"

    async def test_extract_handles_items_missing_fact(self) -> None:
        """Test extract() skips items missing fact field."""
        provider = MagicMock()
        provider.classify = AsyncMock(
            return_value=json.dumps(
                [
                    {"category": "preference", "fact": "likes Python"},
                    {"category": "personal"},  # no fact
                    {"category": "work", "fact": "software engineer"},
                ]
            )
        )

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I love Python and work as an engineer",
            bot_response="Excellent!",
            user_id=200,
        )

        assert len(facts) == 2
        assert facts[0].fact == "likes Python"
        assert facts[1].fact == "software engineer"

    async def test_extract_handles_non_dict_items(self) -> None:
        """Test extract() skips non-dict items in the array."""
        provider = MagicMock()
        provider.classify = AsyncMock(
            return_value=json.dumps(
                [
                    {"category": "preference", "fact": "likes Python"},
                    "not a dict",
                    123,
                    {"category": "work", "fact": "software engineer"},
                ]
            )
        )

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I love Python and work as an engineer",
            bot_response="Excellent!",
            user_id=300,
        )

        assert len(facts) == 2
        assert facts[0].fact == "likes Python"
        assert facts[1].fact == "software engineer"

    async def test_extract_truncates_long_messages(self) -> None:
        """Test extract() truncates messages to 500 characters."""
        provider = MagicMock()
        provider.classify = AsyncMock(return_value="[]")

        extractor = FactExtractor(chat_provider=provider)

        long_message = "a" * 1000
        long_response = "b" * 1000

        await extractor.extract(
            user_message=long_message,
            bot_response=long_response,
            user_id=400,
        )

        # Verify the exchange passed to provider was truncated
        call_args = provider.classify.call_args
        exchange = call_args.kwargs["prompt"]
        assert len(exchange) < 1100  # "User: " + 500 + "\nAssistant: " + 500

    async def test_extract_handles_exception_from_provider(self) -> None:
        """Test extract() handles exceptions from provider gracefully."""
        provider = MagicMock()
        provider.classify = AsyncMock(side_effect=Exception("Provider error"))

        extractor = FactExtractor(chat_provider=provider)
        facts = await extractor.extract(
            user_message="I like Python",
            bot_response="Great!",
            user_id=500,
        )

        assert facts == []

    async def test_summarize_returns_none_when_no_provider(self) -> None:
        """Test summarize() returns None when no provider is configured."""
        extractor = FactExtractor()
        summary = await extractor.summarize(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        assert summary is None

    async def test_summarize_returns_none_for_less_than_two_messages(self) -> None:
        """Test summarize() returns None when there are fewer than 2 messages."""
        provider = MagicMock()
        extractor = FactExtractor(chat_provider=provider)

        # Empty list
        summary = await extractor.summarize([])
        assert summary is None

        # Single message
        summary = await extractor.summarize([{"role": "user", "content": "Hello"}])
        assert summary is None

    async def test_summarize_returns_content_from_provider(self) -> None:
        """Test summarize() returns content from provider's chat response."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "Summary of conversation"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)
        messages = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language"},
            {"role": "user", "content": "What about async?"},
            {"role": "assistant", "content": "Async allows concurrent execution"},
        ]

        summary = await extractor.summarize(messages)

        assert summary == "Summary of conversation"

        # Verify provider.chat was called with correct arguments
        provider.chat.assert_called_once()
        call_args = provider.chat.call_args
        assert "messages" in call_args.kwargs
        assert "max_tokens" in call_args.kwargs
        assert "temperature" in call_args.kwargs
        assert call_args.kwargs["max_tokens"] == 200
        assert call_args.kwargs["temperature"] == 0.3

        # Verify system message contains SUMMARIZE_SYSTEM
        messages_arg = call_args.kwargs["messages"]
        assert len(messages_arg) == 2
        assert messages_arg[0]["role"] == "system"
        assert messages_arg[0]["content"] == SUMMARIZE_SYSTEM

    async def test_summarize_limits_to_last_20_messages(self) -> None:
        """Test summarize() only includes last 20 messages in transcript."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "Summary"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)

        # Create 25 messages
        messages = []
        for i in range(25):
            messages.append({"role": "user", "content": f"Message {i}"})

        await extractor.summarize(messages)

        # Verify the transcript only includes last 20 messages
        call_args = provider.chat.call_args
        messages_arg = call_args.kwargs["messages"]
        transcript = messages_arg[1]["content"]

        # Should contain "Message 24" (last) but not "Message 0" (first)
        assert "Message 24" in transcript
        assert "Message 0" not in transcript

    async def test_summarize_truncates_message_content(self) -> None:
        """Test summarize() truncates message content to 200 characters."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "Summary"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)

        long_content = "x" * 500
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "Reply"},
        ]

        await extractor.summarize(messages)

        call_args = provider.chat.call_args
        messages_arg = call_args.kwargs["messages"]
        transcript = messages_arg[1]["content"]

        # The transcript should have truncated content
        # Should not contain all 500 'x' characters
        assert transcript.count("x") <= 200

    async def test_summarize_handles_messages_without_role(self) -> None:
        """Test summarize() handles messages missing role field."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "Summary"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)

        messages = [
            {"content": "Message without role"},
            {"role": "assistant", "content": "Reply"},
        ]

        summary = await extractor.summarize(messages)

        # Should still work, defaulting to "user"
        assert summary == "Summary"

    async def test_summarize_handles_messages_without_content(self) -> None:
        """Test summarize() handles messages missing content field."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "Summary"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)

        messages = [
            {"role": "user"},
            {"role": "assistant", "content": "Reply"},
        ]

        summary = await extractor.summarize(messages)

        # Should still work, defaulting to empty string
        assert summary == "Summary"

    async def test_summarize_handles_exception_from_provider(self) -> None:
        """Test summarize() handles exceptions from provider gracefully."""
        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=Exception("Provider error"))

        extractor = FactExtractor(chat_provider=provider)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        summary = await extractor.summarize(messages)

        assert summary is None

    async def test_summarize_strips_whitespace_from_response(self) -> None:
        """Test summarize() strips leading/trailing whitespace from response."""
        provider = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.content = "   Summary with whitespace   \n\n"
        provider.chat = AsyncMock(return_value=mock_chat_response)

        extractor = FactExtractor(chat_provider=provider)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        summary = await extractor.summarize(messages)

        assert summary == "Summary with whitespace"
