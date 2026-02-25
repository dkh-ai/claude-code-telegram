"""Tests for multi-chat awareness and bot trigger detection.

Verifies ChatAwareness correctly detects chat types (private/group/supergroup),
bot mentions in text and entities, reply-to-bot detection, and determines
whether the bot should respond or silently observe.
"""

from unittest.mock import MagicMock

import pytest

from src.bot.chat_awareness import ChatAwareness, ChatContext


class TestChatContext:
    """Test ChatContext dataclass construction and attributes."""

    def test_create_private_chat_context(self):
        """ChatContext should construct with all private chat fields."""
        ctx = ChatContext(
            chat_id=123,
            chat_type="private",
            is_private=True,
            is_group=False,
            bot_mentioned=False,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert ctx.chat_id == 123
        assert ctx.chat_type == "private"
        assert ctx.is_private is True
        assert ctx.is_group is False
        assert ctx.bot_mentioned is False
        assert ctx.is_reply_to_bot is False
        assert ctx.thread_id is None
        assert ctx.user_id == 456

    def test_create_group_chat_context_with_mention(self):
        """ChatContext should construct with group chat fields and bot mention."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="supergroup",
            is_private=False,
            is_group=True,
            bot_mentioned=True,
            is_reply_to_bot=False,
            thread_id=12,
            user_id=456,
        )

        assert ctx.chat_id == 789
        assert ctx.chat_type == "supergroup"
        assert ctx.is_private is False
        assert ctx.is_group is True
        assert ctx.bot_mentioned is True
        assert ctx.is_reply_to_bot is False
        assert ctx.thread_id == 12
        assert ctx.user_id == 456

    def test_create_group_chat_context_with_reply(self):
        """ChatContext should construct with reply to bot flag."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=False,
            is_reply_to_bot=True,
            thread_id=None,
            user_id=456,
        )

        assert ctx.is_reply_to_bot is True
        assert ctx.bot_mentioned is False


@pytest.fixture
def chat_awareness():
    """Create a ChatAwareness instance."""
    return ChatAwareness()


class TestAnalyzePrivateChat:
    """Test analyze() for private chats."""

    def test_private_chat_basic(self, chat_awareness):
        """Private chat should set is_private=True and chat_type='private'."""
        update = MagicMock()
        update.effective_chat.id = 123
        update.effective_chat.type = "private"
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.chat_id == 123
        assert ctx.chat_type == "private"
        assert ctx.is_private is True
        assert ctx.is_group is False
        assert ctx.bot_mentioned is False
        assert ctx.is_reply_to_bot is False
        assert ctx.thread_id is None
        assert ctx.user_id == 456


class TestAnalyzeGroupChat:
    """Test analyze() for group and supergroup chats."""

    def test_group_chat_type(self, chat_awareness):
        """Group chat should set is_group=True and chat_type='group'."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hello everyone"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.chat_id == 789
        assert ctx.chat_type == "group"
        assert ctx.is_private is False
        assert ctx.is_group is True

    def test_supergroup_chat_type(self, chat_awareness):
        """Supergroup chat should set is_group=True and chat_type='supergroup'."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "supergroup"
        update.effective_message.text = "hello everyone"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.chat_type == "supergroup"
        assert ctx.is_group is True


class TestAnalyzeBotMention:
    """Test bot mention detection in text and entities."""

    def test_bot_mentioned_in_text(self, chat_awareness):
        """Bot mention in text should set bot_mentioned=True."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hey @testbot can you help?"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is True

    def test_bot_text_mention_is_case_sensitive(self, chat_awareness):
        """Bot text mention via 'in' operator is case-sensitive (only entity check is insensitive)."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hey @TestBot can you help?"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        # Text check is case-sensitive; entity check handles case-insensitive
        assert ctx.bot_mentioned is False

    def test_bot_not_mentioned_in_text(self, chat_awareness):
        """No bot mention should set bot_mentioned=False."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hey @otherbot can you help?"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is False

    def test_bot_mentioned_via_entity(self, chat_awareness):
        """Bot mention via message entity should set bot_mentioned=True."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@testbot help me"
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        # Create mention entity
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 8  # "@testbot"
        update.effective_message.entities = [entity]

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is True

    def test_bot_mentioned_via_entity_case_insensitive(self, chat_awareness):
        """Bot mention via entity should be case-insensitive."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@TestBot help me"
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 8
        update.effective_message.entities = [entity]

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is True

    def test_other_mention_entity_ignored(self, chat_awareness):
        """Mention of other users should not trigger bot_mentioned."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@otheruser help"
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 10
        update.effective_message.entities = [entity]

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is False

    def test_non_mention_entity_ignored(self, chat_awareness):
        """Non-mention entities (hashtag, url, etc.) should be ignored."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "#hashtag test"
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        entity = MagicMock()
        entity.type = "hashtag"
        entity.offset = 0
        entity.length = 8
        update.effective_message.entities = [entity]

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is False


class TestAnalyzeReplyToBot:
    """Test reply-to-bot detection."""

    def test_reply_to_bot_message(self, chat_awareness):
        """Reply to bot's message should set is_reply_to_bot=True."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "thanks!"
        update.effective_message.entities = []
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        # Create reply to bot's message
        reply = MagicMock()
        reply.from_user.username = "testbot"
        update.effective_message.reply_to_message = reply

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.is_reply_to_bot is True

    def test_reply_to_bot_case_insensitive(self, chat_awareness):
        """Reply to bot should be case-insensitive."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "thanks!"
        update.effective_message.entities = []
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        reply = MagicMock()
        reply.from_user.username = "TestBot"
        update.effective_message.reply_to_message = reply

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.is_reply_to_bot is True

    def test_reply_to_other_user(self, chat_awareness):
        """Reply to another user should set is_reply_to_bot=False."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "thanks!"
        update.effective_message.entities = []
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        reply = MagicMock()
        reply.from_user.username = "otheruser"
        update.effective_message.reply_to_message = reply

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.is_reply_to_bot is False

    def test_no_reply(self, chat_awareness):
        """No reply should set is_reply_to_bot=False."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.is_reply_to_bot is False

    def test_reply_to_message_without_username(self, chat_awareness):
        """Reply to message where user has no username should not crash."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "thanks!"
        update.effective_message.entities = []
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        reply = MagicMock()
        reply.from_user.username = None
        update.effective_message.reply_to_message = reply

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.is_reply_to_bot is False


class TestAnalyzeThreadId:
    """Test thread_id extraction."""

    def test_thread_id_present(self, chat_awareness):
        """Thread ID should be extracted when present."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "supergroup"
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = 42
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.thread_id == 42

    def test_thread_id_none(self, chat_awareness):
        """Thread ID should be None when not present."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.thread_id is None


class TestAnalyzeEdgeCases:
    """Test analyze() with missing or None values."""

    def test_no_chat(self, chat_awareness):
        """Update with no chat should return default values."""
        update = MagicMock()
        update.effective_chat = None
        update.effective_message = MagicMock()
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.chat_id == 0
        assert ctx.chat_type == "private"
        assert ctx.is_private is True

    def test_no_message(self, chat_awareness):
        """Update with no message should not crash."""
        update = MagicMock()
        update.effective_chat.id = 123
        update.effective_chat.type = "private"
        update.effective_message = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.chat_id == 123
        assert ctx.bot_mentioned is False
        assert ctx.is_reply_to_bot is False
        assert ctx.thread_id is None

    def test_no_user(self, chat_awareness):
        """Update with no user should return user_id=0."""
        update = MagicMock()
        update.effective_chat.id = 123
        update.effective_chat.type = "private"
        update.effective_message.text = "hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user = None

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.user_id == 0

    def test_no_message_text(self, chat_awareness):
        """Message with no text should not crash bot mention detection."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = None
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.bot_mentioned is False

    def test_empty_bot_username(self, chat_awareness):
        """Empty bot username should not crash mention detection."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@testbot hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "")

        assert ctx.bot_mentioned is False

    def test_none_bot_username(self, chat_awareness):
        """None bot username should not crash mention detection."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@testbot hello"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, None)

        assert ctx.bot_mentioned is False


class TestShouldRespond:
    """Test should_respond() decision logic."""

    def test_always_respond_in_private_chat(self, chat_awareness):
        """Bot should always respond in private chats."""
        ctx = ChatContext(
            chat_id=123,
            chat_type="private",
            is_private=True,
            is_group=False,
            bot_mentioned=False,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_respond(ctx) is True

    def test_respond_when_mentioned_in_group(self, chat_awareness):
        """Bot should respond when mentioned in group."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=True,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_respond(ctx) is True

    def test_respond_when_replied_in_group(self, chat_awareness):
        """Bot should respond when someone replies to it in group."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=False,
            is_reply_to_bot=True,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_respond(ctx) is True

    def test_respond_when_mentioned_and_replied_in_group(self, chat_awareness):
        """Bot should respond when both mentioned and replied in group."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=True,
            is_reply_to_bot=True,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_respond(ctx) is True

    def test_do_not_respond_in_group_without_trigger(self, chat_awareness):
        """Bot should not respond in group without mention or reply."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=False,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_respond(ctx) is False


class TestShouldObserve:
    """Test should_observe() decision logic."""

    def test_observe_in_group_without_trigger(self, chat_awareness):
        """Bot should observe (extract memory) in group without trigger."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=False,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_observe(ctx) is True

    def test_do_not_observe_when_responding_in_group(self, chat_awareness):
        """Bot should not observe when it's responding (mentioned)."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=True,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_observe(ctx) is False

    def test_do_not_observe_when_responding_to_reply(self, chat_awareness):
        """Bot should not observe when it's responding (replied)."""
        ctx = ChatContext(
            chat_id=789,
            chat_type="group",
            is_private=False,
            is_group=True,
            bot_mentioned=False,
            is_reply_to_bot=True,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_observe(ctx) is False

    def test_do_not_observe_in_private_chat(self, chat_awareness):
        """Bot should not observe in private chats (always responds)."""
        ctx = ChatContext(
            chat_id=123,
            chat_type="private",
            is_private=True,
            is_group=False,
            bot_mentioned=False,
            is_reply_to_bot=False,
            thread_id=None,
            user_id=456,
        )

        assert chat_awareness.should_observe(ctx) is False


class TestIntegration:
    """Integration tests combining analyze() with decision methods."""

    def test_private_chat_flow(self, chat_awareness):
        """Private chat should always trigger response, never observe."""
        update = MagicMock()
        update.effective_chat.id = 123
        update.effective_chat.type = "private"
        update.effective_message.text = "help me"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert chat_awareness.should_respond(ctx) is True
        assert chat_awareness.should_observe(ctx) is False

    def test_group_chat_with_mention_flow(self, chat_awareness):
        """Group chat with mention should respond, not observe."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "@testbot help"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert chat_awareness.should_respond(ctx) is True
        assert chat_awareness.should_observe(ctx) is False

    def test_group_chat_without_trigger_flow(self, chat_awareness):
        """Group chat without trigger should observe, not respond."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "group"
        update.effective_message.text = "just chatting"
        update.effective_message.entities = []
        update.effective_message.reply_to_message = None
        update.effective_message.message_thread_id = None
        update.effective_user.id = 456

        ctx = chat_awareness.analyze(update, "testbot")

        assert chat_awareness.should_respond(ctx) is False
        assert chat_awareness.should_observe(ctx) is True

    def test_supergroup_with_reply_flow(self, chat_awareness):
        """Supergroup with reply to bot should respond, not observe."""
        update = MagicMock()
        update.effective_chat.id = 789
        update.effective_chat.type = "supergroup"
        update.effective_message.text = "thanks!"
        update.effective_message.entities = []
        update.effective_message.message_thread_id = 42
        update.effective_user.id = 456

        reply = MagicMock()
        reply.from_user.username = "testbot"
        update.effective_message.reply_to_message = reply

        ctx = chat_awareness.analyze(update, "testbot")

        assert ctx.thread_id == 42
        assert chat_awareness.should_respond(ctx) is True
        assert chat_awareness.should_observe(ctx) is False
