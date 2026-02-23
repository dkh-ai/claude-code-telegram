"""Multi-chat awareness — understand chat type and bot trigger context."""

from dataclasses import dataclass
from typing import Optional

from telegram import Update


@dataclass
class ChatContext:
    """Context extracted from a Telegram update."""

    chat_id: int
    chat_type: str  # "private" | "group" | "supergroup"
    is_private: bool
    is_group: bool
    bot_mentioned: bool
    is_reply_to_bot: bool
    thread_id: Optional[int]
    user_id: int


class ChatAwareness:
    """Analyze Telegram updates for chat context and response decisions."""

    def analyze(self, update: Update, bot_username: str) -> ChatContext:
        """Extract chat context from a Telegram update."""
        chat = update.effective_chat
        message = update.effective_message
        user = update.effective_user

        chat_id = chat.id if chat else 0
        chat_type = getattr(chat, "type", "private") if chat else "private"
        is_private = chat_type == "private"
        is_group = chat_type in ("group", "supergroup")
        user_id = user.id if user else 0

        # Check if bot was mentioned
        bot_mentioned = False
        if message and message.text and bot_username:
            bot_mentioned = f"@{bot_username}" in message.text

        # Check if bot's message entities mention the bot
        if message and message.entities:
            for entity in message.entities:
                if entity.type == "mention":
                    mention_text = message.text[
                        entity.offset : entity.offset + entity.length
                    ]
                    if mention_text.lower() == f"@{bot_username.lower()}":
                        bot_mentioned = True
                        break

        # Check if this is a reply to bot's message
        is_reply_to_bot = False
        if message and message.reply_to_message:
            reply_from = message.reply_to_message.from_user
            if reply_from and reply_from.username:
                is_reply_to_bot = reply_from.username.lower() == bot_username.lower()

        # Thread/topic ID
        thread_id = getattr(message, "message_thread_id", None) if message else None

        return ChatContext(
            chat_id=chat_id,
            chat_type=chat_type,
            is_private=is_private,
            is_group=is_group,
            bot_mentioned=bot_mentioned,
            is_reply_to_bot=is_reply_to_bot,
            thread_id=thread_id,
            user_id=user_id,
        )

    def should_respond(self, ctx: ChatContext) -> bool:
        """Determine if bot should respond in this context."""
        if ctx.is_private:
            return True
        return ctx.bot_mentioned or ctx.is_reply_to_bot

    def should_observe(self, ctx: ChatContext) -> bool:
        """Should bot silently observe (extract memory) without responding?"""
        return ctx.is_group and not self.should_respond(ctx)
