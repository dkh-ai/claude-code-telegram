"""Task notification handler -- delivers task lifecycle events to Telegram.

Subscribes to Task*Events on the event bus and sends formatted messages
with inline keyboards to the appropriate Telegram chats/threads.
"""

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from ..events.bus import Event, EventBus
from ..events.types import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskProgressEvent,
    TaskTimeoutEvent,
)

logger = structlog.get_logger()


class TaskNotificationHandler:
    """Delivers task lifecycle notifications to Telegram."""

    def __init__(self, event_bus: EventBus, bot: Bot) -> None:
        self.event_bus = event_bus
        self.bot = bot

    def register(self) -> None:
        """Subscribe to task events."""
        self.event_bus.subscribe(TaskProgressEvent, self.handle_progress)
        self.event_bus.subscribe(TaskCompletedEvent, self.handle_completed)
        self.event_bus.subscribe(TaskFailedEvent, self.handle_failed)
        self.event_bus.subscribe(TaskTimeoutEvent, self.handle_timeout)

    async def handle_progress(self, event: Event) -> None:
        """Send heartbeat notification."""
        if not isinstance(event, TaskProgressEvent):
            return
        minutes, seconds = divmod(event.elapsed_seconds, 60)
        text = (
            f"🔄 <code>{event.task_id}</code> | "
            f"⏱ {minutes}m {seconds}s | "
            f"💰 ${event.cost:.2f}\n"
            f"📍 {event.stage}"
        )
        await self._send(event.chat_id, text, event.message_thread_id)

    async def handle_completed(self, event: Event) -> None:
        """Send completion report."""
        if not isinstance(event, TaskCompletedEvent):
            return
        minutes, seconds = divmod(event.duration_seconds, 60)
        lines = [
            f"✅ <b>Задача <code>{event.task_id}</code> завершена!</b>\n",
            f"⏱ Время: {minutes}m {seconds}s",
            f"💰 Стоимость: ${event.cost:.2f}",
        ]
        if event.commits:
            lines.append(f"📝 Коммитов: {len(event.commits)}")
            for c in event.commits[:5]:
                sha = c.get("sha", c.get("hash", "?"))
                msg = c.get("message", "")
                lines.append(f"   • <code>{sha}</code> {msg}")
        if event.result_summary:
            lines.append(f"\n📋 {event.result_summary[:300]}")
        await self._send(event.chat_id, "\n".join(lines), event.message_thread_id)

    async def handle_failed(self, event: Event) -> None:
        """Send failure notification with action buttons."""
        if not isinstance(event, TaskFailedEvent):
            return
        minutes, seconds = divmod(event.duration_seconds, 60)
        text = (
            f"❌ <b>Задача <code>{event.task_id}</code> завершилась с ошибкой</b>\n\n"
            f"⏱ Время: {minutes}m {seconds}s\n"
            f"💰 Стоимость: ${event.cost:.2f}\n"
            f"📋 Ошибка: {event.error_message[:200]}"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "📋 Лог", callback_data=f"tasklog:{event.task_id}"
                ),
                InlineKeyboardButton(
                    "🔄 Повторить", callback_data=f"taskretry:{event.task_id}"
                ),
            ]
        ])
        await self._send(
            event.chat_id, text, event.message_thread_id, reply_markup=keyboard
        )

    async def handle_timeout(self, event: Event) -> None:
        """Send timeout warning with action buttons."""
        if not isinstance(event, TaskTimeoutEvent):
            return
        idle_min = event.idle_seconds // 60
        idle_sec = event.idle_seconds % 60
        text = (
            f"⚠️ <b>Задача <code>{event.task_id}</code> не отвечает</b>\n\n"
            f"Нет активности уже {idle_min}m {idle_sec}s"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "🔄 Перезапустить",
                    callback_data=f"taskretry:{event.task_id}",
                ),
                InlineKeyboardButton(
                    "⏹ Остановить",
                    callback_data=f"taskstop:{event.task_id}",
                ),
            ]
        ])
        await self._send(
            event.chat_id, text, event.message_thread_id, reply_markup=keyboard
        )

    async def _send(
        self,
        chat_id: int,
        text: str,
        message_thread_id: int | None = None,
        **kwargs: object,
    ) -> None:
        """Send message to chat, optionally in a forum thread."""
        if not chat_id:
            logger.warning("Task notification skipped: no chat_id")
            return
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
                message_thread_id=message_thread_id,
                **kwargs,
            )
        except TelegramError as e:
            logger.error(
                "Failed to send task notification",
                chat_id=chat_id,
                error=str(e),
            )
