"""Tests for TaskNotificationHandler."""

from unittest.mock import AsyncMock

import pytest

from src.events.bus import Event, EventBus
from src.events.types import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskProgressEvent,
    TaskTimeoutEvent,
)
from src.notifications.task_notifications import TaskNotificationHandler


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_bot() -> AsyncMock:
    bot = AsyncMock()
    bot.send_message = AsyncMock()
    return bot


@pytest.fixture
def handler(event_bus: EventBus, mock_bot: AsyncMock) -> TaskNotificationHandler:
    h = TaskNotificationHandler(event_bus=event_bus, bot=mock_bot)
    h.register()
    return h


class TestHandleProgress:
    """Tests for progress notification handling."""

    async def test_sends_progress_message(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends formatted progress notification."""
        event = TaskProgressEvent(
            task_id="abc12345",
            elapsed_seconds=125,
            cost=0.42,
            stage="Running tests",
            chat_id=100,
            message_thread_id=5,
        )

        await handler.handle_progress(event)

        mock_bot.send_message.assert_called_once()
        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 100
        assert kwargs["message_thread_id"] == 5
        assert kwargs["parse_mode"] == "HTML"
        assert "abc12345" in kwargs["text"]
        assert "2m 5s" in kwargs["text"]
        assert "$0.42" in kwargs["text"]
        assert "Running tests" in kwargs["text"]

    async def test_ignores_non_progress_events(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Ignores events that are not TaskProgressEvent."""
        event = Event(source="test")
        await handler.handle_progress(event)
        mock_bot.send_message.assert_not_called()


class TestHandleCompleted:
    """Tests for completion notification handling."""

    async def test_sends_completion_message(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends formatted completion notification."""
        event = TaskCompletedEvent(
            task_id="done1234",
            duration_seconds=300,
            cost=1.50,
            commits=[
                {"sha": "abc1234", "message": "[claude] Fix bug"},
                {"sha": "def5678", "message": "[claude] Add tests"},
            ],
            result_summary="Fixed 3 bugs and added tests",
            chat_id=200,
        )

        await handler.handle_completed(event)

        mock_bot.send_message.assert_called_once()
        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 200
        text = kwargs["text"]
        assert "done1234" in text
        assert "5m 0s" in text
        assert "$1.50" in text
        assert "Коммитов: 2" in text
        assert "abc1234" in text
        assert "Fixed 3 bugs" in text

    async def test_completion_without_commits(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends completion notification without commits section."""
        event = TaskCompletedEvent(
            task_id="done5678",
            duration_seconds=60,
            cost=0.25,
            commits=[],
            result_summary="",
            chat_id=200,
        )

        await handler.handle_completed(event)

        text = mock_bot.send_message.call_args[1]["text"]
        assert "Коммитов" not in text

    async def test_ignores_non_completed_events(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Ignores events that are not TaskCompletedEvent."""
        event = Event(source="test")
        await handler.handle_completed(event)
        mock_bot.send_message.assert_not_called()


class TestHandleFailed:
    """Tests for failure notification handling."""

    async def test_sends_failure_with_keyboard(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends failure notification with action buttons."""
        event = TaskFailedEvent(
            task_id="fail1234",
            duration_seconds=120,
            cost=0.75,
            error_message="Connection timed out",
            last_output="Running step 3...",
            chat_id=300,
        )

        await handler.handle_failed(event)

        mock_bot.send_message.assert_called_once()
        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 300
        text = kwargs["text"]
        assert "fail1234" in text
        assert "Connection timed out" in text
        assert "$0.75" in text

        # Check inline keyboard
        reply_markup = kwargs["reply_markup"]
        assert reply_markup is not None
        buttons = reply_markup.inline_keyboard[0]
        callback_data_values = [b.callback_data for b in buttons]
        assert "tasklog:fail1234" in callback_data_values
        assert "taskretry:fail1234" in callback_data_values

    async def test_ignores_non_failed_events(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Ignores events that are not TaskFailedEvent."""
        event = Event(source="test")
        await handler.handle_failed(event)
        mock_bot.send_message.assert_not_called()


class TestHandleTimeout:
    """Tests for timeout notification handling."""

    async def test_sends_timeout_with_keyboard(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends timeout notification with restart/stop buttons."""
        event = TaskTimeoutEvent(
            task_id="hang1234",
            duration_seconds=600,
            cost=2.00,
            idle_seconds=305,
            chat_id=400,
            message_thread_id=10,
        )

        await handler.handle_timeout(event)

        mock_bot.send_message.assert_called_once()
        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 400
        assert kwargs["message_thread_id"] == 10
        text = kwargs["text"]
        assert "hang1234" in text
        assert "5m 5s" in text

        # Check inline keyboard
        reply_markup = kwargs["reply_markup"]
        buttons = reply_markup.inline_keyboard[0]
        callback_data_values = [b.callback_data for b in buttons]
        assert "taskretry:hang1234" in callback_data_values
        assert "taskstop:hang1234" in callback_data_values

    async def test_ignores_non_timeout_events(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Ignores events that are not TaskTimeoutEvent."""
        event = Event(source="test")
        await handler.handle_timeout(event)
        mock_bot.send_message.assert_not_called()


class TestSend:
    """Tests for the internal _send method."""

    async def test_skips_when_no_chat_id(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Skips sending when chat_id is 0 or falsy."""
        await handler._send(0, "test message")
        mock_bot.send_message.assert_not_called()

    async def test_handles_telegram_error(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Logs error but does not raise on TelegramError."""
        from telegram.error import TelegramError

        mock_bot.send_message.side_effect = TelegramError("Chat not found")

        # Should not raise
        await handler._send(999, "test message")

    async def test_passes_thread_id(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Passes message_thread_id when provided."""
        await handler._send(100, "test", message_thread_id=42)

        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["message_thread_id"] == 42

    async def test_sends_without_thread_id(
        self, handler: TaskNotificationHandler, mock_bot: AsyncMock
    ) -> None:
        """Sends without thread_id when not provided."""
        await handler._send(100, "test")

        kwargs = mock_bot.send_message.call_args[1]
        assert kwargs["message_thread_id"] is None


class TestRegister:
    """Tests for event bus subscription."""

    def test_register_subscribes_to_all_task_events(
        self, event_bus: EventBus, mock_bot: AsyncMock
    ) -> None:
        """Registers handlers for all 4 task event types."""
        h = TaskNotificationHandler(event_bus=event_bus, bot=mock_bot)
        h.register()

        # Verify subscriptions exist
        assert TaskProgressEvent in event_bus._handlers
        assert TaskCompletedEvent in event_bus._handlers
        assert TaskFailedEvent in event_bus._handlers
        assert TaskTimeoutEvent in event_bus._handlers
