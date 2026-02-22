"""Tests for background task command handlers."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from src.bot.handlers.task_handlers import (
    _get_project_path,
    task_command,
    taskcontinue_command,
    tasklog_command,
    taskstatus_command,
    taskstop_command,
)
from src.tasks.models import BackgroundTask


def _make_context(
    bot_data: dict | None = None,
    user_data: dict | None = None,
    args: list | None = None,
) -> MagicMock:
    """Create a mock context with configurable bot_data and user_data."""
    ctx = MagicMock()
    ctx.bot_data = bot_data or {}
    ctx.user_data = user_data or {}
    ctx.args = args or []
    return ctx


def _make_update(
    text: str = "/task",
    user_id: int = 42,
    chat_id: int = 100,
    message_thread_id: int | None = None,
) -> MagicMock:
    """Create a mock update with message, user, and chat."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    update.message.text = text
    update.message.message_thread_id = message_thread_id
    return update


def _make_task(
    task_id: str = "abcd1234",
    status: str = "running",
    project_path: Path | None = None,
    **kwargs: object,
) -> BackgroundTask:
    """Build a BackgroundTask with sensible defaults."""
    defaults: dict = dict(
        task_id=task_id,
        user_id=42,
        project_path=project_path or Path("/projects/myapp"),
        prompt="Fix the bug",
        status=status,
        chat_id=100,
        message_thread_id=None,
        created_at=datetime.now(timezone.utc),
        last_activity_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return BackgroundTask(**defaults)


class TestTaskCommand:
    """Tests for /task handler."""

    async def test_no_args_shows_usage(self) -> None:
        """Shows usage when called without arguments."""
        update = _make_update()
        context = _make_context(args=[])

        await task_command(update, context)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Использование" in text

    async def test_no_task_manager_shows_error(self) -> None:
        """Shows error when task_manager is not configured."""
        update = _make_update()
        context = _make_context(args=["do", "something"])

        await task_command(update, context)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "не настроены" in text

    async def test_no_project_path_shows_error(self) -> None:
        """Shows error when project path cannot be determined."""
        update = _make_update()
        task_manager = AsyncMock()
        context = _make_context(
            bot_data={"task_manager": task_manager},
            args=["fix", "the", "bug"],
        )

        await task_command(update, context)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "определить проект" in text

    async def test_successful_task_start(self) -> None:
        """Starts a task and sends confirmation."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.start_task = AsyncMock(return_value="aabb1122")

        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
            args=["fix", "the", "bug"],
        )

        await task_command(update, context)

        task_manager.start_task.assert_called_once()
        call_kwargs = task_manager.start_task.call_args[1]
        assert call_kwargs["prompt"] == "fix the bug"
        assert call_kwargs["project_path"] == Path("/projects/myapp")

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "aabb1122" in text
        assert "Задача запущена" in text

    async def test_value_error_shows_error(self) -> None:
        """Shows error when start_task raises ValueError."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.start_task = AsyncMock(
            side_effect=ValueError("Already running")
        )

        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
            args=["fix", "bug"],
        )

        await task_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Already running" in text


class TestTaskstatusCommand:
    """Tests for /taskstatus handler."""

    async def test_no_task_manager(self) -> None:
        """Shows error when task_manager is not configured."""
        update = _make_update()
        context = _make_context()

        await taskstatus_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "не настроены" in text

    async def test_no_running_tasks(self) -> None:
        """Shows message when no tasks are running."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.get_all_running = AsyncMock(return_value=[])
        context = _make_context(bot_data={"task_manager": task_manager})

        await taskstatus_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Нет активных" in text

    async def test_shows_running_tasks(self) -> None:
        """Lists running tasks with details."""
        update = _make_update()
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]
        task_manager = AsyncMock()
        task_manager.get_all_running = AsyncMock(return_value=tasks)
        context = _make_context(bot_data={"task_manager": task_manager})

        await taskstatus_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "t1" in text
        assert "t2" in text
        assert "Активные задачи" in text


class TestTasklogCommand:
    """Tests for /tasklog handler."""

    async def test_no_task_manager(self) -> None:
        """Shows error when task_manager is not configured."""
        update = _make_update()
        context = _make_context()

        await tasklog_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "не настроены" in text

    async def test_no_active_tasks(self) -> None:
        """Shows message when no tasks are running."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=None)
        task_manager.get_all_running = AsyncMock(return_value=[])
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
        )

        await tasklog_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Нет активных" in text

    async def test_shows_task_output(self) -> None:
        """Shows task output when available."""
        update = _make_update()
        task = _make_task(last_output="Building project...")
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=task)
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
        )

        await tasklog_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Building project..." in text
        assert task.task_id in text

    async def test_shows_no_output_placeholder(self) -> None:
        """Shows placeholder when task has no output."""
        update = _make_update()
        task = _make_task(last_output=None)
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=task)
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
        )

        await tasklog_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "нет вывода" in text


class TestTaskstopCommand:
    """Tests for /taskstop handler."""

    async def test_no_task_manager(self) -> None:
        """Shows error when task_manager is not configured."""
        update = _make_update()
        context = _make_context()

        await taskstop_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "не настроены" in text

    async def test_stop_by_task_id(self) -> None:
        """Stops a task by explicit ID argument."""
        update = _make_update()
        task = _make_task(task_id="abc12345")
        task_manager = AsyncMock()
        task_manager.get_task = AsyncMock(return_value=task)
        task_manager.stop_task = AsyncMock()
        context = _make_context(
            bot_data={"task_manager": task_manager},
            args=["abc12345"],
        )

        await taskstop_command(update, context)

        task_manager.stop_task.assert_called_once_with("abc12345")
        text = update.message.reply_text.call_args[0][0]
        assert "остановлена" in text

    async def test_stop_nonexistent_task_id(self) -> None:
        """Shows error for nonexistent task ID."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.get_task = AsyncMock(return_value=None)
        context = _make_context(
            bot_data={"task_manager": task_manager},
            args=["nonexist"],
        )

        await taskstop_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "не найдена" in text
        task_manager.stop_task.assert_not_called()

    async def test_stop_completed_task_shows_error(self) -> None:
        """Shows error when trying to stop a completed task."""
        update = _make_update()
        task = _make_task(task_id="abc12345", status="completed")
        task_manager = AsyncMock()
        task_manager.get_task = AsyncMock(return_value=task)
        context = _make_context(
            bot_data={"task_manager": task_manager},
            args=["abc12345"],
        )

        await taskstop_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "уже завершена" in text
        task_manager.stop_task.assert_not_called()

    async def test_no_args_no_running_tasks(self) -> None:
        """Shows message when no tasks to stop."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=None)
        task_manager.get_all_running = AsyncMock(return_value=[])
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
        )

        await taskstop_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Нет активных" in text

    async def test_no_args_single_task_auto_stop(self) -> None:
        """Auto-stops the only running task when no ID given."""
        update = _make_update()
        task = _make_task(task_id="solo1234")
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=None)
        task_manager.get_all_running = AsyncMock(return_value=[task])
        task_manager.stop_task = AsyncMock()
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/other")},
        )

        await taskstop_command(update, context)

        task_manager.stop_task.assert_called_once_with("solo1234")
        text = update.message.reply_text.call_args[0][0]
        assert "остановлена" in text

    async def test_no_args_multiple_tasks_shows_keyboard(self) -> None:
        """Shows inline keyboard to choose which task to stop."""
        update = _make_update()
        tasks = [
            _make_task(task_id="t1", project_path=Path("/projects/app1")),
            _make_task(task_id="t2", project_path=Path("/projects/app2")),
        ]
        task_manager = AsyncMock()
        task_manager.get_running_task = AsyncMock(return_value=None)
        task_manager.get_all_running = AsyncMock(return_value=tasks)
        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/other")},
        )

        await taskstop_command(update, context)

        call_kwargs = update.message.reply_text.call_args[1]
        assert "reply_markup" in call_kwargs
        assert call_kwargs["reply_markup"] is not None
        task_manager.stop_task.assert_not_called()


class TestTaskcontinueCommand:
    """Tests for /taskcontinue handler."""

    async def test_no_args_shows_usage(self) -> None:
        """Shows usage when called without arguments."""
        update = _make_update()
        context = _make_context(args=[])

        await taskcontinue_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Использование" in text

    async def test_no_task_manager(self) -> None:
        """Shows error when task_manager is not configured."""
        update = _make_update()
        context = _make_context(args=["continue", "this"])

        await taskcontinue_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "не настроены" in text

    async def test_no_project_path(self) -> None:
        """Shows error when project path cannot be determined."""
        update = _make_update()
        task_manager = AsyncMock()
        context = _make_context(
            bot_data={"task_manager": task_manager},
            args=["continue"],
        )

        await taskcontinue_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "определить проект" in text

    async def test_continue_with_session(self) -> None:
        """Continues a task with previous session context."""
        update = _make_update()
        last_task = _make_task(
            task_id="prev1234",
            status="completed",
            session_id="sess-old",
        )
        task_manager = AsyncMock()
        task_manager.get_task_for_continue = AsyncMock(return_value=last_task)
        task_manager.start_task = AsyncMock(return_value="new12345")

        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
            args=["fix", "tests"],
        )

        await taskcontinue_command(update, context)

        call_kwargs = task_manager.start_task.call_args[1]
        assert call_kwargs["session_id"] == "sess-old"
        assert call_kwargs["prompt"] == "fix tests"

        text = update.message.reply_text.call_args[0][0]
        assert "с контекстом предыдущей" in text
        assert "new12345" in text

    async def test_continue_without_previous_session(self) -> None:
        """Starts a fresh task when no previous session exists."""
        update = _make_update()
        task_manager = AsyncMock()
        task_manager.get_task_for_continue = AsyncMock(return_value=None)
        task_manager.start_task = AsyncMock(return_value="fresh123")

        context = _make_context(
            bot_data={"task_manager": task_manager},
            user_data={"current_directory": Path("/projects/myapp")},
            args=["new", "feature"],
        )

        await taskcontinue_command(update, context)

        call_kwargs = task_manager.start_task.call_args[1]
        assert call_kwargs["session_id"] is None

        text = update.message.reply_text.call_args[0][0]
        assert "с контекстом предыдущей" not in text


class TestGetProjectPath:
    """Tests for _get_project_path helper."""

    def test_from_user_data_path(self) -> None:
        """Returns Path from user_data current_directory."""
        context = _make_context(
            user_data={"current_directory": Path("/projects/myapp")}
        )
        result = _get_project_path(context)
        assert result == Path("/projects/myapp")

    def test_from_user_data_string(self) -> None:
        """Converts string current_directory to Path."""
        context = _make_context(
            user_data={"current_directory": "/projects/myapp"}
        )
        result = _get_project_path(context)
        assert result == Path("/projects/myapp")

    def test_from_settings_fallback(self) -> None:
        """Falls back to settings.approved_directory."""
        settings = MagicMock()
        settings.approved_directory = Path("/home/projects")
        context = _make_context(bot_data={"settings": settings})
        result = _get_project_path(context)
        assert result == Path("/home/projects")

    def test_returns_none_when_unavailable(self) -> None:
        """Returns None when no directory info is available."""
        context = _make_context()
        result = _get_project_path(context)
        assert result is None
