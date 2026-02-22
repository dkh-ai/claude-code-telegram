"""Tests for HeartbeatService."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.types import TaskProgressEvent, TaskTimeoutEvent
from src.tasks.heartbeat import HeartbeatService
from src.tasks.models import BackgroundTask


def _make_task(
    task_id: str = "task-001",
    status: str = "running",
    last_output: str | None = None,
    created_at: datetime | None = None,
    last_activity_at: datetime | None = None,
    chat_id: int = 100,
    message_thread_id: int | None = None,
    total_cost: float = 0.25,
) -> BackgroundTask:
    """Helper to build a BackgroundTask with sensible defaults."""
    now = datetime.now(UTC)
    return BackgroundTask(
        task_id=task_id,
        user_id=42,
        project_path=Path("/projects/myapp"),
        prompt="Fix the bug",
        status=status,
        created_at=created_at or now,
        last_activity_at=last_activity_at or now,
        last_output=last_output,
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        total_cost=total_cost,
    )


class TestParseStage:
    """Tests for HeartbeatService.parse_stage static method."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Read file src/main.py", "исследует код"),
            ("Glob pattern **/*.ts", "исследует код"),
            ("Grep for errors", "исследует код"),
            ("Searching for config", "исследует код"),
            ("Write new module", "пишет код"),
            ("Edit src/utils.py", "пишет код"),
            ("creating file tests/test_new.py", "пишет код"),
            ("pytest tests/ -v", "запускает тесты"),
            ("npm test", "запускает тесты"),
            ("jest --coverage", "запускает тесты"),
            ("make test", "запускает тесты"),
            ("git commit -m 'fix'", "коммитит"),
            ("git push origin main", "коммитит"),
            ("thinking about approach", "планирует"),
            ("planning the implementation", "планирует"),
            ("analyzing the codebase", "планирует"),
            ("pip install requests", "устанавливает зависимости"),
            ("npm install lodash", "устанавливает зависимости"),
            ("poetry add fastapi", "устанавливает зависимости"),
        ],
    )
    def test_stage_detection(self, text: str, expected: str) -> None:
        """parse_stage returns correct stage for each keyword pattern."""
        assert HeartbeatService.parse_stage(text) == expected

    def test_none_returns_default(self) -> None:
        """parse_stage returns default for None input."""
        assert HeartbeatService.parse_stage(None) == "работает"

    def test_empty_string_returns_default(self) -> None:
        """parse_stage returns default for empty string."""
        assert HeartbeatService.parse_stage("") == "работает"

    def test_unknown_text_returns_default(self) -> None:
        """parse_stage returns default for text matching no pattern."""
        assert HeartbeatService.parse_stage("doing something unusual") == "работает"


class TestStartStop:
    """Tests for start/stop/stop_all lifecycle methods."""

    @pytest.fixture
    def service(self) -> HeartbeatService:
        """Create a HeartbeatService with mock dependencies."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=None)
        event_bus = AsyncMock()
        return HeartbeatService(repo, event_bus, interval=0.05, timeout=0.2)

    async def test_start_creates_task(self, service: HeartbeatService) -> None:
        """start() creates an asyncio task in _tasks dict."""
        await service.start("task-001")
        assert "task-001" in service._tasks
        assert isinstance(service._tasks["task-001"], asyncio.Task)
        # Cleanup
        await service.stop("task-001")

    async def test_start_idempotent(self, service: HeartbeatService) -> None:
        """Calling start() twice with the same task_id does not create a duplicate."""
        await service.start("task-001")
        first_task = service._tasks["task-001"]
        await service.start("task-001")
        assert service._tasks["task-001"] is first_task
        await service.stop("task-001")

    async def test_stop_cancels_and_removes(self, service: HeartbeatService) -> None:
        """stop() cancels the asyncio task and removes it from _tasks."""
        await service.start("task-001")
        assert "task-001" in service._tasks
        await service.stop("task-001")
        assert "task-001" not in service._tasks

    async def test_stop_nonexistent_is_safe(self, service: HeartbeatService) -> None:
        """stop() on a task_id not in _tasks does not raise."""
        await service.stop("nonexistent")

    async def test_stop_all(self, service: HeartbeatService) -> None:
        """stop_all() stops all tracked heartbeat tasks."""
        await service.start("task-001")
        await service.start("task-002")
        assert len(service._tasks) == 2

        await service.stop_all()
        assert len(service._tasks) == 0


class TestHeartbeatLoop:
    """Tests for the heartbeat _loop behavior with mocked repo and event_bus."""

    async def test_emits_progress_event(self) -> None:
        """Loop emits TaskProgressEvent when task is running."""
        task = _make_task(
            created_at=datetime.now(UTC) - timedelta(seconds=30),
            last_output="Grep for patterns",
        )
        repo = AsyncMock()
        # First call returns running task, second returns completed to stop loop
        repo.get = AsyncMock(
            side_effect=[task, _make_task(status="completed")]
        )
        event_bus = AsyncMock()
        event_bus.publish = AsyncMock()

        service = HeartbeatService(repo, event_bus, interval=0.05, timeout=300.0)
        await service.start("task-001")

        # Wait for loop to process
        await asyncio.sleep(0.2)
        await service.stop("task-001")

        # Verify at least one progress event was published
        published_events = [
            call.args[0]
            for call in event_bus.publish.call_args_list
            if isinstance(call.args[0], TaskProgressEvent)
        ]
        assert len(published_events) >= 1
        event = published_events[0]
        assert event.task_id == "task-001"
        assert event.stage == "исследует код"
        assert event.cost == 0.25

    async def test_emits_timeout_event(self) -> None:
        """Loop emits TaskTimeoutEvent when task is idle beyond timeout."""
        long_ago = datetime.now(UTC) - timedelta(seconds=600)
        task = _make_task(
            created_at=long_ago,
            last_activity_at=long_ago,
            chat_id=200,
            message_thread_id=5,
        )
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=task)
        event_bus = AsyncMock()
        event_bus.publish = AsyncMock()

        service = HeartbeatService(repo, event_bus, interval=0.05, timeout=0.01)
        await service.start("task-001")

        await asyncio.sleep(0.2)
        # Loop should have broken after emitting timeout
        await service.stop("task-001")

        published_events = [
            call.args[0]
            for call in event_bus.publish.call_args_list
            if isinstance(call.args[0], TaskTimeoutEvent)
        ]
        assert len(published_events) >= 1
        event = published_events[0]
        assert event.task_id == "task-001"
        assert event.chat_id == 200
        assert event.message_thread_id == 5
        assert event.idle_seconds > 0

    async def test_loop_stops_when_task_not_running(self) -> None:
        """Loop exits when task status is no longer 'running'."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=_make_task(status="completed"))
        event_bus = AsyncMock()

        service = HeartbeatService(repo, event_bus, interval=0.05, timeout=300.0)
        await service.start("task-001")

        await asyncio.sleep(0.15)
        # Task should have been removed from _tasks after loop exited
        assert "task-001" not in service._tasks
        # No events should have been published
        event_bus.publish.assert_not_called()

    async def test_loop_stops_when_task_not_found(self) -> None:
        """Loop exits when repo.get returns None."""
        repo = AsyncMock()
        repo.get = AsyncMock(return_value=None)
        event_bus = AsyncMock()

        service = HeartbeatService(repo, event_bus, interval=0.05, timeout=300.0)
        await service.start("task-001")

        await asyncio.sleep(0.15)
        assert "task-001" not in service._tasks
        event_bus.publish.assert_not_called()

    async def test_loop_cleans_up_on_exception(self) -> None:
        """Loop removes itself from _tasks even on unexpected errors."""
        repo = AsyncMock()
        repo.get = AsyncMock(side_effect=RuntimeError("DB gone"))
        event_bus = AsyncMock()

        service = HeartbeatService(repo, event_bus, interval=0.05, timeout=300.0)
        await service.start("task-001")

        await asyncio.sleep(0.15)
        # Task should have been removed via finally block
        assert "task-001" not in service._tasks
