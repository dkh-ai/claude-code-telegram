"""Tests for TaskManager lifecycle management."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.types import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from src.llm.interface import LLMResponse
from src.tasks.manager import CostLimitExceeded, TaskManager
from src.tasks.models import BackgroundTask


def _make_settings(**overrides: object) -> MagicMock:
    """Create a mock settings object with sensible defaults."""
    settings = MagicMock()
    settings.max_concurrent_tasks = overrides.get("max_concurrent_tasks", 3)
    settings.task_max_cost = overrides.get("task_max_cost", 10.0)
    settings.task_max_duration_seconds = overrides.get(
        "task_max_duration_seconds", 3600
    )
    return settings


def _make_task(
    task_id: str = "abcd1234",
    status: str = "running",
    project_path: Path | None = None,
    **kwargs: object,
) -> BackgroundTask:
    """Helper to build a BackgroundTask with sensible defaults."""
    defaults: dict = dict(
        task_id=task_id,
        user_id=42,
        project_path=project_path or Path("/projects/myapp"),
        prompt="Fix the bug",
        status=status,
        chat_id=100,
        message_thread_id=None,
        created_at=datetime.now(UTC),
        last_activity_at=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return BackgroundTask(**defaults)


@pytest.fixture
def provider() -> AsyncMock:
    """Mock LLM provider."""
    mock = AsyncMock()
    mock.execute = AsyncMock(
        return_value=LLMResponse(
            content="Task completed successfully",
            session_id="sess-123",
            cost=0.50,
            duration_ms=5000,
            num_turns=3,
            is_error=False,
        )
    )
    return mock


@pytest.fixture
def repo() -> AsyncMock:
    """Mock task repository."""
    mock = AsyncMock()
    mock.create = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.update_status = AsyncMock()
    mock.update_progress = AsyncMock()
    mock.get_running_for_project = AsyncMock(return_value=None)
    mock.get_all_running = AsyncMock(return_value=[])
    mock.count_running = AsyncMock(return_value=0)
    mock.get_last_completed = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def event_bus() -> AsyncMock:
    """Mock event bus."""
    mock = AsyncMock()
    mock.publish = AsyncMock()
    return mock


@pytest.fixture
def heartbeat() -> AsyncMock:
    """Mock heartbeat service."""
    mock = AsyncMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.stop_all = AsyncMock()
    return mock


@pytest.fixture
def settings() -> MagicMock:
    """Mock settings."""
    return _make_settings()


@pytest.fixture
def manager(
    provider: AsyncMock,
    repo: AsyncMock,
    event_bus: AsyncMock,
    heartbeat: AsyncMock,
    settings: MagicMock,
) -> TaskManager:
    """Create TaskManager with all mocked dependencies."""
    return TaskManager(
        provider=provider,
        repo=repo,
        event_bus=event_bus,
        heartbeat=heartbeat,
        settings=settings,
    )


class TestStartTask:
    """Tests for TaskManager.start_task()."""

    async def test_start_task_returns_8char_id(
        self, manager: TaskManager
    ) -> None:
        """start_task returns an 8-character hex task ID."""
        task_id = await manager.start_task(
            prompt="Fix the bug",
            project_path=Path("/projects/myapp"),
            user_id=42,
            chat_id=100,
        )
        assert len(task_id) == 8
        # Verify it is valid hex
        int(task_id, 16)

    async def test_start_task_creates_db_record(
        self,
        manager: TaskManager,
        repo: AsyncMock,
    ) -> None:
        """start_task creates a BackgroundTask record in the repository."""
        task_id = await manager.start_task(
            prompt="Fix the bug",
            project_path=Path("/projects/myapp"),
            user_id=42,
            chat_id=100,
            message_thread_id=5,
            session_id="sess-old",
        )

        repo.create.assert_called_once()
        created_task = repo.create.call_args[0][0]
        assert isinstance(created_task, BackgroundTask)
        assert created_task.task_id == task_id
        assert created_task.user_id == 42
        assert created_task.prompt == "Fix the bug"
        assert created_task.status == "running"
        assert created_task.chat_id == 100
        assert created_task.message_thread_id == 5
        assert created_task.session_id == "sess-old"

    async def test_start_task_publishes_started_event(
        self,
        manager: TaskManager,
        event_bus: AsyncMock,
    ) -> None:
        """start_task publishes a TaskStartedEvent."""
        task_id = await manager.start_task(
            prompt="Fix the bug",
            project_path=Path("/projects/myapp"),
            user_id=42,
            chat_id=100,
        )

        event_bus.publish.assert_called_once()
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskStartedEvent)
        assert event.task_id == task_id
        assert event.project_path == Path("/projects/myapp")
        assert event.prompt == "Fix the bug"
        assert event.user_id == 42
        assert event.chat_id == 100

    async def test_start_task_starts_heartbeat(
        self,
        manager: TaskManager,
        heartbeat: AsyncMock,
    ) -> None:
        """start_task starts heartbeat monitoring for the task."""
        task_id = await manager.start_task(
            prompt="Fix the bug",
            project_path=Path("/projects/myapp"),
            user_id=42,
            chat_id=100,
        )

        heartbeat.start.assert_called_once_with(task_id)

    async def test_start_task_stores_asyncio_task(
        self, manager: TaskManager
    ) -> None:
        """start_task stores the asyncio.Task in _running_tasks."""
        task_id = await manager.start_task(
            prompt="Fix the bug",
            project_path=Path("/projects/myapp"),
            user_id=42,
            chat_id=100,
        )

        assert task_id in manager._running_tasks
        assert isinstance(manager._running_tasks[task_id], asyncio.Task)

        # Cleanup: cancel the running task
        manager._running_tasks[task_id].cancel()
        try:
            await manager._running_tasks[task_id]
        except asyncio.CancelledError:
            pass

    async def test_start_task_rejects_busy_project(
        self,
        manager: TaskManager,
        repo: AsyncMock,
    ) -> None:
        """start_task raises ValueError if project already has a running task."""
        existing_task = _make_task(task_id="existing1")
        repo.get_running_for_project.return_value = existing_task

        with pytest.raises(ValueError, match="already has a running task"):
            await manager.start_task(
                prompt="Another task",
                project_path=Path("/projects/myapp"),
                user_id=42,
                chat_id=100,
            )

        # No DB record should be created
        repo.create.assert_not_called()

    async def test_start_task_rejects_max_concurrent(
        self,
        manager: TaskManager,
        repo: AsyncMock,
    ) -> None:
        """start_task raises ValueError when max concurrent tasks reached."""
        repo.count_running.return_value = 3  # Equal to max_concurrent_tasks

        with pytest.raises(ValueError, match="Maximum concurrent tasks"):
            await manager.start_task(
                prompt="One more task",
                project_path=Path("/projects/myapp"),
                user_id=42,
                chat_id=100,
            )

        repo.create.assert_not_called()


class TestStopTask:
    """Tests for TaskManager.stop_task()."""

    async def test_stop_task_cancels_and_updates_db(
        self,
        manager: TaskManager,
        repo: AsyncMock,
        heartbeat: AsyncMock,
    ) -> None:
        """stop_task cancels the asyncio task, stops heartbeat, updates DB."""
        # Create a long-running coroutine to cancel
        async def slow_task() -> None:
            await asyncio.sleep(3600)

        task_id = "test1234"
        asyncio_task = asyncio.create_task(slow_task())
        manager._running_tasks[task_id] = asyncio_task

        await manager.stop_task(task_id)

        # Task should be removed from running
        assert task_id not in manager._running_tasks
        # Heartbeat should be stopped
        heartbeat.stop.assert_called_once_with(task_id)
        # DB should be updated to stopped
        repo.update_status.assert_called_once_with(task_id, "stopped")

    async def test_stop_task_nonexistent_is_safe(
        self,
        manager: TaskManager,
        repo: AsyncMock,
        heartbeat: AsyncMock,
    ) -> None:
        """stop_task on a non-existent task_id does not raise."""
        await manager.stop_task("nonexistent")

        heartbeat.stop.assert_called_once_with("nonexistent")
        repo.update_status.assert_called_once_with("nonexistent", "stopped")


class TestQueryMethods:
    """Tests for has_running_task, get_running_task, get_all_running, etc."""

    async def test_has_running_task_true(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """has_running_task returns True when project has a running task."""
        repo.get_running_for_project.return_value = _make_task()

        result = await manager.has_running_task(Path("/projects/myapp"))
        assert result is True

    async def test_has_running_task_false(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """has_running_task returns False when no running task exists."""
        repo.get_running_for_project.return_value = None

        result = await manager.has_running_task(Path("/projects/myapp"))
        assert result is False

    async def test_get_running_task(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """get_running_task delegates to repo."""
        expected = _make_task()
        repo.get_running_for_project.return_value = expected

        result = await manager.get_running_task(Path("/projects/myapp"))
        assert result is expected

    async def test_get_all_running(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """get_all_running delegates to repo.get_all_running."""
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]
        repo.get_all_running.return_value = tasks

        result = await manager.get_all_running()
        assert result == tasks
        repo.get_all_running.assert_called_once()

    async def test_get_task(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """get_task delegates to repo.get."""
        expected = _make_task()
        repo.get.return_value = expected

        result = await manager.get_task("abcd1234")
        assert result is expected
        repo.get.assert_called_once_with("abcd1234")

    async def test_get_task_for_continue(
        self, manager: TaskManager, repo: AsyncMock
    ) -> None:
        """get_task_for_continue delegates to repo.get_last_completed."""
        completed = _make_task(status="completed")
        repo.get_last_completed.return_value = completed

        result = await manager.get_task_for_continue(Path("/projects/myapp"))
        assert result is completed
        repo.get_last_completed.assert_called_once_with(Path("/projects/myapp"))


class TestRecover:
    """Tests for TaskManager.recover()."""

    async def test_recover_marks_orphaned_tasks_as_failed(
        self,
        manager: TaskManager,
        repo: AsyncMock,
    ) -> None:
        """recover marks all running tasks as failed with restart message."""
        orphaned = [
            _make_task(task_id="t1"),
            _make_task(task_id="t2", project_path=Path("/projects/other")),
        ]
        repo.get_all_running.return_value = orphaned

        await manager.recover()

        assert repo.update_status.call_count == 2

        # Verify both tasks were marked as failed
        calls = repo.update_status.call_args_list
        for call in calls:
            assert call[0][1] == "failed"
            assert "перезапущен" in call[1]["error_message"]

        # Verify task IDs
        recovered_ids = {call[0][0] for call in calls}
        assert recovered_ids == {"t1", "t2"}

    async def test_recover_no_orphaned_tasks(
        self,
        manager: TaskManager,
        repo: AsyncMock,
    ) -> None:
        """recover does nothing when no orphaned tasks exist."""
        repo.get_all_running.return_value = []

        await manager.recover()

        repo.update_status.assert_not_called()


class TestRunTask:
    """Tests for the internal _run_task execution flow."""

    async def test_successful_execution_publishes_completed(
        self,
        manager: TaskManager,
        provider: AsyncMock,
        repo: AsyncMock,
        event_bus: AsyncMock,
        heartbeat: AsyncMock,
    ) -> None:
        """Successful _run_task marks task completed and publishes event."""
        task = _make_task()

        with patch.object(
            manager, "_collect_commits", return_value=[]
        ) as mock_commits:
            await manager._run_task(task)

        # DB should be updated to completed
        repo.update_status.assert_called_once()
        call_args = repo.update_status.call_args
        assert call_args[0][0] == task.task_id
        assert call_args[0][1] == "completed"
        assert "result_summary" in call_args[1]
        assert call_args[1]["session_id"] == "sess-123"

        # Completed event should be published
        event_bus.publish.assert_called_once()
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskCompletedEvent)
        assert event.task_id == task.task_id

    async def test_provider_error_triggers_retry_then_fail(
        self,
        manager: TaskManager,
        provider: AsyncMock,
        repo: AsyncMock,
        event_bus: AsyncMock,
    ) -> None:
        """When provider raises, _run_task retries once then fails."""
        provider.execute.side_effect = RuntimeError("Connection lost")
        task = _make_task()

        # Patch RETRY_DELAY_SECONDS to avoid waiting
        with patch("src.tasks.manager.RETRY_DELAY_SECONDS", 0):
            await manager._run_task(task)

        # Provider should have been called twice (initial + retry)
        assert provider.execute.call_count == 2

        # DB should be updated to failed
        repo.update_status.assert_called_once()
        call_args = repo.update_status.call_args
        assert call_args[0][1] == "failed"
        assert "Connection lost" in call_args[1]["error_message"]

        # Failed event should be published
        event_bus.publish.assert_called_once()
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskFailedEvent)
        assert "Connection lost" in event.error_message

    async def test_llm_error_response_triggers_retry_then_fail(
        self,
        manager: TaskManager,
        provider: AsyncMock,
        repo: AsyncMock,
        event_bus: AsyncMock,
    ) -> None:
        """When LLM returns is_error=True, _run_task retries once then fails."""
        provider.execute.return_value = LLMResponse(
            content="",
            session_id=None,
            cost=0.0,
            duration_ms=0,
            num_turns=0,
            is_error=True,
            error_message="Model overloaded",
        )
        task = _make_task()

        with patch("src.tasks.manager.RETRY_DELAY_SECONDS", 0):
            await manager._run_task(task)

        assert provider.execute.call_count == 2

        repo.update_status.assert_called_once()
        assert repo.update_status.call_args[0][1] == "failed"

        event_bus.publish.assert_called_once()
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskFailedEvent)

    async def test_cost_limit_exceeded_no_retry(
        self,
        manager: TaskManager,
        provider: AsyncMock,
        repo: AsyncMock,
        event_bus: AsyncMock,
    ) -> None:
        """CostLimitExceeded fails immediately without retry."""
        # Make the provider raise CostLimitExceeded through stream callback
        async def execute_with_cost_overrun(
            prompt: str, working_dir: Path, user_id: int, **kwargs: object
        ) -> LLMResponse:
            callback = kwargs.get("stream_callback")
            if callback:
                # Simulate a stream event with excessive cost
                event = MagicMock()
                event.cost = 15.0  # exceeds limit of 10.0
                event.content = "Working..."
                event.tool_name = None
                await callback(event)
            # Should not reach here due to CostLimitExceeded
            return LLMResponse(
                content="done",
                session_id=None,
                cost=15.0,
                duration_ms=1000,
                num_turns=1,
                is_error=False,
            )

        provider.execute = AsyncMock(side_effect=execute_with_cost_overrun)
        task = _make_task()

        await manager._run_task(task)

        # Provider should only be called once (no retry for cost limit)
        assert provider.execute.call_count == 1

        # DB should be updated to failed
        repo.update_status.assert_called_once()
        assert repo.update_status.call_args[0][1] == "failed"
        assert "cost limit" in repo.update_status.call_args[1]["error_message"].lower()

        # Failed event published
        event_bus.publish.assert_called_once()
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskFailedEvent)

    async def test_cancelled_error_not_retried(
        self,
        manager: TaskManager,
        provider: AsyncMock,
    ) -> None:
        """CancelledError is re-raised without retry."""
        provider.execute.side_effect = asyncio.CancelledError()
        task = _make_task()

        with pytest.raises(asyncio.CancelledError):
            await manager._run_task(task)

        # Only one call, no retry
        assert provider.execute.call_count == 1

    async def test_successful_execution_collects_commits(
        self,
        manager: TaskManager,
        provider: AsyncMock,
        repo: AsyncMock,
        event_bus: AsyncMock,
    ) -> None:
        """Successful _run_task calls _collect_commits and includes in result."""
        task = _make_task()
        commits = [
            {"sha": "abc1234", "message": "[claude] Fix bug"},
            {"sha": "def5678", "message": "[claude] Add tests"},
        ]

        with patch.object(
            manager, "_collect_commits", return_value=commits
        ):
            await manager._run_task(task)

        # Commits should be passed to update_status
        call_args = repo.update_status.call_args
        assert call_args[1]["commits"] == commits

        # Commits should be in the completed event
        event = event_bus.publish.call_args[0][0]
        assert isinstance(event, TaskCompletedEvent)
        assert event.commits == commits


class TestCostLimitExceeded:
    """Tests for the CostLimitExceeded exception."""

    def test_exception_attributes(self) -> None:
        """CostLimitExceeded stores task_id, cost, and limit."""
        exc = CostLimitExceeded("abc123", 12.50, 10.0)
        assert exc.task_id == "abc123"
        assert exc.cost == 12.50
        assert exc.limit == 10.0

    def test_exception_message(self) -> None:
        """CostLimitExceeded has a descriptive message."""
        exc = CostLimitExceeded("abc123", 12.50, 10.0)
        assert "abc123" in str(exc)
        assert "$12.50" in str(exc)
        assert "$10.00" in str(exc)


class TestCollectCommits:
    """Tests for _collect_commits."""

    async def test_collect_commits_parses_output(
        self, manager: TaskManager
    ) -> None:
        """_collect_commits parses git log output into commit dicts."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"abc1234 [claude] Fix bug\ndef5678 [claude] Add tests\n",
            b"",
        )
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ):
            commits = await manager._collect_commits(
                Path("/projects/myapp"),
                datetime.now(UTC),
            )

        assert len(commits) == 2
        assert commits[0] == {"sha": "abc1234", "message": "[claude] Fix bug"}
        assert commits[1] == {"sha": "def5678", "message": "[claude] Add tests"}

    async def test_collect_commits_empty_on_error(
        self, manager: TaskManager
    ) -> None:
        """_collect_commits returns empty list on git error."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"fatal: not a git repository")
        mock_proc.returncode = 128

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ):
            commits = await manager._collect_commits(
                Path("/projects/myapp"),
                datetime.now(UTC),
            )

        assert commits == []

    async def test_collect_commits_handles_missing_git(
        self, manager: TaskManager
    ) -> None:
        """_collect_commits returns empty list if git is not available."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("git not found"),
        ):
            commits = await manager._collect_commits(
                Path("/projects/myapp"),
                datetime.now(UTC),
            )

        assert commits == []
