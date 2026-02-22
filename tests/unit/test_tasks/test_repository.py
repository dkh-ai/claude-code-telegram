"""Tests for TaskRepository."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.storage.database import DatabaseManager
from src.tasks.models import BackgroundTask
from src.tasks.repository import TaskRepository


@pytest.fixture
async def db_manager():
    """Create test database manager with migrations applied."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        manager = DatabaseManager(f"sqlite:///{db_path}")
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.fixture
async def task_repo(db_manager):
    """Create task repository."""
    # background_tasks has a FK to users, so seed a user first
    async with db_manager.get_connection() as conn:
        await conn.execute(
            "INSERT INTO users (user_id, telegram_username, is_allowed) VALUES (?, ?, ?)",
            (42, "testuser", True),
        )
        await conn.commit()
    return TaskRepository(db_manager)


def _make_task(**overrides) -> BackgroundTask:
    """Helper to build a BackgroundTask with sensible defaults."""
    defaults = dict(
        task_id="task-001",
        user_id=42,
        project_path=Path("/projects/myapp"),
        prompt="Fix the bug",
        status="running",
        provider="anthropic",
        created_at=datetime.now(UTC),
        last_activity_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return BackgroundTask(**defaults)


class TestTaskRepository:
    """Tests for TaskRepository CRUD operations."""

    async def test_create_and_get(self, task_repo):
        """Create a task and retrieve it by ID."""
        task = _make_task()
        await task_repo.create(task)

        retrieved = await task_repo.get("task-001")
        assert retrieved is not None
        assert retrieved.task_id == "task-001"
        assert retrieved.user_id == 42
        assert str(retrieved.project_path) == "/projects/myapp"
        assert retrieved.prompt == "Fix the bug"
        assert retrieved.status == "running"
        assert retrieved.total_cost == 0.0
        assert retrieved.commits == []

    async def test_get_nonexistent_returns_none(self, task_repo):
        """Getting a task that does not exist returns None."""
        result = await task_repo.get("does-not-exist")
        assert result is None

    async def test_update_status_completed(self, task_repo):
        """update_status sets status, finished_at, and optional fields."""
        task = _make_task()
        await task_repo.create(task)

        await task_repo.update_status(
            "task-001",
            "completed",
            result_summary="All tests pass",
            commits=[{"sha": "abc123", "message": "Fix bug"}],
        )

        updated = await task_repo.get("task-001")
        assert updated is not None
        assert updated.status == "completed"
        assert updated.finished_at is not None
        assert updated.result_summary == "All tests pass"
        assert len(updated.commits) == 1
        assert updated.commits[0]["sha"] == "abc123"

    async def test_update_status_failed(self, task_repo):
        """update_status with 'failed' sets error_message and finished_at."""
        task = _make_task()
        await task_repo.create(task)

        await task_repo.update_status(
            "task-001",
            "failed",
            error_message="Timeout after 10 minutes",
        )

        updated = await task_repo.get("task-001")
        assert updated is not None
        assert updated.status == "failed"
        assert updated.finished_at is not None
        assert updated.error_message == "Timeout after 10 minutes"

    async def test_update_status_with_session_id(self, task_repo):
        """update_status can set session_id."""
        task = _make_task()
        await task_repo.create(task)

        await task_repo.update_status(
            "task-001",
            "running",
            session_id="sess-abc",
        )

        updated = await task_repo.get("task-001")
        assert updated is not None
        assert updated.session_id == "sess-abc"
        # Non-terminal status should not set finished_at
        assert updated.finished_at is None

    async def test_update_progress(self, task_repo):
        """update_progress accumulates cost and increments turns."""
        task = _make_task()
        await task_repo.create(task)

        await task_repo.update_progress("task-001", cost=0.05, last_output="Step 1 done")
        await task_repo.update_progress("task-001", cost=0.10, last_output="Step 2 done")

        updated = await task_repo.get("task-001")
        assert updated is not None
        assert updated.total_cost == pytest.approx(0.15)
        assert updated.total_turns == 2
        assert updated.last_output == "Step 2 done"

    async def test_get_running_for_project(self, task_repo):
        """get_running_for_project returns the running task for a path."""
        running = _make_task(task_id="run-1", status="running")
        completed = _make_task(task_id="done-1", status="completed")
        await task_repo.create(running)
        await task_repo.create(completed)

        result = await task_repo.get_running_for_project(Path("/projects/myapp"))
        assert result is not None
        assert result.task_id == "run-1"

    async def test_get_running_for_project_none(self, task_repo):
        """get_running_for_project returns None when no running tasks exist."""
        completed = _make_task(task_id="done-1", status="completed")
        await task_repo.create(completed)

        result = await task_repo.get_running_for_project(Path("/projects/myapp"))
        assert result is None

    async def test_get_all_running(self, task_repo):
        """get_all_running returns all tasks with status 'running'."""
        await task_repo.create(_make_task(task_id="r1", status="running"))
        await task_repo.create(
            _make_task(
                task_id="r2",
                status="running",
                project_path=Path("/projects/other"),
            )
        )
        await task_repo.create(_make_task(task_id="c1", status="completed"))

        running = await task_repo.get_all_running()
        assert len(running) == 2
        running_ids = {t.task_id for t in running}
        assert running_ids == {"r1", "r2"}

    async def test_count_running(self, task_repo):
        """count_running returns the number of running tasks."""
        assert await task_repo.count_running() == 0

        await task_repo.create(_make_task(task_id="r1", status="running"))
        assert await task_repo.count_running() == 1

        await task_repo.create(
            _make_task(
                task_id="r2",
                status="running",
                project_path=Path("/projects/other"),
            )
        )
        assert await task_repo.count_running() == 2

        await task_repo.create(_make_task(task_id="c1", status="completed"))
        assert await task_repo.count_running() == 2

    async def test_get_last_completed(self, task_repo):
        """get_last_completed returns the most recently finished task."""
        # Create two completed tasks
        task1 = _make_task(task_id="t1", status="running")
        await task_repo.create(task1)
        await task_repo.update_status("t1", "completed", result_summary="First")

        task2 = _make_task(task_id="t2", status="running")
        await task_repo.create(task2)
        await task_repo.update_status("t2", "failed", error_message="Broke")

        last = await task_repo.get_last_completed(Path("/projects/myapp"))
        assert last is not None
        # t2 was finished after t1 so it should be returned
        assert last.task_id == "t2"
        assert last.status == "failed"

    async def test_get_last_completed_none(self, task_repo):
        """get_last_completed returns None when no completed tasks exist."""
        await task_repo.create(_make_task(task_id="r1", status="running"))

        result = await task_repo.get_last_completed(Path("/projects/myapp"))
        assert result is None

    async def test_get_last_completed_different_project(self, task_repo):
        """get_last_completed only considers tasks for the given project."""
        other = _make_task(
            task_id="o1",
            status="running",
            project_path=Path("/projects/other"),
        )
        await task_repo.create(other)
        await task_repo.update_status("o1", "completed", result_summary="Other done")

        result = await task_repo.get_last_completed(Path("/projects/myapp"))
        assert result is None
