"""Integration test: full background task lifecycle."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.bus import EventBus
from src.events.types import TaskCompletedEvent, TaskStartedEvent
from src.llm.interface import LLMResponse
from src.storage.database import DatabaseManager
from src.tasks.heartbeat import HeartbeatService
from src.tasks.manager import TaskManager
from src.tasks.repository import TaskRepository


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    await manager.initialize()
    # Insert a test user to satisfy foreign key constraints
    async with manager.get_connection() as conn:
        await conn.execute(
            "INSERT INTO users (user_id, telegram_username) VALUES (?, ?)",
            (42, "testuser"),
        )
        await conn.commit()
    yield manager
    await manager.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.execute = AsyncMock(
        return_value=LLMResponse(
            content="Done. Task completed.",
            session_id="sess-1",
            cost=0.5,
            duration_ms=1000,
            num_turns=3,
            is_error=False,
        )
    )
    return provider


@pytest.fixture
def settings():
    s = MagicMock()
    s.max_concurrent_tasks = 3
    s.task_max_cost = 10.0
    s.task_max_duration_seconds = 3600
    return s


@pytest.mark.asyncio
async def test_full_task_lifecycle(db, event_bus, mock_provider, settings, tmp_path):
    """Test: start task -> task completes -> events emitted -> DB updated."""
    received_events = []

    async def capture(event):
        received_events.append(event)

    event_bus.subscribe(TaskStartedEvent, capture)
    event_bus.subscribe(TaskCompletedEvent, capture)

    # Start the event bus so events are dispatched to handlers
    await event_bus.start()

    repo = TaskRepository(db)
    heartbeat = HeartbeatService(
        repo=repo, event_bus=event_bus, interval=60.0, timeout=300.0
    )

    manager = TaskManager(
        provider=mock_provider,
        repo=repo,
        event_bus=event_bus,
        heartbeat=heartbeat,
        settings=settings,
    )

    # Start the task
    project = tmp_path / "project"
    project.mkdir()
    task_id = await manager.start_task(
        prompt="Test task",
        project_path=project,
        user_id=42,
        chat_id=100,
    )

    assert len(task_id) == 8

    # Wait for task to complete
    for _ in range(30):
        await asyncio.sleep(0.1)
        task = await repo.get(task_id)
        if task and task.status != "running":
            break

    # Verify task completed
    task = await repo.get(task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.result_summary is not None

    # Give the event bus time to dispatch queued events
    await asyncio.sleep(0.3)

    # Verify events were emitted
    event_types = [type(e).__name__ for e in received_events]
    assert "TaskStartedEvent" in event_types
    assert "TaskCompletedEvent" in event_types

    # Verify provider was called
    mock_provider.execute.assert_called_once()

    await event_bus.stop()


@pytest.mark.asyncio
@patch("src.tasks.manager.RETRY_DELAY_SECONDS", 0.1)
async def test_task_error_handling(db, event_bus, settings, tmp_path):
    """Test: task fails -> marked as failed -> TaskFailedEvent emitted."""
    from src.events.types import TaskFailedEvent

    received_events = []

    async def capture(event):
        received_events.append(event)

    event_bus.subscribe(TaskFailedEvent, capture)

    failing_provider = AsyncMock()
    failing_provider.execute = AsyncMock(
        return_value=LLMResponse(
            content="",
            session_id=None,
            cost=0.1,
            duration_ms=500,
            num_turns=0,
            is_error=True,
            error_message="SDK timeout",
        )
    )

    repo = TaskRepository(db)
    heartbeat = HeartbeatService(
        repo=repo, event_bus=event_bus, interval=60.0, timeout=300.0
    )

    manager = TaskManager(
        provider=failing_provider,
        repo=repo,
        event_bus=event_bus,
        heartbeat=heartbeat,
        settings=settings,
    )

    project = tmp_path / "error_project"
    project.mkdir()
    task_id = await manager.start_task(
        prompt="Failing task",
        project_path=project,
        user_id=42,
        chat_id=100,
    )

    # Wait for task to fail (retry delay is patched to 0.1s)
    for _ in range(30):
        await asyncio.sleep(0.1)
        task = await repo.get(task_id)
        if task and task.status != "running":
            break

    task = await repo.get(task_id)
    assert task is not None
    assert task.status == "failed"
    assert task.error_message is not None


@pytest.mark.asyncio
async def test_task_recovery(db, event_bus, mock_provider, settings, tmp_path):
    """Test: orphaned tasks are recovered on startup."""
    repo = TaskRepository(db)

    # Manually insert a "running" task as if bot crashed
    from src.tasks.models import BackgroundTask

    orphan = BackgroundTask(
        task_id="orphaned1",
        user_id=42,
        project_path=tmp_path / "orphan_project",
        prompt="Orphaned task",
        status="running",
        chat_id=100,
    )
    await repo.create(orphan)

    heartbeat = HeartbeatService(
        repo=repo, event_bus=event_bus, interval=60.0, timeout=300.0
    )
    manager = TaskManager(
        provider=mock_provider,
        repo=repo,
        event_bus=event_bus,
        heartbeat=heartbeat,
        settings=settings,
    )

    # Recover
    await manager.recover()

    # Check orphan is marked as failed
    task = await repo.get("orphaned1")
    assert task is not None
    assert task.status == "failed"
    assert "перезапущен" in task.error_message.lower()


@pytest.mark.asyncio
async def test_concurrent_task_rejection(db, event_bus, mock_provider, settings, tmp_path):
    """Test: cannot start two tasks on the same project."""
    repo = TaskRepository(db)
    heartbeat = HeartbeatService(
        repo=repo, event_bus=event_bus, interval=60.0, timeout=300.0
    )

    # Make provider slow so task stays running
    async def slow_execute(**kwargs):
        await asyncio.sleep(10)
        return LLMResponse(
            content="Done",
            session_id="s1",
            cost=0.1,
            duration_ms=100,
            num_turns=1,
            is_error=False,
        )

    slow_provider = AsyncMock()
    slow_provider.execute = slow_execute

    manager = TaskManager(
        provider=slow_provider,
        repo=repo,
        event_bus=event_bus,
        heartbeat=heartbeat,
        settings=settings,
    )

    project = tmp_path / "busy_project"
    project.mkdir()

    # Start first task
    task_id = await manager.start_task(
        prompt="Task 1",
        project_path=project,
        user_id=42,
        chat_id=100,
    )

    # Try starting second task on same project
    with pytest.raises(ValueError, match="already has a running task"):
        await manager.start_task(
            prompt="Task 2",
            project_path=project,
            user_id=42,
            chat_id=100,
        )

    # Cleanup
    await manager.stop_task(task_id)
