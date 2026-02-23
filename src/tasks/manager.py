"""TaskManager -- background task lifecycle management.

Orchestrates the full lifecycle of background tasks: creation, execution,
progress tracking, cost enforcement, retry logic, and cleanup. Each task
runs an LLM provider call in a fire-and-forget asyncio task, with heartbeat
monitoring and event-driven notifications.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.events.bus import EventBus
from src.events.types import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from src.llm.interface import LLMProvider

from .heartbeat import HeartbeatService
from .models import BackgroundTask
from .repository import TaskRepository

logger = structlog.get_logger()

RETRY_DELAY_SECONDS = 30


class CostLimitExceeded(Exception):
    """Raised when a task exceeds its configured cost limit."""

    def __init__(self, task_id: str, cost: float, limit: float) -> None:
        self.task_id = task_id
        self.cost = cost
        self.limit = limit
        super().__init__(
            f"Task {task_id} exceeded cost limit: ${cost:.2f} > ${limit:.2f}"
        )


class TaskManager:
    """Manages the full lifecycle of background tasks.

    Responsibilities:
    - Enforce concurrency limits (per-project and global)
    - Launch tasks as asyncio background coroutines
    - Track cost accumulation and enforce cost limits
    - Coordinate heartbeat monitoring
    - Publish lifecycle events (started, completed, failed)
    - Handle retry on transient failures
    - Recover orphaned tasks on restart
    """

    def __init__(
        self,
        provider: LLMProvider,
        repo: TaskRepository,
        event_bus: EventBus,
        heartbeat: HeartbeatService,
        settings: Any,
    ) -> None:
        self._provider = provider
        self._repo = repo
        self._event_bus = event_bus
        self._heartbeat = heartbeat
        self._settings = settings
        self._running_tasks: Dict[str, asyncio.Task[None]] = {}

    async def start_task(
        self,
        prompt: str,
        project_path: Path,
        user_id: int,
        chat_id: int,
        message_thread_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Launch a new background task.

        Args:
            prompt: The user prompt to execute.
            project_path: Working directory for the task.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID for notifications.
            message_thread_id: Optional forum topic thread ID.
            session_id: Optional session ID for conversation continuation.

        Returns:
            The 8-character hex task ID.

        Raises:
            ValueError: If the project already has a running task or
                the global concurrent task limit is reached.
        """
        # Check per-project exclusivity
        existing = await self._repo.get_running_for_project(project_path)
        if existing:
            raise ValueError(
                f"Project {project_path} already has a running task: "
                f"{existing.task_id}"
            )

        # Check global concurrency limit
        running_count = await self._repo.count_running()
        max_concurrent = self._settings.max_concurrent_tasks
        if running_count >= max_concurrent:
            raise ValueError(
                f"Maximum concurrent tasks reached ({max_concurrent})"
            )

        # Generate task ID and create DB record
        task_id = uuid.uuid4().hex[:8]
        task = BackgroundTask(
            task_id=task_id,
            user_id=user_id,
            project_path=project_path,
            prompt=prompt,
            status="running",
            session_id=session_id,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
        )
        await self._repo.create(task)

        logger.info(
            "Background task created",
            task_id=task_id,
            project_path=str(project_path),
            user_id=user_id,
        )

        # Publish start event
        await self._event_bus.publish(
            TaskStartedEvent(
                task_id=task_id,
                project_path=project_path,
                prompt=prompt,
                user_id=user_id,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
            )
        )

        # Launch the asyncio task and start heartbeat
        asyncio_task = asyncio.create_task(
            self._run_task_with_cleanup(task), name=f"bg-task-{task_id}"
        )
        self._running_tasks[task_id] = asyncio_task

        await self._heartbeat.start(task_id)

        return task_id

    async def stop_task(self, task_id: str) -> None:
        """Stop a running background task.

        Cancels the asyncio task, stops heartbeat, and marks as stopped in DB.
        """
        asyncio_task = self._running_tasks.pop(task_id, None)
        if asyncio_task and not asyncio_task.done():
            asyncio_task.cancel()
            try:
                await asyncio_task
            except asyncio.CancelledError:
                pass

        await self._heartbeat.stop(task_id)
        await self._repo.update_status(task_id, "stopped")

        logger.info("Background task stopped", task_id=task_id)

    async def has_running_task(self, project_path: Path) -> bool:
        """Check whether a project has a running task."""
        task = await self._repo.get_running_for_project(project_path)
        return task is not None

    async def get_running_task(self, project_path: Path) -> Optional[BackgroundTask]:
        """Get the running task for a project, if any."""
        return await self._repo.get_running_for_project(project_path)

    async def get_all_running(self) -> List[BackgroundTask]:
        """Get all currently running tasks."""
        return await self._repo.get_all_running()

    async def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a task by ID."""
        return await self._repo.get(task_id)

    async def get_task_for_continue(
        self, project_path: Path
    ) -> Optional[BackgroundTask]:
        """Get the last completed/failed task for /taskcontinue."""
        return await self._repo.get_last_completed(project_path)

    async def recover(self) -> None:
        """Recover from restart by marking orphaned running tasks as timed out.

        Called at startup to clean up tasks that were running when the bot
        was stopped or crashed.
        """
        orphaned = await self._repo.get_all_running()
        for task in orphaned:
            await self._repo.update_status(
                task.task_id,
                "failed",
                error_message="Бот был перезапущен, задача прервана",
            )
            logger.warning(
                "Recovered orphaned task",
                task_id=task.task_id,
                project_path=str(task.project_path),
            )

        if orphaned:
            logger.info(
                "Task recovery complete", recovered_count=len(orphaned)
            )

    async def _run_task(self, task: BackgroundTask) -> None:
        """Execute a background task with retry logic.

        Flow:
        1. Call the LLM provider with a stream callback for progress
        2. On success: collect git commits, mark completed, publish event
        3. On failure: retry once after delay (except cost limit errors)
        4. Always: stop heartbeat, remove from running tasks
        """
        task_id = task.task_id
        start_time = datetime.now(timezone.utc)
        accumulated_cost = 0.0
        cost_limit = self._settings.task_max_cost
        last_error: Optional[Exception] = None

        for attempt in range(2):  # max 2 attempts (initial + 1 retry)
            try:
                if attempt > 0:
                    logger.info(
                        "Retrying background task",
                        task_id=task_id,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                # Define the stream callback for progress tracking
                async def stream_callback(event: Any) -> None:
                    nonlocal accumulated_cost

                    # Extract cost delta from stream event
                    cost_delta = getattr(event, "cost", 0.0) or 0.0
                    accumulated_cost += cost_delta

                    # Extract output text
                    output = (
                        getattr(event, "content", None)
                        or getattr(event, "tool_name", None)
                        or None
                    )

                    # Update progress in DB
                    await self._repo.update_progress(
                        task_id, cost_delta, output
                    )

                    # Check cost limit
                    if accumulated_cost > cost_limit:
                        raise CostLimitExceeded(
                            task_id, accumulated_cost, cost_limit
                        )

                # Execute the LLM call (background tasks use configured model)
                bg_model = getattr(self._settings, "model_background", None)
                response = await self._provider.execute(
                    prompt=task.prompt,
                    working_dir=task.project_path,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    stream_callback=stream_callback,
                    model=bg_model,
                )

                if response.is_error:
                    raise RuntimeError(
                        response.error_message or "LLM execution failed"
                    )

                # Success: collect git commits
                commits = await self._collect_commits(
                    task.project_path, start_time
                )

                # Determine result summary
                result_summary = response.content[:500] if response.content else ""

                # Update DB with success
                await self._repo.update_status(
                    task_id,
                    "completed",
                    result_summary=result_summary,
                    session_id=response.session_id,
                    commits=commits,
                )

                # Publish completion event
                duration = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds()
                )
                await self._event_bus.publish(
                    TaskCompletedEvent(
                        task_id=task_id,
                        duration_seconds=duration,
                        cost=accumulated_cost + response.cost,
                        commits=commits,
                        result_summary=result_summary,
                        chat_id=task.chat_id or 0,
                        message_thread_id=task.message_thread_id,
                    )
                )

                logger.info(
                    "Background task completed",
                    task_id=task_id,
                    duration_seconds=duration,
                    cost=accumulated_cost + response.cost,
                    commits_count=len(commits),
                )
                return  # Success, exit the retry loop

            except asyncio.CancelledError:
                # Task was stopped, do not retry
                logger.info("Background task cancelled", task_id=task_id)
                raise

            except CostLimitExceeded as exc:
                # Cost limit hit, fail immediately without retry
                logger.warning(
                    "Background task exceeded cost limit",
                    task_id=task_id,
                    cost=exc.cost,
                    limit=exc.limit,
                )
                duration = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds()
                )
                await self._repo.update_status(
                    task_id,
                    "failed",
                    error_message=str(exc),
                )
                await self._event_bus.publish(
                    TaskFailedEvent(
                        task_id=task_id,
                        duration_seconds=duration,
                        cost=accumulated_cost,
                        error_message=str(exc),
                        last_output=task.last_output or "",
                        chat_id=task.chat_id or 0,
                        message_thread_id=task.message_thread_id,
                    )
                )
                return  # No retry for cost limit

            except Exception as exc:
                last_error = exc
                logger.error(
                    "Background task attempt failed",
                    task_id=task_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                # Continue to next attempt (or fall through if last)

        # All attempts exhausted — mark as failed
        if last_error is not None:
            duration = int(
                (datetime.now(timezone.utc) - start_time).total_seconds()
            )
            error_msg = str(last_error)
            await self._repo.update_status(
                task_id,
                "failed",
                error_message=error_msg,
            )
            await self._event_bus.publish(
                TaskFailedEvent(
                    task_id=task_id,
                    duration_seconds=duration,
                    cost=accumulated_cost,
                    error_message=error_msg,
                    last_output=task.last_output or "",
                    chat_id=task.chat_id or 0,
                    message_thread_id=task.message_thread_id,
                )
            )
            logger.error(
                "Background task failed after retries",
                task_id=task_id,
                error=error_msg,
            )

    async def _run_task_with_cleanup(self, task: BackgroundTask) -> None:
        """Wrapper ensuring heartbeat stop and running-tasks cleanup."""
        try:
            await self._run_task(task)
        except asyncio.CancelledError:
            pass  # Already logged in _run_task
        except Exception:
            logger.exception(
                "Unexpected error in background task", task_id=task.task_id
            )
        finally:
            await self._heartbeat.stop(task.task_id)
            self._running_tasks.pop(task.task_id, None)

    async def _collect_commits(
        self, project_path: Path, since: datetime
    ) -> List[Dict[str, str]]:
        """Collect git commits made since a given timestamp.

        Looks for commits containing '[claude]' in the message,
        which is the convention for Claude-authored commits.
        """
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "log",
                f"--since={since_iso}",
                "--grep=[claude]",
                "--oneline",
                cwd=str(project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0 or not stdout:
                return []

            commits: List[Dict[str, str]] = []
            for line in stdout.decode().strip().splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    commits.append({"sha": parts[0], "message": parts[1]})
            return commits

        except (OSError, FileNotFoundError):
            # git not available or not a git repository
            logger.debug(
                "Git not available for commit collection",
                project_path=str(project_path),
            )
            return []
