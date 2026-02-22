"""Repository for background task CRUD operations."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

from src.storage.database import DatabaseManager

from .models import BackgroundTask

logger = structlog.get_logger()


class TaskRepository:
    """Background task data access."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize repository."""
        self.db = db_manager

    async def create(self, task: BackgroundTask) -> None:
        """Insert a new background task."""
        async with self.db.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO background_tasks (
                    task_id, user_id, project_path, prompt, status,
                    session_id, provider, created_at, finished_at,
                    total_cost, total_turns, last_output, last_activity_at,
                    result_summary, error_message, commits_json,
                    chat_id, message_thread_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.user_id,
                    str(task.project_path),
                    task.prompt,
                    task.status,
                    task.session_id,
                    task.provider,
                    task.created_at,
                    task.finished_at,
                    task.total_cost,
                    task.total_turns,
                    task.last_output,
                    task.last_activity_at,
                    task.result_summary,
                    task.error_message,
                    json.dumps(task.commits),
                    task.chat_id,
                    task.message_thread_id,
                ),
            )
            await conn.commit()
            logger.info("Created background task", task_id=task.task_id)

    async def get(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a background task by ID."""
        async with self.db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM background_tasks WHERE task_id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
            return BackgroundTask.from_row(row) if row else None

    async def update_status(
        self,
        task_id: str,
        status: str,
        **kwargs: object,
    ) -> None:
        """Update task status and optional fields.

        Supported kwargs: result_summary, error_message, session_id, commits.
        Sets finished_at automatically for terminal statuses.
        """
        now = datetime.now(timezone.utc)
        sets = ["status = ?", "last_activity_at = ?"]
        params: list[object] = [status, now]

        # Terminal statuses get a finished_at timestamp
        if status in ("completed", "failed"):
            sets.append("finished_at = ?")
            params.append(now)

        if "result_summary" in kwargs:
            sets.append("result_summary = ?")
            params.append(kwargs["result_summary"])

        if "error_message" in kwargs:
            sets.append("error_message = ?")
            params.append(kwargs["error_message"])

        if "session_id" in kwargs:
            sets.append("session_id = ?")
            params.append(kwargs["session_id"])

        if "commits" in kwargs:
            sets.append("commits_json = ?")
            params.append(json.dumps(kwargs["commits"]))

        params.append(task_id)

        async with self.db.get_connection() as conn:
            await conn.execute(
                f"UPDATE background_tasks SET {', '.join(sets)} WHERE task_id = ?",
                params,
            )
            await conn.commit()

    async def update_progress(
        self,
        task_id: str,
        cost: float,
        last_output: Optional[str] = None,
    ) -> None:
        """Update task progress (cost accumulation and last output)."""
        now = datetime.now(timezone.utc)
        async with self.db.get_connection() as conn:
            await conn.execute(
                """
                UPDATE background_tasks
                SET total_cost = total_cost + ?,
                    total_turns = total_turns + 1,
                    last_output = ?,
                    last_activity_at = ?
                WHERE task_id = ?
                """,
                (cost, last_output, now, task_id),
            )
            await conn.commit()

    async def get_running_for_project(
        self, project_path: Path
    ) -> Optional[BackgroundTask]:
        """Get running task for a specific project, if any."""
        async with self.db.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM background_tasks
                WHERE project_path = ? AND status = 'running'
                LIMIT 1
                """,
                (str(project_path),),
            )
            row = await cursor.fetchone()
            return BackgroundTask.from_row(row) if row else None

    async def get_all_running(self) -> List[BackgroundTask]:
        """Get all currently running tasks."""
        async with self.db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM background_tasks WHERE status = 'running'"
            )
            rows = await cursor.fetchall()
            return [BackgroundTask.from_row(row) for row in rows]

    async def count_running(self) -> int:
        """Count currently running tasks."""
        async with self.db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE status = 'running'"
            )
            row = await cursor.fetchone()
            return row[0]

    async def get_last_completed(
        self, project_path: Path
    ) -> Optional[BackgroundTask]:
        """Get the most recently finished task for a project (completed or failed)."""
        async with self.db.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM background_tasks
                WHERE project_path = ? AND status IN ('completed', 'failed')
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (str(project_path),),
            )
            row = await cursor.fetchone()
            return BackgroundTask.from_row(row) if row else None
