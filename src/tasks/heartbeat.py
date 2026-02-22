"""HeartbeatService -- periodic progress notifications for background tasks."""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Optional

from src.events.types import TaskProgressEvent, TaskTimeoutEvent

logger = logging.getLogger(__name__)

# Stage detection patterns from Claude Code output
STAGE_PATTERNS = [
    (re.compile(r"Read|Glob|Grep|searching", re.I), "исследует код"),
    (re.compile(r"Write|Edit|creating file", re.I), "пишет код"),
    (re.compile(r"pytest|npm test|jest|make test", re.I), "запускает тесты"),
    (re.compile(r"git commit|git push", re.I), "коммитит"),
    (re.compile(r"thinking|planning|analyzing", re.I), "планирует"),
    (re.compile(r"pip install|npm install|poetry", re.I), "устанавливает зависимости"),
]


class HeartbeatService:
    """Sends periodic progress updates for running background tasks."""

    def __init__(
        self,
        repo: object,
        event_bus: object,
        interval: float = 60.0,
        timeout: float = 300.0,
    ):
        self._repo = repo
        self._event_bus = event_bus
        self._interval = interval
        self._timeout = timeout
        self._tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

    async def start(self, task_id: str) -> None:
        """Start heartbeat loop for a task."""
        if task_id in self._tasks:
            return
        self._tasks[task_id] = asyncio.create_task(
            self._loop(task_id), name=f"heartbeat-{task_id}"
        )

    async def stop(self, task_id: str) -> None:
        """Stop heartbeat loop."""
        t = self._tasks.pop(task_id, None)
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    async def stop_all(self) -> None:
        """Stop all heartbeat loops."""
        for task_id in list(self._tasks.keys()):
            await self.stop(task_id)

    async def _loop(self, task_id: str) -> None:
        """Heartbeat loop: check task, emit progress or timeout."""
        try:
            while True:
                await asyncio.sleep(self._interval)
                task = await self._repo.get(task_id)
                if not task or task.status != "running":
                    break

                now = datetime.now(timezone.utc)
                elapsed = (now - task.created_at).total_seconds()

                # Check for hung task
                idle = (now - task.last_activity_at).total_seconds()
                if idle > self._timeout:
                    await self._event_bus.publish(
                        TaskTimeoutEvent(
                            task_id=task_id,
                            duration_seconds=int(elapsed),
                            cost=task.total_cost,
                            idle_seconds=int(idle),
                            chat_id=task.chat_id or 0,
                            message_thread_id=task.message_thread_id,
                        )
                    )
                    break

                # Emit progress
                stage = self.parse_stage(task.last_output)
                await self._event_bus.publish(
                    TaskProgressEvent(
                        task_id=task_id,
                        elapsed_seconds=int(elapsed),
                        cost=task.total_cost,
                        stage=stage,
                        chat_id=task.chat_id or 0,
                        message_thread_id=task.message_thread_id,
                    )
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Heartbeat error for task %s", task_id)
        finally:
            self._tasks.pop(task_id, None)

    @staticmethod
    def parse_stage(last_output: Optional[str]) -> str:
        """Determine current stage from Claude output keywords."""
        if not last_output:
            return "работает"
        for pattern, stage in STAGE_PATTERNS:
            if pattern.search(last_output):
                return stage
        return "работает"
