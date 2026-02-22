"""Pydantic models for background tasks."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BackgroundTask(BaseModel):
    """Background task state."""

    task_id: str
    user_id: int
    project_path: Path
    prompt: str
    status: str = "running"
    session_id: Optional[str] = None
    provider: str = "anthropic"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    total_cost: float = 0.0
    total_turns: int = 0
    last_output: Optional[str] = None
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    commits: List[Dict[str, str]] = Field(default_factory=list)
    chat_id: Optional[int] = None
    message_thread_id: Optional[int] = None

    @classmethod
    def from_row(cls, row: Any) -> "BackgroundTask":
        """Create from database row."""
        data = dict(row)
        commits_raw = data.pop("commits_json", "[]")
        data["commits"] = json.loads(commits_raw) if commits_raw else []
        # Handle string datetimes (sqlite3 converters may already parse them)
        for field in ("created_at", "last_activity_at", "finished_at"):
            val = data.get(field)
            if isinstance(val, str) and val:
                data[field] = datetime.fromisoformat(val)
        return cls(**data)
