"""Concrete event types for the event bus."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bus import Event


@dataclass
class UserMessageEvent(Event):
    """A message from a Telegram user."""

    user_id: int = 0
    chat_id: int = 0
    text: str = ""
    working_directory: Path = field(default_factory=lambda: Path("."))
    source: str = "telegram"


@dataclass
class WebhookEvent(Event):
    """An external webhook delivery (GitHub, Notion, etc.)."""

    provider: str = ""
    event_type_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    delivery_id: str = ""
    source: str = "webhook"


@dataclass
class ScheduledEvent(Event):
    """A cron/scheduled trigger."""

    job_id: str = ""
    job_name: str = ""
    prompt: str = ""
    working_directory: Path = field(default_factory=lambda: Path("."))
    target_chat_ids: List[int] = field(default_factory=list)
    skill_name: Optional[str] = None
    source: str = "scheduler"


@dataclass
class AgentResponseEvent(Event):
    """An agent has produced a response to deliver."""

    chat_id: int = 0
    text: str = ""
    parse_mode: Optional[str] = "HTML"
    reply_to_message_id: Optional[int] = None
    source: str = "agent"
    originating_event_id: Optional[str] = None


# === Background Task Events ===


@dataclass
class TaskStartedEvent(Event):
    """Emitted when a background task is launched."""

    task_id: str = ""
    project_path: Path = field(default_factory=lambda: Path("."))
    prompt: str = ""
    user_id: int = 0
    chat_id: int = 0
    message_thread_id: Optional[int] = None
    source: str = "task_manager"


@dataclass
class TaskProgressEvent(Event):
    """Emitted periodically during background task execution (heartbeat)."""

    task_id: str = ""
    elapsed_seconds: int = 0
    cost: float = 0.0
    stage: str = ""
    chat_id: int = 0
    message_thread_id: Optional[int] = None
    source: str = "task_manager"


@dataclass
class TaskCompletedEvent(Event):
    """Emitted when a background task finishes successfully."""

    task_id: str = ""
    duration_seconds: int = 0
    cost: float = 0.0
    commits: List[Dict[str, str]] = field(default_factory=list)
    result_summary: str = ""
    chat_id: int = 0
    message_thread_id: Optional[int] = None
    source: str = "task_manager"


@dataclass
class TaskFailedEvent(Event):
    """Emitted when a background task fails."""

    task_id: str = ""
    duration_seconds: int = 0
    cost: float = 0.0
    error_message: str = ""
    last_output: str = ""
    chat_id: int = 0
    message_thread_id: Optional[int] = None
    source: str = "task_manager"


@dataclass
class TaskTimeoutEvent(Event):
    """Emitted when a background task appears hung."""

    task_id: str = ""
    duration_seconds: int = 0
    cost: float = 0.0
    idle_seconds: int = 0
    chat_id: int = 0
    message_thread_id: Optional[int] = None
    source: str = "task_manager"
