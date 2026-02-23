"""Memory data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class UserFact:
    """A persistent fact about a user."""

    user_id: int
    category: str  # preference, personal, work, location, contact, technical, custom
    fact: str
    source: Optional[str] = None
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "UserFact":
        """Create from database row."""
        return cls(
            user_id=row["user_id"],
            category=row["category"],
            fact=row["fact"],
            source=row.get("source"),
            confidence=row.get("confidence", 1.0),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class ConversationSummary:
    """Compressed summary of a past conversation session."""

    user_id: int
    summary: str
    key_topics: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class MemoryContext:
    """Aggregated memory context for prompt injection."""

    facts: list[UserFact] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    working_memory: list[dict[str, str]] = field(default_factory=list)
