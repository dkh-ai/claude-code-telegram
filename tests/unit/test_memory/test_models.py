"""Test memory data models — UserFact, ConversationSummary, MemoryContext."""

from datetime import datetime, timezone

import pytest

from src.memory.models import ConversationSummary, MemoryContext, UserFact


class TestUserFact:
    """Test UserFact dataclass."""

    def test_construction_with_required_fields(self) -> None:
        """Test UserFact can be created with only required fields."""
        fact = UserFact(user_id=123, category="preference", fact="likes Python")

        assert fact.user_id == 123
        assert fact.category == "preference"
        assert fact.fact == "likes Python"
        assert fact.source is None
        assert fact.confidence == 1.0
        assert fact.created_at is None
        assert fact.updated_at is None

    def test_construction_with_all_fields(self) -> None:
        """Test UserFact can be created with all fields specified."""
        created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        fact = UserFact(
            user_id=456,
            category="personal",
            fact="lives in San Francisco",
            source="manual",
            confidence=0.95,
            created_at=created,
            updated_at=updated,
        )

        assert fact.user_id == 456
        assert fact.category == "personal"
        assert fact.fact == "lives in San Francisco"
        assert fact.source == "manual"
        assert fact.confidence == 0.95
        assert fact.created_at == created
        assert fact.updated_at == updated

    def test_default_confidence_is_one(self) -> None:
        """Test that confidence defaults to 1.0."""
        fact = UserFact(user_id=1, category="work", fact="software engineer")
        assert fact.confidence == 1.0

    def test_from_row_with_full_data(self) -> None:
        """Test from_row() creates UserFact from complete database row."""
        created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        row = {
            "user_id": 789,
            "category": "technical",
            "fact": "uses VS Code",
            "source": "auto_extract",
            "confidence": 0.85,
            "created_at": created,
            "updated_at": updated,
        }

        fact = UserFact.from_row(row)

        assert fact.user_id == 789
        assert fact.category == "technical"
        assert fact.fact == "uses VS Code"
        assert fact.source == "auto_extract"
        assert fact.confidence == 0.85
        assert fact.created_at == created
        assert fact.updated_at == updated

    def test_from_row_with_minimal_data(self) -> None:
        """Test from_row() handles row with only required fields."""
        row = {
            "user_id": 100,
            "category": "location",
            "fact": "timezone EST",
        }

        fact = UserFact.from_row(row)

        assert fact.user_id == 100
        assert fact.category == "location"
        assert fact.fact == "timezone EST"
        assert fact.source is None
        assert fact.confidence == 1.0
        assert fact.created_at is None
        assert fact.updated_at is None

    def test_from_row_with_missing_optional_fields(self) -> None:
        """Test from_row() applies defaults when optional fields are missing."""
        row = {
            "user_id": 200,
            "category": "contact",
            "fact": "email test@example.com",
            "source": None,
            # confidence missing, should default to 1.0
        }

        fact = UserFact.from_row(row)

        assert fact.user_id == 200
        assert fact.source is None
        assert fact.confidence == 1.0


class TestConversationSummary:
    """Test ConversationSummary dataclass."""

    def test_construction_with_required_fields(self) -> None:
        """Test ConversationSummary can be created with only required fields."""
        summary = ConversationSummary(
            user_id=123,
            summary="User discussed Python programming and asked about asyncio.",
        )

        assert summary.user_id == 123
        assert summary.summary == "User discussed Python programming and asked about asyncio."
        assert summary.key_topics is None
        assert summary.session_id is None
        assert summary.created_at is None

    def test_construction_with_all_fields(self) -> None:
        """Test ConversationSummary can be created with all fields specified."""
        created = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)

        summary = ConversationSummary(
            user_id=456,
            summary="Discussed project architecture and database schema.",
            key_topics="architecture, database, postgresql",
            session_id="session-abc-123",
            created_at=created,
        )

        assert summary.user_id == 456
        assert summary.summary == "Discussed project architecture and database schema."
        assert summary.key_topics == "architecture, database, postgresql"
        assert summary.session_id == "session-abc-123"
        assert summary.created_at == created

    def test_optional_fields_default_to_none(self) -> None:
        """Test that all optional fields default to None."""
        summary = ConversationSummary(user_id=789, summary="Brief chat")

        assert summary.key_topics is None
        assert summary.session_id is None
        assert summary.created_at is None


class TestMemoryContext:
    """Test MemoryContext dataclass."""

    def test_construction_with_defaults(self) -> None:
        """Test MemoryContext initializes with empty lists by default."""
        context = MemoryContext()

        assert context.facts == []
        assert context.summaries == []
        assert context.working_memory == []

    def test_construction_with_facts(self) -> None:
        """Test MemoryContext can be created with facts."""
        fact1 = UserFact(user_id=1, category="preference", fact="likes Python")
        fact2 = UserFact(user_id=1, category="work", fact="software engineer")

        context = MemoryContext(facts=[fact1, fact2])

        assert len(context.facts) == 2
        assert context.facts[0].fact == "likes Python"
        assert context.facts[1].fact == "software engineer"
        assert context.summaries == []
        assert context.working_memory == []

    def test_construction_with_summaries(self) -> None:
        """Test MemoryContext can be created with summaries."""
        summaries = ["Summary 1", "Summary 2", "Summary 3"]

        context = MemoryContext(summaries=summaries)

        assert len(context.summaries) == 3
        assert context.summaries == summaries
        assert context.facts == []
        assert context.working_memory == []

    def test_construction_with_working_memory(self) -> None:
        """Test MemoryContext can be created with working memory."""
        working_memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        context = MemoryContext(working_memory=working_memory)

        assert len(context.working_memory) == 2
        assert context.working_memory[0]["role"] == "user"
        assert context.working_memory[1]["role"] == "assistant"
        assert context.facts == []
        assert context.summaries == []

    def test_construction_with_all_fields(self) -> None:
        """Test MemoryContext can be created with all fields populated."""
        fact = UserFact(user_id=1, category="personal", fact="name is Alice")
        summaries = ["Previous discussion about databases"]
        working_memory = [{"role": "user", "content": "Tell me about SQLite"}]

        context = MemoryContext(
            facts=[fact],
            summaries=summaries,
            working_memory=working_memory,
        )

        assert len(context.facts) == 1
        assert len(context.summaries) == 1
        assert len(context.working_memory) == 1

    def test_default_factory_creates_independent_instances(self) -> None:
        """Test that each MemoryContext instance gets independent lists."""
        context1 = MemoryContext()
        context2 = MemoryContext()

        context1.facts.append(UserFact(user_id=1, category="test", fact="test fact"))

        # context2 should still have empty lists
        assert len(context1.facts) == 1
        assert len(context2.facts) == 0
        assert context1.facts is not context2.facts
