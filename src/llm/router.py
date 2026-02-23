"""Hybrid regex + LLM intent router for tri-mode execution."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

from ..config.settings import Settings
from .chat_pool import ChatProviderPool

logger = structlog.get_logger()


@dataclass
class RoutingDecision:
    """Result of intent classification."""

    mode: str  # "agent" | "assistant" | "chat"
    model: str  # resolved model ID
    confidence: float  # 0.0-1.0
    method: str  # "regex" | "llm" | "override" | "context"


# Escalation chains: list of (mode, model) pairs
AGENT_CHAIN = [
    ("agent", "model_agent_default"),
    ("agent", "model_agent_heavy"),
]

CHAT_CHAIN = [
    ("chat", "model_chat_default"),
    ("chat", "model_chat_fallback"),
    ("agent", "model_agent_default"),
]


# --- Regex patterns ---

AGENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(edit|create|write|read|run|execute|fix|refactor|debug|deploy|build|test"
        r"|compile|install|commit|push|pull|merge|branch|git|npm|pip|poetry|make"
        r"|docker|bash|shell|terminal|script|file|directory|folder|code|implement"
        r"|delete|remove|move|copy|rename|setup|configure|migrate)\b",
        re.I,
    ),
    re.compile(
        r"\b(отредактируй|создай|напиши|прочитай|запусти|исправь|рефактор|дебаг"
        r"|собери|тест|скомпилируй|установи|коммит|пуш|пул|мерж|ветк[аиу]|файл"
        r"|директори[яюе]|папк[аиу]|код|реализуй|удали|перемести|скопируй"
        r"|настрой|мигрируй|деплой)\b",
        re.I,
    ),
    re.compile(r"\.(py|js|ts|go|rs|rb|java|cpp|c|sh|yaml|json|toml|md|sql)\b", re.I),
    re.compile(r"```", re.I),
]

ASSISTANT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(remind|reminder|schedule|напомни|напоминание|через \d+\s*(мин|час|сек)"
        r"|timer|alarm|будильник|поставь|запланируй)\b",
        re.I,
    ),
    re.compile(
        r"\b(knowledge base|kb|wiki|найди ресторан|забронируй|закажи"
        r"|рецепт|погода|weather|translate|переведи)\b",
        re.I,
    ),
]

CLASSIFIER_SYSTEM_PROMPT = """\
Classify the user message into one of three modes:
- "agent": needs file access, code editing, bash commands, git operations, testing, deployment
- "assistant": reminders, scheduling, restaurant search, knowledge base queries, personal tasks
- "chat": general questions, explanations, advice, conversation, translations

Respond with JSON only: {"mode": "...", "confidence": 0.0-1.0}"""


class IntentRouter:
    """Hybrid regex + LLM intent router."""

    def __init__(
        self,
        settings: Settings,
        chat_pool: Optional[ChatProviderPool] = None,
    ) -> None:
        self._settings = settings
        self._chat_pool = chat_pool
        self._intent_log: list[dict] = []

    async def route(
        self,
        message: str,
        user_context: Optional[dict] = None,
    ) -> RoutingDecision:
        """Route a message to the appropriate mode and model."""
        ctx = user_context or {}

        # 1. User override (/model command)
        override = ctx.get("model_override")
        if override:
            mode = self._mode_for_model(override)
            return RoutingDecision(
                mode=mode,
                model=override,
                confidence=1.0,
                method="override",
            )

        # 2. Active agent session → stay in agent mode
        if ctx.get("claude_session_id") and not ctx.get("force_new_session"):
            return RoutingDecision(
                mode="agent",
                model=self._settings.model_agent_default,
                confidence=0.8,
                method="context",
            )

        # 3. Auto-routing disabled → default to agent
        if not self._settings.auto_route_enabled:
            return RoutingDecision(
                mode="agent",
                model=self._settings.model_agent_default,
                confidence=1.0,
                method="override",
            )

        # 4. Regex match (instant, free)
        regex_result = self._match_regex(message)
        if regex_result:
            self._log_classification(message, regex_result)
            return regex_result

        # 5. LLM classification (fallback for ambiguous)
        llm_result = await self._classify_with_llm(message)
        if llm_result:
            self._log_classification(message, llm_result)
            return llm_result

        # 6. Default → chat
        default = RoutingDecision(
            mode="chat",
            model=self._settings.model_chat_default,
            confidence=0.5,
            method="regex",
        )
        self._log_classification(message, default)
        return default

    def _match_regex(self, message: str) -> Optional[RoutingDecision]:
        """Try regex patterns for instant classification."""
        for pattern in AGENT_PATTERNS:
            if pattern.search(message):
                return RoutingDecision(
                    mode="agent",
                    model=self._settings.model_agent_default,
                    confidence=0.85,
                    method="regex",
                )

        for pattern in ASSISTANT_PATTERNS:
            if pattern.search(message):
                return RoutingDecision(
                    mode="assistant",
                    model=self._settings.model_chat_default,
                    confidence=0.85,
                    method="regex",
                )

        return None

    async def _classify_with_llm(self, message: str) -> Optional[RoutingDecision]:
        """Use cheap LLM to classify intent."""
        if not self._chat_pool:
            return None

        provider = self._chat_pool.get_router_provider()
        if not provider:
            return None

        try:
            raw = await provider.classify(
                prompt=message,
                system=CLASSIFIER_SYSTEM_PROMPT,
            )
            data = json.loads(raw.strip())
            mode = data.get("mode", "chat")
            confidence = float(data.get("confidence", 0.5))

            if mode not in ("agent", "assistant", "chat"):
                mode = "chat"

            model = self._resolve_model(mode)
            return RoutingDecision(
                mode=mode,
                model=model,
                confidence=confidence,
                method="llm",
            )
        except Exception as exc:
            logger.warning("LLM classification failed", error=str(exc))
            return None

    def _resolve_model(self, mode: str) -> str:
        """Resolve default model for a mode."""
        if mode == "agent":
            return self._settings.model_agent_default
        elif mode == "assistant":
            return self._settings.model_chat_default
        else:
            return self._settings.model_chat_default

    def _mode_for_model(self, model: str) -> str:
        """Determine mode from model name."""
        if model.startswith("claude"):
            return "agent"
        return "chat"

    def escalate(self, current: RoutingDecision) -> Optional[RoutingDecision]:
        """Get next model in escalation chain."""
        chain = AGENT_CHAIN if current.mode == "agent" else CHAT_CHAIN

        found_current = False
        for mode, model_attr in chain:
            model = getattr(self._settings, model_attr, None)
            if not model:
                continue
            if found_current:
                return RoutingDecision(
                    mode=mode,
                    model=model,
                    confidence=current.confidence,
                    method="escalation",
                )
            if model == current.model:
                found_current = True

        # Cross-mode escalation: chat exhausted → agent
        if current.mode == "chat" and not found_current:
            return RoutingDecision(
                mode="agent",
                model=self._settings.model_agent_default,
                confidence=current.confidence,
                method="escalation",
            )

        return None

    def _log_classification(self, message: str, decision: RoutingDecision) -> None:
        """Log classification for analytics."""
        msg_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
        self._intent_log.append({
            "message_hash": msg_hash,
            "mode": decision.mode,
            "confidence": decision.confidence,
            "method": decision.method,
        })
        # Keep only last 1000 entries in memory
        if len(self._intent_log) > 1000:
            self._intent_log = self._intent_log[-500:]
