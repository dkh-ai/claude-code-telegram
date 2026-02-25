"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.exceptions import ClaudeToolValidationError
from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .chat_awareness import ChatAwareness
from .utils.html_format import escape_html

logger = structlog.get_logger()

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


# Model aliases for /model command
_MODEL_ALIASES: Dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250514",
    "opus": "claude-opus-4-6",
    "gpt": "gpt-4o-mini",
    "gpt4": "gpt-4o",
    "gpt4o": "gpt-4o",
    "ds": "deepseek-chat",
    "deepseek": "deepseek-chat",
    "auto": "auto",
}

# Max chat history messages in working memory
_CHAT_HISTORY_LIMIT = 20


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("verbose", self.agentic_verbose),
            ("repo", self.agentic_repo),
            ("model", self.agentic_model),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        # Background task commands (conditional on feature flag)
        features = self.deps.get("features")
        if features and features.background_tasks_enabled:
            from .handlers.task_handlers import (
                task_command,
                taskcontinue_command,
                tasklog_command,
                taskstatus_command,
                taskstop_command,
            )

            handlers.extend([
                ("task", task_command),
                ("taskstatus", taskstatus_command),
                ("tasklog", tasklog_command),
                ("taskstop", taskstop_command),
                ("taskcontinue", taskcontinue_command),
            ])

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        # Task callback handlers for inline keyboard buttons
        if features and features.background_tasks_enabled:
            app.add_handler(
                CallbackQueryHandler(
                    self._inject_deps(self._taskstop_callback),
                    pattern=r"^taskstop:",
                )
            )
            app.add_handler(
                CallbackQueryHandler(
                    self._inject_deps(self._tasklog_callback),
                    pattern=r"^tasklog:",
                )
            )
            app.add_handler(
                CallbackQueryHandler(
                    self._inject_deps(self._taskretry_callback),
                    pattern=r"^taskretry:",
                )
            )

        # Feedback and escalation callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._feedback_callback),
                pattern=r"^fb:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._escalate_callback),
                pattern=r"^escalate:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("repo", "List repos / switch workspace"),
                BotCommand("model", "Switch model (auto/haiku/sonnet/opus/gpt/deepseek)"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            features = self.deps.get("features")
            if features and features.background_tasks_enabled:
                commands.extend([
                    BotCommand("task", "Run a background task"),
                    BotCommand("taskstatus", "Show running tasks"),
                    BotCommand("tasklog", "Show task output"),
                    BotCommand("taskstop", "Stop a running task"),
                    BotCommand("taskcontinue", "Continue last task"),
                ])
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — I can read, write, and run code.\n\n"
            f"Working in: {dir_display}\n"
            f"Commands: /new (reset) · /status"
            f"{sync_line}",
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text("Session reset. What's next?")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        await update.message.reply_text(
            f"📂 {dir_display} · Session: {session_status}{cost_str}"
        )

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        Returns None when verbose_level is 0 (nothing to display).
        Typing indicators are handled by a separate heartbeat task.
        """
        if verbose_level == 0:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    tool_log.append({"kind": "tool", "name": name, "detail": detail})

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text and verbose_level >= 1:
                    # Collapse to first meaningful line, cap length
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        tool_log.append({"kind": "text", "detail": first_line[:120]})

            # Throttle progress message edits to avoid Telegram rate limits
            now = time.time()
            if (now - last_edit_time[0]) >= 2.0 and tool_log:
                last_edit_time[0] = now
                new_text = self._format_verbose_progress(
                    tool_log, verbose_level, start_time
                )
                try:
                    await progress_msg.edit_text(new_text)
                except Exception:
                    pass

        return _on_stream

    def _build_user_context(
        self, context: ContextTypes.DEFAULT_TYPE
    ) -> Dict[str, Any]:
        """Build user context dict for router."""
        return {
            "model_override": context.user_data.get("model_override"),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "force_new_session": context.user_data.get("force_new_session"),
        }

    async def _execute_chat(
        self,
        message: str,
        context: ContextTypes.DEFAULT_TYPE,
        model: str,
        user_id: int,
    ) -> Optional[str]:
        """Execute via ChatProvider (text-only, no tools)."""
        chat_pool = context.bot_data.get("chat_provider_pool")
        if not chat_pool:
            return None

        provider = chat_pool.get_for_model(model)
        if not provider:
            return None

        # Build messages with chat history + memory
        messages: List[Dict[str, str]] = []

        # System prompt with memory
        memory_manager = context.bot_data.get("memory_manager")
        system_parts = ["You are a helpful assistant. Respond in the same language as the user."]
        if memory_manager:
            memory_ctx = await memory_manager.recall(user_id, message)
            memory_text = memory_manager.format_for_prompt(memory_ctx)
            if memory_text:
                system_parts.append(memory_text)

        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # Chat history (working memory)
        history = context.user_data.get("chat_history", [])
        for entry in history[-_CHAT_HISTORY_LIMIT:]:
            messages.append(entry)

        messages.append({"role": "user", "content": message})

        response = await provider.chat(messages, model=model)

        # Update chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.content})
        # Trim history
        if len(history) > _CHAT_HISTORY_LIMIT * 2:
            history = history[-_CHAT_HISTORY_LIMIT * 2:]
        context.user_data["chat_history"] = history

        return response.content

    def _should_escalate(self, response_text: str, message: str) -> bool:
        """Check if response needs escalation to a more capable model."""
        if not response_text:
            return True
        # Short response to a long question
        if len(response_text) < 50 and len(message) > 100:
            return True
        # Response suggests needing file access
        escalation_phrases = [
            "I can't access files",
            "I cannot access",
            "не могу получить доступ",
            "я не имею доступа",
            "I don't have access to",
        ]
        for phrase in escalation_phrases:
            if phrase.lower() in response_text.lower():
                return True
        return False

    def _make_feedback_keyboard(
        self, msg_id: int, model: str, cost: float, duration_ms: int
    ) -> InlineKeyboardMarkup:
        """Create feedback + escalation inline keyboard."""
        duration_s = duration_ms / 1000
        footer = f"[{model} · ${cost:.4f} · {duration_s:.1f}s]"
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("\U0001f504", callback_data=f"escalate:{msg_id}"),
            InlineKeyboardButton("\U0001f44d", callback_data=f"fb:good:{msg_id}"),
            InlineKeyboardButton("\U0001f44e", callback_data=f"fb:bad:{msg_id}"),
        ]])

    async def _feedback_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle feedback button press."""
        query = update.callback_query
        await query.answer()

        parts = query.data.split(":")
        if len(parts) < 3:
            return

        feedback_type = parts[1]  # "good" or "bad"
        user_id = query.from_user.id

        # Store feedback
        storage = context.bot_data.get("storage")
        if storage:
            try:
                await storage.cost_tracking.record_feedback(user_id, feedback_type)
            except Exception:
                pass

        emoji = "\U0001f44d" if feedback_type == "good" else "\U0001f44e"
        await query.answer(f"{emoji} Спасибо за обратную связь!")

    async def _escalate_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle escalation (retry with better model) button press."""
        query = update.callback_query
        await query.answer("Escalating...")

        router = context.bot_data.get("model_router")
        if not router:
            await query.answer("Router not available.")
            return

        from ..llm.router import RoutingDecision

        current_model = context.user_data.get(
            "_last_model", self.settings.model_agent_default
        )
        current_mode = context.user_data.get("_last_mode", "chat")
        current_decision = RoutingDecision(
            mode=current_mode, model=current_model, confidence=0.5, method="escalation"
        )
        next_decision = router.escalate(current_decision)
        if not next_decision:
            await query.answer("No more models to escalate to.")
            return

        original_text = context.user_data.get("_last_user_message")
        if not original_text:
            await query.answer("Original message not found.")
            return

        user_id = query.from_user.id
        chat = query.message.chat

        # Show escalation progress
        progress_msg = await chat.send_message(
            f"\U0001f504 Escalating to <code>{next_decision.model}</code>...",
            parse_mode="HTML",
        )

        await chat.send_action("typing")

        response_text = None
        response_cost = 0.0
        response_duration_ms = 0

        try:
            if next_decision.mode == "chat":
                start_time = time.time()
                response_text = await self._execute_chat(
                    original_text, context, next_decision.model, user_id
                )
                response_duration_ms = int((time.time() - start_time) * 1000)
            else:
                # Agent mode escalation
                claude_integration = context.bot_data.get("claude_integration")
                if claude_integration:
                    current_dir = context.user_data.get(
                        "current_directory", self.settings.approved_directory
                    )
                    session_id = context.user_data.get("claude_session_id")
                    claude_response = await claude_integration.run_command(
                        prompt=original_text,
                        working_directory=current_dir,
                        user_id=user_id,
                        session_id=session_id,
                        model=next_decision.model
                        if next_decision.model != self.settings.model_agent_default
                        else None,
                    )
                    response_text = claude_response.content
                    response_cost = claude_response.cost
                    response_duration_ms = claude_response.duration_ms
                    context.user_data["claude_session_id"] = claude_response.session_id
        except Exception as e:
            logger.error("Escalation failed", error=str(e))
            try:
                await progress_msg.edit_text(
                    f"Escalation failed: {escape_html(str(e)[:200])}",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return

        try:
            await progress_msg.delete()
        except Exception:
            pass

        # Update tracking
        context.user_data["_last_model"] = next_decision.model
        context.user_data["_last_mode"] = next_decision.mode

        if response_text:
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted = formatter.format_claude_response(response_text)

            feedback_keyboard = self._make_feedback_keyboard(
                query.message.message_id,
                next_decision.model,
                response_cost,
                response_duration_ms,
            )

            for i, msg in enumerate(formatted):
                text = msg.text
                if not text or not text.strip():
                    continue
                if i == len(formatted) - 1:
                    duration_s = response_duration_ms / 1000
                    text += f"\n\n<i>[{next_decision.model} · ${response_cost:.4f} · {duration_s:.1f}s]</i>"

                try:
                    await chat.send_message(
                        text,
                        parse_mode=msg.parse_mode or "HTML",
                        reply_markup=feedback_keyboard if i == len(formatted) - 1 else None,
                    )
                except Exception as e:
                    logger.error("Failed to send escalated response", error=str(e))
        else:
            await chat.send_message("Escalation produced no response.")

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Tri-mode text handler: routes to agent, assistant, or chat mode."""
        user_id = update.effective_user.id
        message_text = update.message.text

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Multi-chat awareness
        awareness = ChatAwareness()
        chat_ctx = awareness.analyze(update, self.settings.telegram_bot_username)

        # Silent observation in groups (extract memory, don't respond)
        if awareness.should_observe(chat_ctx):
            memory_manager = context.bot_data.get("memory_manager")
            if memory_manager:
                asyncio.create_task(
                    memory_manager.extract_and_store(user_id, message_text, "")
                )
            return

        # Don't respond in groups unless mentioned/replied-to
        if not awareness.should_respond(chat_ctx):
            return

        # Block interactive mode if project has a running background task
        task_manager = context.bot_data.get("task_manager")
        if task_manager:
            current_dir = context.user_data.get(
                "current_directory", self.settings.approved_directory
            )
            running_task = await task_manager.get_running_task(
                Path(str(current_dir)) if not isinstance(current_dir, Path) else current_dir
            )
            if running_task:
                await update.message.reply_text(
                    f"⚠️ В проекте уже выполняется фоновая задача "
                    f"<code>{running_task.task_id}</code>.\n\n"
                    f"Используй /taskstatus для просмотра статуса, "
                    f"/taskstop для остановки, или переключись на другой проект "
                    f"через /repo.",
                    parse_mode="HTML",
                )
                return

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        # Recall memory
        memory_manager = context.bot_data.get("memory_manager")
        memory_ctx = None
        if memory_manager:
            memory_ctx = await memory_manager.recall(user_id, message_text)

        # Save for escalation callback
        context.user_data["_last_user_message"] = message_text

        # Route intent
        router = context.bot_data.get("model_router")
        user_ctx = self._build_user_context(context)

        if router:
            decision = await router.route(message_text, user_ctx)
        else:
            # Fallback: always agent mode
            from ..llm.router import RoutingDecision

            decision = RoutingDecision(
                mode="agent",
                model=self.settings.model_agent_default,
                confidence=1.0,
                method="override",
            )

        logger.info(
            "Routing decision",
            mode=decision.mode,
            model=decision.model,
            confidence=decision.confidence,
            method=decision.method,
        )

        # Store last model for escalation reference
        context.user_data["_last_model"] = decision.model
        context.user_data["_last_mode"] = decision.mode

        verbose_level = self._get_verbose_level(context)
        progress_msg = await update.message.reply_text("Working...")

        response_text: Optional[str] = None
        response_cost = 0.0
        response_duration_ms = 0
        response_model = decision.model
        success = True
        formatted_messages = []

        heartbeat = self._start_typing_heartbeat(chat)

        try:
            if decision.mode == "assistant":
                # Try assistant plugin dispatch
                dispatcher = context.bot_data.get("assistant_dispatcher")
                if dispatcher:
                    plugin_ctx = {
                        "user_id": user_id,
                        "chat_id": chat.id,
                        "memory": memory_ctx,
                    }
                    plugin_response = await dispatcher.dispatch(message_text, plugin_ctx)
                    if plugin_response:
                        response_text = plugin_response.content
                        response_model = plugin_response.model
                        response_cost = plugin_response.cost

                # Fallback to chat if plugin didn't handle
                if response_text is None:
                    decision.mode = "chat"

            if decision.mode == "chat":
                # Chat mode via ChatProvider
                start_time = time.time()
                response_text = await self._execute_chat(
                    message_text, context, decision.model, user_id
                )
                response_duration_ms = int((time.time() - start_time) * 1000)

                # Auto-escalate if response is weak
                if response_text and self._should_escalate(response_text, message_text):
                    if router:
                        next_decision = router.escalate(decision)
                        if next_decision:
                            logger.info(
                                "Auto-escalating",
                                from_model=decision.model,
                                to_model=next_decision.model,
                            )
                            decision = next_decision
                            response_model = next_decision.model
                            context.user_data["_last_model"] = next_decision.model
                            context.user_data["_last_mode"] = next_decision.mode
                            if next_decision.mode == "chat":
                                start_time = time.time()
                                response_text = await self._execute_chat(
                                    message_text, context, next_decision.model, user_id
                                )
                                response_duration_ms = int((time.time() - start_time) * 1000)
                            else:
                                # Escalated beyond chat — will be handled by agent block
                                response_text = None

                if response_text is None:
                    # No chat provider available or escalated to agent
                    decision.mode = "agent"
                    decision.model = self.settings.model_agent_default

            if decision.mode == "agent":
                # Agent mode via Claude integration
                claude_integration = context.bot_data.get("claude_integration")
                if not claude_integration:
                    await progress_msg.edit_text(
                        "Claude integration not available. Check configuration."
                    )
                    return

                current_dir = context.user_data.get(
                    "current_directory", self.settings.approved_directory
                )
                session_id = context.user_data.get("claude_session_id")
                force_new = bool(context.user_data.get("force_new_session"))

                tool_log: List[Dict[str, Any]] = []
                start_time = time.time()
                on_stream = self._make_stream_callback(
                    verbose_level, progress_msg, tool_log, start_time
                )

                # Inject memory into prompt if available
                prompt = message_text
                if memory_ctx and memory_manager:
                    memory_text = memory_manager.format_for_prompt(memory_ctx)
                    if memory_text:
                        prompt = f"[Context]\n{memory_text}\n\n[User message]\n{message_text}"

                claude_response = await claude_integration.run_command(
                    prompt=prompt,
                    working_directory=current_dir,
                    user_id=user_id,
                    session_id=session_id,
                    on_stream=on_stream,
                    force_new=force_new,
                    model=decision.model if decision.model != self.settings.model_agent_default else None,
                )

                if force_new:
                    context.user_data["force_new_session"] = False

                context.user_data["claude_session_id"] = claude_response.session_id
                response_text = claude_response.content
                response_cost = claude_response.cost
                response_duration_ms = claude_response.duration_ms
                response_model = decision.model

                # Track directory changes
                from .handlers.message import _update_working_directory_from_claude_response

                _update_working_directory_from_claude_response(
                    claude_response, context, self.settings, user_id
                )

                # Store interaction
                storage = context.bot_data.get("storage")
                if storage:
                    try:
                        await storage.save_claude_interaction(
                            user_id=user_id,
                            session_id=claude_response.session_id,
                            prompt=message_text,
                            response=claude_response,
                            ip_address=None,
                        )
                    except Exception as e:
                        logger.warning("Failed to log interaction", error=str(e))

            # Format response
            if response_text:
                from .utils.formatting import ResponseFormatter

                formatter = ResponseFormatter(self.settings)
                formatted_messages = formatter.format_claude_response(response_text)

        except ClaudeToolValidationError as e:
            success = False
            logger.error("Tool validation error", error=str(e), user_id=user_id)
            from .utils.formatting import FormattedMessage

            formatted_messages = [FormattedMessage(str(e), parse_mode="HTML")]

        except Exception as e:
            success = False
            logger.error("Execution failed", error=str(e), user_id=user_id, mode=decision.mode)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()

        await progress_msg.delete()

        # Build feedback keyboard for non-agent responses
        feedback_keyboard = None
        if success:
            msg_id = update.message.message_id
            feedback_keyboard = self._make_feedback_keyboard(
                msg_id, response_model, response_cost, response_duration_ms
            )

        # Send formatted response
        for i, message in enumerate(formatted_messages):
            if not message.text or not message.text.strip():
                continue

            # Add model badge to last message
            text = message.text
            if i == len(formatted_messages) - 1:
                duration_s = response_duration_ms / 1000
                text += f"\n\n<i>[{response_model} · ${response_cost:.4f} · {duration_s:.1f}s]</i>"

            reply_markup = feedback_keyboard if i == len(formatted_messages) - 1 else None

            try:
                await update.message.reply_text(
                    text,
                    parse_mode=message.parse_mode or "HTML",
                    reply_markup=reply_markup,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)
            except Exception as send_err:
                logger.warning(
                    "Failed to send HTML response, retrying as plain text",
                    error=str(send_err),
                    message_index=i,
                )
                try:
                    await update.message.reply_text(
                        message.text,
                        reply_markup=reply_markup,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                except Exception as plain_err:
                    await update.message.reply_text(
                        f"Failed to deliver response "
                        f"(Telegram error: {str(plain_err)[:150]}). "
                        f"Please try again.",
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )

        # Extract and store memory (async, non-blocking)
        if memory_manager and response_text and success:
            asyncio.create_task(
                memory_manager.extract_and_store(user_id, message_text, response_text)
            )

        # Cost tracking with model/mode
        storage = context.bot_data.get("storage")
        if storage and response_cost > 0:
            try:
                await storage.cost_tracking.update_daily_cost(
                    user_id=user_id,
                    cost=response_cost,
                    model=response_model,
                    mode=decision.mode,
                )
            except Exception:
                pass

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        # Try enhanced file handler, fall back to basic
        feature_registry = context.bot_data.get("feature_registry")
        file_handler = feature_registry.get_file_handler() if feature_registry else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        on_stream = self._make_stream_callback(
            verbose_level, progress_msg, tool_log, time.time()
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await progress_msg.delete()

            for i, message in enumerate(formatted_messages):
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        feature_registry = context.bot_data.get("feature_registry")
        image_handler = feature_registry.get_image_handler() if feature_registry else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )

            claude_integration = context.bot_data.get("claude_integration")
            if not claude_integration:
                await progress_msg.edit_text(
                    "Claude integration not available. Check configuration."
                )
                return

            current_dir = context.user_data.get(
                "current_directory", self.settings.approved_directory
            )
            session_id = context.user_data.get("claude_session_id")

            # Check if /new was used — skip auto-resume for this first message.
            # Flag is only cleared after a successful run so retries keep the intent.
            force_new = bool(context.user_data.get("force_new_session"))

            verbose_level = self._get_verbose_level(context)
            tool_log: List[Dict[str, Any]] = []
            on_stream = self._make_stream_callback(
                verbose_level, progress_msg, tool_log, time.time()
            )

            heartbeat = self._start_typing_heartbeat(chat)
            try:
                claude_response = await claude_integration.run_command(
                    prompt=processed_image.prompt,
                    working_directory=current_dir,
                    user_id=user_id,
                    session_id=session_id,
                    on_stream=on_stream,
                    force_new=force_new,
                )
            finally:
                heartbeat.cancel()

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await progress_msg.delete()

            for i, message in enumerate(formatted_messages):
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Switch model: /model [name|auto]."""
        args = update.message.text.split()[1:] if update.message.text else []

        if not args:
            # Show current configuration
            override = context.user_data.get("model_override")
            mode_str = "auto" if not override else override
            vendors = []
            if self.settings.deepseek_api_key:
                vendors.append("DeepSeek")
            if self.settings.openai_api_key:
                vendors.append("OpenAI")
            vendors.append("Claude")
            vendors_str = ", ".join(vendors)

            await update.message.reply_text(
                f"<b>Model routing</b>\n\n"
                f"Current: <code>{mode_str}</code>\n"
                f"Agent default: <code>{self.settings.model_agent_default}</code>\n"
                f"Chat default: <code>{self.settings.model_chat_default}</code>\n"
                f"Vendors: {vendors_str}\n\n"
                f"Usage: <code>/model haiku|sonnet|opus|gpt|deepseek|auto</code>",
                parse_mode="HTML",
            )
            return

        alias = args[0].lower()
        if alias == "auto":
            context.user_data.pop("model_override", None)
            await update.message.reply_text("Model: <b>auto</b> (routing enabled)", parse_mode="HTML")
            return

        model_id = _MODEL_ALIASES.get(alias, alias)
        context.user_data["model_override"] = model_id

        mode = "agent" if model_id.startswith("claude") else "chat"
        await update.message.reply_text(
            f"Model: <code>{model_id}</code> ({mode} mode)",
            parse_mode="HTML",
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _taskstop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle taskstop: callbacks from inline keyboard buttons."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, task_id = data.split(":", 1)

        task_manager = context.bot_data.get("task_manager")
        if not task_manager:
            await query.edit_message_text("Фоновые задачи не настроены.")
            return

        task = await task_manager.get_task(task_id)
        if not task or task.status != "running":
            await query.edit_message_text(
                f"Задача <code>{escape_html(task_id)}</code> не найдена "
                f"или уже завершена.",
                parse_mode="HTML",
            )
            return

        try:
            await task_manager.stop_task(task_id)
        except Exception as e:
            logger.error("Failed to stop task", task_id=task_id, error=str(e))
            await query.edit_message_text(
                f"Ошибка при остановке задачи: {escape_html(str(e)[:200])}",
                parse_mode="HTML",
            )
            return
        await query.edit_message_text(
            f"⏹ Задача <code>{escape_html(task_id)}</code> остановлена.",
            parse_mode="HTML",
        )

    async def _tasklog_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle tasklog: callbacks — show last output of a task."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, task_id = data.split(":", 1)

        task_manager = context.bot_data.get("task_manager")
        if not task_manager:
            await query.edit_message_text("Фоновые задачи не настроены.")
            return

        task = await task_manager.get_task(task_id)
        if not task:
            await query.edit_message_text(
                f"Задача <code>{escape_html(task_id)}</code> не найдена.",
                parse_mode="HTML",
            )
            return

        output = task.last_output or "(нет вывода)"
        safe_output = escape_html(output[:3000])
        await query.edit_message_text(
            f"📋 Задача <code>{task_id}</code>:\n\n"
            f"<pre>{safe_output}</pre>",
            parse_mode="HTML",
        )

    async def _taskretry_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle taskretry: callbacks — restart a failed/timed-out task."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, task_id = data.split(":", 1)

        task_manager = context.bot_data.get("task_manager")
        if not task_manager:
            await query.edit_message_text("Фоновые задачи не настроены.")
            return

        task = await task_manager.get_task(task_id)
        if not task:
            await query.edit_message_text(
                f"Задача <code>{escape_html(task_id)}</code> не найдена.",
                parse_mode="HTML",
            )
            return

        # If task is still running, stop it first
        if task.status == "running":
            try:
                await task_manager.stop_task(task_id)
            except Exception:
                pass

        try:
            new_task_id = await task_manager.start_task(
                prompt=task.prompt,
                project_path=task.project_path,
                user_id=task.user_id,
                chat_id=task.chat_id or query.message.chat_id,
                message_thread_id=task.message_thread_id,
                session_id=task.session_id,
            )
            await query.edit_message_text(
                f"🔄 Задача перезапущена\n"
                f"Новый ID: <code>{new_task_id}</code>",
                parse_mode="HTML",
            )
        except ValueError as e:
            await query.edit_message_text(f"❌ {escape_html(str(e))}")

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )
