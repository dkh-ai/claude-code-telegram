# TODO: Development Roadmap

## Completed

- [x] **Multi-vendor model routing** -- `ChatProviderPool` + `IntentRouter` with hybrid regex/LLM classification
  - Three modes: Agent (Claude), Chat (DeepSeek/GPT), Assistant (plugins)
  - Escalation chains: Sonnet→Opus, DeepSeek→GPT→Haiku→Sonnet
  - `/model` command for manual override, auto-routing by default
  - Provider-specific cost tracking (model, mode, feedback columns in `cost_tracking`)
- [x] **Assistant plugin system** -- Protocol-based architecture in `src/assistant/`
  - `AssistantPlugin` protocol, `PluginRegistry`, `AssistantDispatcher`
  - Built-in `ReminderPlugin` with LLM extraction
  - Confidence-based plugin selection, fallback to chat mode
- [x] **Persistent user memory** -- Three-layer memory in `src/memory/`
  - Working memory (last 20 messages), user profile (facts in DB), conversation summaries
  - `MemoryManager` with recall/extract/store, `FactExtractor` (async, cheap LLM)
  - Memory per-user, shared across chats, injected into system prompts
- [x] **Multi-chat awareness** -- `ChatAwareness` in `src/bot/chat_awareness.py`
  - Private DM: always respond; Group: only on @mention or reply-to-bot
  - Silent observation in groups (memory extraction without responding)
- [x] **Unit tests for new modules** -- 264 tests covering all new modules (797 total)

## High Priority

- [ ] **VPS deployment automation** -- Production deployment scripts
  - systemd service file for the bot
  - `deploy/setup.sh` for initial server provisioning
  - Health check endpoint and restart on failure
  - Log rotation configuration

- [ ] **Background task persistence across restarts** -- Currently `TaskManager.recover()` marks orphaned tasks as failed; should support graceful resume
  - Serialize task parameters before shutdown (prompt, project, cost budget)
  - Re-queue in-progress tasks on startup with session context
  - Distinguish "crashed" vs "intentionally stopped" tasks

- [ ] **Wire routing into orchestrator** -- `IntentRouter`, `ChatProviderPool`, `AssistantDispatcher`, `MemoryManager` created but not yet integrated into `MessageOrchestrator.agentic_text()`
  - Tri-mode execution flow (agent/chat/assistant)
  - Memory injection into prompts
  - Auto-escalation on short/failing responses
  - Feedback buttons (thumbs up/down, retry)

## Medium Priority

- [ ] **Cost dashboard** -- Aggregate cost data per user, per model, per day
  - `/costs` command showing spend by model and mode
  - Leverage new `model`/`mode`/`feedback` columns in `cost_tracking`

- [ ] **Task retry UI** -- `taskretry:` callback handler starts a new task but loses conversation context beyond session_id
  - Preserve full task parameters (cost budget remaining, retry count)
  - Add user confirmation before retry

- [ ] **Task output streaming** -- Currently only `last_output` is stored; no real-time log following
  - WebSocket or long-poll endpoint for live output
  - `/tasklog --follow` mode

- [ ] **More assistant plugins** -- Extend plugin system beyond reminders
  - Knowledge base search plugin
  - Restaurant/place finder plugin
  - Calendar integration plugin

## Low Priority

- [ ] **Multi-user task visibility** -- Currently tasks are visible to all users
  - Filter `/taskstatus` by user_id
  - Admin override to see all tasks

- [ ] **Task scheduling** -- Combine background tasks with APScheduler
  - `/task --cron "0 9 * * *" Run daily health check`
  - Persistent scheduled tasks in database

- [ ] **Task templates** -- Pre-defined task prompts
  - `/task --template code-review`
  - User-defined templates stored in YAML

## Technical Debt

- [ ] **Test coverage for orchestrator.py** -- Large file (~1400 lines), callback handlers need more edge case tests
- [ ] **DRY task_handlers.py** -- Repeated `task_manager` boilerplate in each handler
- [ ] **Type annotations for kwargs** -- Several `**kwargs: object` in notifications and task repository
- [ ] **Audit logging for task commands** -- `/task`, `/taskstop` etc. should log to audit_log table
- [ ] **Increase test coverage** -- Target >85% across all modules (currently ~70% with new tests)

## Known Issues

- [ ] **Race condition in task stop** -- Task can complete between status check and `stop_task()` call (mitigated with try/except, not eliminated)
- [ ] **HeartbeatService stage detection** -- Regex-based stage detection from Claude output is fragile; depends on specific tool call patterns
- [ ] **Router regex boundaries** -- `\b` in Russian patterns doesn't always work at Cyrillic word boundaries; some patterns use truncated forms (e.g. "мин" instead of "минут")

## Future Ideas

- Local model support via Ollama/LM Studio for cost-free tasks
- Task dependency chains (task B starts after task A completes)
- Git branch per task (automatic feature branch creation)
- Task cost estimation before execution
- Web UI for task monitoring (FastAPI + HTMX)
- Intent classification self-improvement from `intent_log` data
- Memory search with embeddings (currently keyword-based)
