# TODO: Development Roadmap

## High Priority

- [ ] **LLM Gateway integration** -- Add `GatewayProvider` to `src/llm/` for multi-model routing
  - Support model switching (Claude, GPT, local models via LiteLLM/Ollama)
  - Config: `LLM_PROVIDER=gateway`, `LLM_GATEWAY_URL`, `LLM_GATEWAY_TOKEN`
  - Provider-specific cost tracking

- [ ] **VPS deployment automation** -- Production deployment scripts
  - systemd service file for the bot
  - `deploy/setup.sh` for initial server provisioning
  - Health check endpoint and restart on failure
  - Log rotation configuration

- [ ] **Background task persistence across restarts** -- Currently `TaskManager.recover()` marks orphaned tasks as failed; should support graceful resume
  - Serialize asyncio task state before shutdown
  - Resume in-progress tasks on startup with session context

## Medium Priority

- [ ] **Task retry UI** -- `taskretry:` callback handler starts a new task but loses conversation context beyond session_id
  - Preserve full task parameters (cost budget remaining, retry count)
  - Add user confirmation before retry

- [ ] **Task output streaming** -- Currently only `last_output` is stored; no real-time log following
  - WebSocket or long-poll endpoint for live output
  - `/tasklog --follow` mode

- [ ] **Cost dashboard** -- Aggregate cost data per user, per project, per day
  - `/costs` command for users
  - Admin dashboard via webhook API

- [ ] **Plugin system** -- Third-party extension support
  - Plugin discovery and loading
  - Hook points in middleware chain and event bus

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
- [ ] **Complete testing suite** -- Target >85% coverage across all modules (currently ~57%)

## Known Issues

- [ ] **Race condition in task stop** -- Task can complete between status check and `stop_task()` call (mitigated with try/except, not eliminated)
- [ ] **HeartbeatService stage detection** -- Regex-based stage detection from Claude output is fragile; depends on specific tool call patterns

## Future Ideas

- Model switching per task (e.g., use Claude for code, GPT for docs)
- Local model support via Ollama/LM Studio for cost-free tasks
- Task dependency chains (task B starts after task A completes)
- Git branch per task (automatic feature branch creation)
- Task cost estimation before execution
- Web UI for task monitoring (FastAPI + HTMX)
