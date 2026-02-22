"""Handlers for /task* commands -- background task management."""

import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..utils.html_format import escape_html

logger = structlog.get_logger()


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /task <prompt> -- start background task."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /task <описание задачи>\n"
            "Пример: /task Добавь JWT авторизацию"
        )
        return

    prompt = " ".join(context.args)
    task_manager = context.bot_data.get("task_manager")
    if not task_manager:
        await update.message.reply_text("Фоновые задачи не настроены.")
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    thread_id = getattr(update.message, "message_thread_id", None)

    project_path = _get_project_path(context)
    if not project_path:
        await update.message.reply_text(
            "Не удалось определить проект. "
            "Используй /repo для выбора проекта."
        )
        return

    try:
        task_id = await task_manager.start_task(
            prompt=prompt,
            project_path=project_path,
            user_id=user_id,
            chat_id=chat_id,
            message_thread_id=thread_id,
        )
        await update.message.reply_text(
            f"✅ Задача запущена\n"
            f"ID: <code>{task_id}</code>\n"
            f"📁 {escape_html(project_path.name)}\n"
            f"📝 {escape_html(prompt[:100])}\n\n"
            f"Буду отправлять обновления по ходу выполнения.",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")


async def taskstatus_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /taskstatus -- show running tasks."""
    task_manager = context.bot_data.get("task_manager")
    if not task_manager:
        await update.message.reply_text("Фоновые задачи не настроены.")
        return

    tasks = await task_manager.get_all_running()
    if not tasks:
        await update.message.reply_text("Нет активных задач.")
        return

    lines = ["🔄 <b>Активные задачи:</b>\n"]
    for t in tasks:
        elapsed = int(
            (datetime.now(timezone.utc) - t.created_at).total_seconds()
        )
        minutes, seconds = divmod(elapsed, 60)
        lines.append(
            f"📁 {escape_html(t.project_path.name)} | "
            f"<code>{t.task_id}</code>\n"
            f"⏱ {minutes}m {seconds}s | 💰 ${t.total_cost:.2f}\n"
            f"📝 {escape_html(t.prompt[:60])}\n"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def tasklog_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /tasklog -- show last output of running task."""
    task_manager = context.bot_data.get("task_manager")
    if not task_manager:
        await update.message.reply_text("Фоновые задачи не настроены.")
        return

    project_path = _get_project_path(context)
    task = None
    if project_path:
        task = await task_manager.get_running_task(project_path)

    if not task:
        tasks = await task_manager.get_all_running()
        task = tasks[0] if tasks else None

    if not task:
        await update.message.reply_text("Нет активных задач.")
        return

    output = task.last_output or "(нет вывода)"
    # Escape and truncate for Telegram
    safe_output = escape_html(output[:3000])
    await update.message.reply_text(
        f"📋 Задача <code>{task.task_id}</code>:\n\n"
        f"<pre>{safe_output}</pre>",
        parse_mode="HTML",
    )


async def taskstop_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /taskstop [task_id] -- stop running task."""
    task_manager = context.bot_data.get("task_manager")
    if not task_manager:
        await update.message.reply_text("Фоновые задачи не настроены.")
        return

    # If task_id provided as argument
    if context.args:
        task_id = context.args[0]
        task = await task_manager.get_task(task_id)
        if not task or task.status != "running":
            await update.message.reply_text(
                f"Задача <code>{escape_html(task_id)}</code> не найдена "
                f"или уже завершена.",
                parse_mode="HTML",
            )
            return
        try:
            await task_manager.stop_task(task_id)
        except Exception as e:
            logger.error("Failed to stop task", task_id=task_id, error=str(e))
            await update.message.reply_text(
                f"Ошибка при остановке задачи: {escape_html(str(e)[:200])}"
            )
            return
        await update.message.reply_text(
            f"⏹ Задача <code>{escape_html(task_id)}</code> остановлена.",
            parse_mode="HTML",
        )
        return

    # Auto-detect task for current project
    project_path = _get_project_path(context)
    task = None
    if project_path:
        task = await task_manager.get_running_task(project_path)

    if not task:
        tasks = await task_manager.get_all_running()
        if not tasks:
            await update.message.reply_text("Нет активных задач для остановки.")
            return
        if len(tasks) == 1:
            task = tasks[0]
        else:
            keyboard = [
                [InlineKeyboardButton(
                    f"{t.project_path.name}: {t.task_id}",
                    callback_data=f"taskstop:{t.task_id}",
                )]
                for t in tasks
            ]
            await update.message.reply_text(
                "Какую задачу остановить?",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

    try:
        await task_manager.stop_task(task.task_id)
    except Exception as e:
        logger.error("Failed to stop task", task_id=task.task_id, error=str(e))
        await update.message.reply_text(
            f"Ошибка при остановке задачи: {escape_html(str(e)[:200])}"
        )
        return
    await update.message.reply_text(
        f"⏹ Задача <code>{task.task_id}</code> остановлена.",
        parse_mode="HTML",
    )


async def taskcontinue_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /taskcontinue <prompt> -- resume with previous session context."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /taskcontinue <уточнение>\n"
            "Возобновляет последнюю завершённую задачу с новым промптом."
        )
        return

    prompt = " ".join(context.args)
    task_manager = context.bot_data.get("task_manager")
    if not task_manager:
        await update.message.reply_text("Фоновые задачи не настроены.")
        return

    chat_id = update.effective_chat.id
    thread_id = getattr(update.message, "message_thread_id", None)
    project_path = _get_project_path(context)

    if not project_path:
        await update.message.reply_text("Не удалось определить проект.")
        return

    last_task = await task_manager.get_task_for_continue(project_path)
    session_id = last_task.session_id if last_task else None

    try:
        task_id = await task_manager.start_task(
            prompt=prompt,
            project_path=project_path,
            user_id=update.effective_user.id,
            chat_id=chat_id,
            message_thread_id=thread_id,
            session_id=session_id,
        )
        resume_note = " (с контекстом предыдущей)" if session_id else ""
        await update.message.reply_text(
            f"✅ Задача запущена{resume_note}\n"
            f"ID: <code>{task_id}</code>",
            parse_mode="HTML",
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")


def _get_project_path(context: ContextTypes.DEFAULT_TYPE) -> Optional[Path]:
    """Determine project path from user context.

    Uses the same mechanism as the existing bot:
    - current_directory from context.user_data (set by /repo or thread routing)
    - Falls back to settings.approved_directory
    """
    current_dir = context.user_data.get("current_directory")
    if current_dir:
        if isinstance(current_dir, str):
            return Path(current_dir)
        return current_dir

    settings = context.bot_data.get("settings")
    if settings:
        return settings.approved_directory

    return None
