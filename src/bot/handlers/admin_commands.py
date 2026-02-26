"""Admin commands for master user management.

Commands:
- /allow <@username|user_id> - Add user to allowlist
- /deny <@username|user_id> - Remove user from allowlist
- /allowgroup <@groupname|chat_id> - Add group to allowlist
- /denygroup <@groupname|chat_id> - Remove group from allowlist
- /listusers - List all users
- /listgroups - List all allowed groups
- /whoami - Show your info
"""

import structlog
from telegram import Update
from telegram.ext import ContextTypes

from ..utils.html_format import escape_html

logger = structlog.get_logger()


def _is_master(user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is the master user."""
    master_id = context.bot_data.get("master_user_id")
    return master_id is not None and user_id == master_id


async def _master_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check master access and reply if denied. Returns True if allowed."""
    if _is_master(update.effective_user.id, context):
        return True
    await update.message.reply_text(
        "🔒 <b>Master Access Required</b>\n\n"
        "This command is only available to the master user.",
        parse_mode="HTML",
    )
    return False


async def allow_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Add user to allowlist: /allow <@username|user_id>."""
    if not await _master_only(update, context):
        return

    args = update.message.text.split()[1:] if update.message.text else []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/allow @username</code> or <code>/allow 123456</code>",
            parse_mode="HTML",
        )
        return

    storage = context.bot_data.get("storage")
    auth_manager = context.bot_data.get("auth_manager")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    target = args[0]

    if target.startswith("@"):
        # Username-based
        username = target.lstrip("@")
        existing = await storage.users.find_by_username(username)
        if existing and existing.user_id > 0:
            # Known user — just allow
            await storage.users.set_user_allowed(existing.user_id, True)
            if auth_manager:
                for p in auth_manager.providers:
                    if hasattr(p, "invalidate_cache"):
                        await p.invalidate_cache()
            await update.message.reply_text(
                f"✅ User <code>@{escape_html(username)}</code> "
                f"(ID: <code>{existing.user_id}</code>) allowed.",
                parse_mode="HTML",
            )
        else:
            # Unknown user — create pending
            if existing and existing.user_id < 0:
                # Already pending
                await update.message.reply_text(
                    f"⏳ <code>@{escape_html(username)}</code> already pending activation.",
                    parse_mode="HTML",
                )
                return
            temp_id = await storage.users.create_pending_user(username)
            if auth_manager:
                for p in auth_manager.providers:
                    if hasattr(p, "invalidate_cache"):
                        await p.invalidate_cache()
            await update.message.reply_text(
                f"⏳ Pending activation for <code>@{escape_html(username)}</code>.\n"
                f"They will be activated when they first message the bot.",
                parse_mode="HTML",
            )
    else:
        # Numeric ID
        try:
            uid = int(target)
        except ValueError:
            await update.message.reply_text(
                "Invalid argument. Use <code>@username</code> or numeric ID.",
                parse_mode="HTML",
            )
            return

        existing = await storage.users.get_user(uid)
        if existing:
            await storage.users.set_user_allowed(uid, True)
        else:
            from src.storage.models import UserModel

            await storage.users.create_user(UserModel(user_id=uid, is_allowed=True))

        if auth_manager:
            for p in auth_manager.providers:
                if hasattr(p, "invalidate_cache"):
                    await p.invalidate_cache()
        await update.message.reply_text(
            f"✅ User <code>{uid}</code> allowed.",
            parse_mode="HTML",
        )


async def deny_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Remove user from allowlist: /deny <@username|user_id>."""
    if not await _master_only(update, context):
        return

    args = update.message.text.split()[1:] if update.message.text else []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/deny @username</code> or <code>/deny 123456</code>",
            parse_mode="HTML",
        )
        return

    storage = context.bot_data.get("storage")
    auth_manager = context.bot_data.get("auth_manager")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    target = args[0]

    if target.startswith("@"):
        username = target.lstrip("@")
        existing = await storage.users.find_by_username(username)
        if not existing:
            await update.message.reply_text(
                f"User <code>@{escape_html(username)}</code> not found.",
                parse_mode="HTML",
            )
            return
        uid = existing.user_id
    else:
        try:
            uid = int(target)
        except ValueError:
            await update.message.reply_text(
                "Invalid argument. Use <code>@username</code> or numeric ID.",
                parse_mode="HTML",
            )
            return

    # Don't allow denying master
    master_id = context.bot_data.get("master_user_id")
    if master_id and uid == master_id:
        await update.message.reply_text("Cannot deny the master user.")
        return

    await storage.users.set_user_allowed(uid, False)
    if auth_manager:
        auth_manager.end_session(uid)
        for p in auth_manager.providers:
            if hasattr(p, "invalidate_cache"):
                await p.invalidate_cache()

    display = f"@{escape_html(target.lstrip('@'))}" if target.startswith("@") else str(uid)
    await update.message.reply_text(
        f"🚫 User <code>{display}</code> denied.",
        parse_mode="HTML",
    )


async def allowgroup_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Add group to allowlist: /allowgroup <@groupname|chat_id>."""
    if not await _master_only(update, context):
        return

    args = update.message.text.split()[1:] if update.message.text else []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/allowgroup @groupname</code> or "
            "<code>/allowgroup -100123456</code>",
            parse_mode="HTML",
        )
        return

    storage = context.bot_data.get("storage")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    target = args[0]
    master_id = update.effective_user.id

    try:
        if target.startswith("@"):
            chat = await context.bot.get_chat(target)
        else:
            chat = await context.bot.get_chat(int(target))
    except Exception as e:
        await update.message.reply_text(
            f"Cannot resolve group: <code>{escape_html(str(e)[:200])}</code>",
            parse_mode="HTML",
        )
        return

    await storage.allowed_groups.add(
        group_id=chat.id,
        title=chat.title or str(chat.id),
        added_by=master_id,
        username=getattr(chat, "username", None),
    )

    await update.message.reply_text(
        f"✅ Group <b>{escape_html(chat.title or str(chat.id))}</b> "
        f"(<code>{chat.id}</code>) added to allowlist.",
        parse_mode="HTML",
    )


async def denygroup_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Remove group from allowlist: /denygroup <@groupname|chat_id>."""
    if not await _master_only(update, context):
        return

    args = update.message.text.split()[1:] if update.message.text else []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/denygroup @groupname</code> or "
            "<code>/denygroup -100123456</code>",
            parse_mode="HTML",
        )
        return

    storage = context.bot_data.get("storage")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    target = args[0]

    try:
        if target.startswith("@"):
            chat = await context.bot.get_chat(target)
            group_id = chat.id
        else:
            group_id = int(target)
    except Exception as e:
        await update.message.reply_text(
            f"Cannot resolve group: <code>{escape_html(str(e)[:200])}</code>",
            parse_mode="HTML",
        )
        return

    removed = await storage.allowed_groups.remove(group_id)
    if removed:
        await update.message.reply_text(
            f"🚫 Group <code>{group_id}</code> removed from allowlist.",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            f"Group <code>{group_id}</code> was not in the allowlist.",
            parse_mode="HTML",
        )


async def listusers_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """List all users: /listusers."""
    if not await _master_only(update, context):
        return

    storage = context.bot_data.get("storage")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    users = await storage.users.get_all_users()
    master_id = context.bot_data.get("master_user_id")

    allowed = []
    pending = []
    denied = []

    for u in users:
        name = f"@{u.telegram_username}" if u.telegram_username else str(u.user_id)
        is_master = master_id and u.user_id == master_id
        label = f"<code>{escape_html(name)}</code>"
        if u.user_id > 0:
            label += f" (ID: {u.user_id})"
        if is_master:
            label += " 👑"

        if u.pending_username and u.user_id < 0:
            pending.append(label)
        elif u.is_allowed:
            allowed.append(label)
        else:
            denied.append(label)

    lines = ["<b>Users</b>\n"]
    if allowed:
        lines.append(f"<b>Allowed ({len(allowed)}):</b>")
        lines.extend(f"  ✅ {u}" for u in allowed)
    if pending:
        lines.append(f"\n<b>Pending ({len(pending)}):</b>")
        lines.extend(f"  ⏳ {u}" for u in pending)
    if denied:
        lines.append(f"\n<b>Denied ({len(denied)}):</b>")
        lines.extend(f"  🚫 {u}" for u in denied)

    if not users:
        lines.append("No users found.")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def listgroups_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """List all allowed groups: /listgroups."""
    if not await _master_only(update, context):
        return

    storage = context.bot_data.get("storage")
    if not storage:
        await update.message.reply_text("Storage not available.")
        return

    groups = await storage.allowed_groups.get_all()

    if not groups:
        await update.message.reply_text("No groups in allowlist.")
        return

    lines = [f"<b>Allowed Groups ({len(groups)}):</b>\n"]
    for g in groups:
        name = g.group_title
        username = f" (@{g.group_username})" if g.group_username else ""
        lines.append(
            f"  • <b>{escape_html(name)}</b>{username}\n"
            f"    ID: <code>{g.group_id}</code>"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def whoami_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show user info: /whoami. Available to all authenticated users."""
    user_id = update.effective_user.id
    username = getattr(update.effective_user, "username", None)
    storage = context.bot_data.get("storage")
    master_id = context.bot_data.get("master_user_id")
    is_master = master_id and user_id == master_id

    lines = ["<b>Your Info</b>\n"]
    lines.append(f"User ID: <code>{user_id}</code>")
    if username:
        lines.append(f"Username: @{escape_html(username)}")
    lines.append(f"Master: {'Yes 👑' if is_master else 'No'}")

    if storage:
        user = await storage.users.get_user(user_id)
        if user:
            lines.append(f"Allowed: {'Yes' if user.is_allowed else 'No'}")
            lines.append(f"Messages: {user.message_count}")
            lines.append(f"Total cost: ${user.total_cost:.4f}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")
