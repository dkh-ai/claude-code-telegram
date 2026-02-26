"""Telegram bot authentication middleware."""

from datetime import UTC, datetime
from typing import Any, Callable, Dict

import structlog

logger = structlog.get_logger()


async def auth_middleware(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Check authentication before processing messages.

    This middleware:
    1. Checks if user is authenticated
    2. Attempts authentication if not authenticated
    3. Updates session activity
    4. Logs authentication events
    """
    # Extract user information
    user_id = event.effective_user.id if event.effective_user else None
    username = (
        getattr(event.effective_user, "username", None)
        if event.effective_user
        else None
    )

    if not user_id:
        logger.warning("No user information in update")
        return

    # Resolve pending username (user allowed by @username before they messaged)
    if username:
        storage = data.get("storage")
        if storage:
            try:
                activated = await storage.users.resolve_pending(username, user_id)
                if activated:
                    auth_manager_pre = data.get("auth_manager")
                    if auth_manager_pre:
                        for p in auth_manager_pre.providers:
                            if hasattr(p, "invalidate_cache"):
                                await p.invalidate_cache()
                    logger.info(
                        "Pending user activated",
                        user_id=user_id,
                        username=username,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to resolve pending username",
                    username=username,
                    error=str(e),
                )

    # Get dependencies from context
    auth_manager = data.get("auth_manager")
    audit_logger = data.get("audit_logger")

    if not auth_manager:
        logger.error("Authentication manager not available in middleware context")
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Authentication system unavailable. Please try again later."
            )
        return

    # Check if user is already authenticated
    if auth_manager.is_authenticated(user_id):
        # Update session activity
        if auth_manager.refresh_session(user_id):
            session = auth_manager.get_session(user_id)
            logger.debug(
                "Session refreshed",
                user_id=user_id,
                username=username,
                auth_provider=session.auth_provider if session else None,
            )

        # Group allowlist check (even for existing sessions)
        chat = event.effective_chat
        if chat and getattr(chat, "type", "") in ("group", "supergroup"):
            storage = data.get("storage")
            if storage:
                try:
                    is_group_ok = await storage.allowed_groups.is_allowed(chat.id)
                    if not is_group_ok:
                        logger.debug(
                            "Group not in allowlist (existing session)",
                            chat_id=chat.id,
                            user_id=user_id,
                        )
                        return  # Silently ignore
                except Exception:
                    pass

        # Continue to handler
        return await handler(event, data)

    # User not authenticated - attempt authentication
    logger.info(
        "Attempting authentication for user", user_id=user_id, username=username
    )

    # Try to authenticate (providers will check whitelist and tokens)
    authentication_successful = await auth_manager.authenticate_user(user_id)

    # Log authentication attempt
    if audit_logger:
        await audit_logger.log_auth_attempt(
            user_id=user_id,
            success=authentication_successful,
            method="automatic",
            reason="message_received",
        )

    if authentication_successful:
        session = auth_manager.get_session(user_id)
        logger.info(
            "User authenticated successfully",
            user_id=user_id,
            username=username,
            auth_provider=session.auth_provider if session else None,
        )

        # Group allowlist check
        chat = event.effective_chat
        if chat and getattr(chat, "type", "") in ("group", "supergroup"):
            storage = data.get("storage")
            if storage:
                try:
                    is_group_ok = await storage.allowed_groups.is_allowed(chat.id)
                    if not is_group_ok:
                        logger.info(
                            "Group not in allowlist",
                            chat_id=chat.id,
                            user_id=user_id,
                        )
                        return  # Silently ignore — group not whitelisted
                except Exception as e:
                    logger.warning(
                        "Failed group allowlist check",
                        chat_id=chat.id,
                        error=str(e),
                    )

        # Welcome message for new session
        if event.effective_message:
            await event.effective_message.reply_text(
                f"🔓 Welcome! You are now authenticated.\n"
                f"Session started at {datetime.now(UTC).strftime('%H:%M:%S UTC')}"
            )

        # Continue to handler
        return await handler(event, data)

    else:
        # Authentication failed
        logger.warning("Authentication failed", user_id=user_id, username=username)

        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 <b>Authentication Required</b>\n\n"
                "You are not authorized to use this bot.\n"
                "Please contact the administrator for access.\n\n"
                f"Your Telegram ID: <code>{user_id}</code>\n"
                "Share this ID with the administrator to request access.",
                parse_mode="HTML",
            )
        return  # Stop processing


async def require_auth(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Decorator-style middleware that requires authentication.

    This is a stricter version that only allows authenticated users.
    """
    user_id = event.effective_user.id if event.effective_user else None
    auth_manager = data.get("auth_manager")

    if not auth_manager or not auth_manager.is_authenticated(user_id):
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Authentication required to use this command."
            )
        return

    return await handler(event, data)


async def admin_required(handler: Callable, event: Any, data: Dict[str, Any]) -> Any:
    """Middleware that requires admin privileges.

    Note: This is a placeholder - admin privileges would need to be
    implemented in the authentication system.
    """
    user_id = event.effective_user.id if event.effective_user else None
    auth_manager = data.get("auth_manager")

    if not auth_manager or not auth_manager.is_authenticated(user_id):
        if event.effective_message:
            await event.effective_message.reply_text("🔒 Authentication required.")
        return

    session = auth_manager.get_session(user_id)
    if not session or not session.user_info:
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 Session information unavailable."
            )
        return

    # Check for admin permissions (placeholder logic)
    permissions = session.user_info.get("permissions", [])
    if "admin" not in permissions:
        if event.effective_message:
            await event.effective_message.reply_text(
                "🔒 <b>Admin Access Required</b>\n\n"
                "This command requires administrator privileges.",
                parse_mode="HTML",
            )
        return

    return await handler(event, data)
