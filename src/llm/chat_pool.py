"""Chat provider pool — registry of providers by vendor/model prefix."""

from typing import Optional

import structlog

from ..config.settings import Settings
from .chat_provider import ChatProvider

logger = structlog.get_logger()

# Vendor routing: model prefix -> (base_url, api_key_attr)
_VENDOR_MAP = {
    "deepseek": ("https://api.deepseek.com", "deepseek_api_key_str"),
    "gpt": (None, "openai_api_key_str"),  # default OpenAI base_url
    "o1": (None, "openai_api_key_str"),
    "o3": (None, "openai_api_key_str"),
    "o4": (None, "openai_api_key_str"),
}


class ChatProviderPool:
    """Registry of chat providers by vendor."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, ChatProvider] = {}

    def _resolve_vendor(self, model: str) -> tuple[Optional[str], Optional[str]]:
        """Resolve vendor base_url and api_key from model name."""
        for prefix, (base_url, key_attr) in _VENDOR_MAP.items():
            if model.startswith(prefix):
                api_key = getattr(self._settings, key_attr, None)
                return base_url, api_key
        # Fallback: try OpenAI
        return None, self._settings.openai_api_key_str

    def get_for_model(self, model: str) -> Optional[ChatProvider]:
        """Get or create a provider for the given model."""
        if model in self._providers:
            return self._providers[model]

        base_url, api_key = self._resolve_vendor(model)
        if not api_key:
            logger.warning("No API key for model", model=model)
            return None

        provider = ChatProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self._providers[model] = provider
        return provider

    def get_router_provider(self) -> Optional[ChatProvider]:
        """Get the cheapest provider for intent classification."""
        return self.get_for_model(self._settings.model_router_llm)

    def available_vendors(self) -> list[str]:
        """List available vendors based on configured API keys."""
        vendors = []
        if self._settings.deepseek_api_key_str:
            vendors.append("deepseek")
        if self._settings.openai_api_key_str:
            vendors.append("openai")
        if self._settings.anthropic_api_key_str:
            vendors.append("anthropic")
        return vendors
