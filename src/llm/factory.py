"""LLM provider factory.

Creates the appropriate LLMProvider based on application settings.
"""

from typing import Any

from .claude_provider import ClaudeProvider
from .interface import LLMProvider


def create_llm_provider(settings: Any, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider based on settings.

    Args:
        settings: Application settings with an `llm_provider` attribute.
            Supported values: "claude_sdk".
        **kwargs: Provider-specific arguments.
            For "claude_sdk": requires `claude_integration` (ClaudeIntegration instance).

    Returns:
        An LLMProvider implementation.

    Raises:
        ValueError: If the provider name is unknown.
        KeyError: If required kwargs are missing for the selected provider.
    """
    provider_name = getattr(settings, "llm_provider", "claude_sdk")

    if provider_name == "claude_sdk":
        return ClaudeProvider(
            claude_integration=kwargs["claude_integration"],
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        f"Supported providers: 'claude_sdk'"
    )
