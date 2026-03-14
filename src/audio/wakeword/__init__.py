"""Wake word module with pluggable detectors."""

from typing import Any

from .base import WakeWordDetector
from .keyword import KeywordWakeWord

PROVIDERS: dict[str, type[WakeWordDetector]] = {
    "keyword": KeywordWakeWord,
}


def get_provider(name: str, config: dict[str, Any]) -> WakeWordDetector:
    """Get a wake word detector by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown wakeword provider: '{name}'. Available: {available}")
    return provider_class(config)


def list_providers() -> list[str]:
    """List all registered wake word providers."""
    return list(PROVIDERS.keys())


__all__ = ["WakeWordDetector", "get_provider", "list_providers"]
