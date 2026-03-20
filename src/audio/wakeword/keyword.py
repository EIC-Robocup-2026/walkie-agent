"""Keyword-based wake word detector using STT."""

from __future__ import annotations

from typing import Any

from .base import WakeWordDetector
from ..stt import STT


class KeywordWakeWord(WakeWordDetector):
    """Wake word detector that transcribes short audio and matches keywords."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.keywords = [k.lower() for k in config.get("keywords", ["walkie"])]
        self.stt_provider = config.get("stt_provider", "whisper")
        self.stt_config = config.get("stt_config", {})
        self.prompt = config.get("prompt")
        self.stt = STT(provider=self.stt_provider, **self.stt_config)

    def detect(self, audio_content: bytes) -> bool:
        text = self.stt.transcribe(audio_content, prompt=self.prompt)
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.keywords)
