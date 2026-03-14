"""Wake word detector base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
import time


class MicrophoneLike(Protocol):
    """Protocol for microphone implementations."""

    def record_seconds(self, duration: float) -> bytes:
        """Record audio for a fixed duration."""
        ...


class WakeWordDetector(ABC):
    """Abstract wake word detector."""

    @abstractmethod
    def detect(self, audio_content: bytes) -> bool:
        """Return True if wake word is detected in audio."""
        raise NotImplementedError

    def listen_until_wake(
        self,
        microphone: MicrophoneLike,
        timeout: float = 30.0,
        chunk_duration: float = 1.5,
    ) -> bool:
        """Listen in short chunks until wake word is detected or timeout."""
        start = time.time()
        while (time.time() - start) < timeout:
            audio = microphone.record_seconds(chunk_duration)
            if self.detect(audio):
                return True
        return False
