"""WalkieAudio - Unified audio interface for speech-to-text and text-to-speech.

Usage:
    from src.audio import WalkieAudio
    
    # Simple initialization with defaults
    audio = WalkieAudio()
    
    # Or with explicit config
    audio = WalkieAudio(
        stt_provider="google",
        stt_config={"language_codes": ["en-US"]},
        tts_provider="elevenlabs",
        tts_config={"voice_id": "21m00Tcm4TlvDq8ikWAM"},
        microphone_device=3,
    )
    
    # Listen and transcribe
    text = audio.listen()
    
    # Speak with streaming
    audio.speak("Hello!")
"""

from typing import Callable

from .stt import STT
from .stt.providers import list_providers as list_stt_providers
from .microphone import Microphone, list_audio_devices, print_audio_devices
from .tts import TTS
from .speaker import list_output_devices, print_output_devices, Speaker
from .tts.providers import list_providers as list_tts_providers


class WalkieAudio:
    """Unified audio interface combining STT and TTS functionality.
    
    Provides simple listen() and speak() methods for voice interaction.
    """

    def __init__(
        self,
        stt_provider: str = "google",
        tts_provider: str = "elevenlabs",
        stt_config: dict | None = None,
        tts_config: dict | None = None,
        microphone_device: int | str | None = None,
        speaker_device: int | None = None,
        microphone_threshold: float = 0.5,
        microphone_min_silence_ms: int = 1000,
    ) -> None:
        """Initialize WalkieAudio with STT and TTS providers.
        
        Args:
            stt_provider: Speech-to-text provider name (e.g., "google").
            tts_provider: Text-to-speech provider name (e.g., "elevenlabs").
            stt_config: Provider-specific STT configuration.
            tts_config: Provider-specific TTS configuration.
            microphone_device: Audio input device ID or name.
            speaker_device: Audio output device index.
            microphone_threshold: VAD sensitivity (0.0-1.0). Higher = less sensitive.
            microphone_min_silence_ms: Silence duration (ms) to end recording.
        """
        self._stt_provider_name = stt_provider
        self._tts_provider_name = tts_provider
        self._stt_config = stt_config or {}
        self._tts_config = tts_config or {}
        
        # Initialize STT
        self._stt = STT(provider=stt_provider, **self._stt_config)
        
        # Initialize TTS
        self._tts = TTS(provider=tts_provider, **self._tts_config)
        
        # Initialize Microphone
        self._microphone = Microphone(
            device=microphone_device,
            threshold=microphone_threshold,
            min_silence_duration_ms=microphone_min_silence_ms,
        )
        
        # Initialize Speaker
        self._speaker = Speaker(device=speaker_device)

    @property
    def stt(self) -> STT:
        """Get the STT instance."""
        return self._stt

    @property
    def tts(self) -> TTS:
        """Get the TTS instance."""
        return self._tts

    @property
    def microphone(self) -> Microphone:
        """Get the Microphone instance."""
        return self._microphone

    @property
    def speaker(self) -> Speaker:
        """Get the Speaker instance."""
        return self._speaker

    def listen(
        self,
        timeout: float = 30.0,
        min_duration: float = 2.0,
        wait_for_speech: bool = True,
    ) -> str:
        """Record audio until silence and transcribe to text.
        
        Args:
            timeout: Maximum recording duration in seconds.
            min_duration: Minimum recording duration before silence can stop.
            wait_for_speech: If True, wait for speech before stopping.
            
        Returns:
            Transcribed text from the recorded audio.
        """
        audio = self._microphone.record_until_silence(
            timeout=timeout,
            min_duration=min_duration,
            wait_for_speech=wait_for_speech,
        )
        return self._stt.transcribe(audio)

    def listen_seconds(self, duration: float = 5.0) -> str:
        """Record audio for a fixed duration and transcribe.
        
        Args:
            duration: Recording duration in seconds.
            
        Returns:
            Transcribed text from the recorded audio.
        """
        audio = self._microphone.record_seconds(duration)
        return self._stt.transcribe(audio)

    def speak(
        self,
        text: str,
        stream: bool = True,
        stream_handler: Callable[[bytes], None] | None = None,
    ) -> bytes | None:
        """Synthesize text to speech and play it.
        
        Args:
            text: The text to speak.
            stream: If True, stream audio for lower latency. If False, wait for
                    complete synthesis before playing.
            stream_handler: Optional callback for each audio chunk during streaming.
            
        Returns:
            Complete audio bytes if streaming, None otherwise.
        """
        output_format = self._get_tts_output_format()
        
        if stream and self._tts.supports_streaming():
            # Stream audio for lower latency
            audio_stream = self._tts.synthesize_stream(text)
            return self._speaker.play_stream(
                audio_stream,
                format=output_format,
                stream_handler=stream_handler,
            )
        else:
            # Non-streaming: wait for complete audio then play
            audio = self._tts.synthesize(text)
            self._speaker.play(audio, format=output_format)
            return None

    def _get_tts_output_format(self) -> str:
        """Get the output format from TTS config or default."""
        return self._tts_config.get("output_format", "pcm_24000")

    def set_stt_provider(self, provider: str, **config) -> None:
        """Switch to a different STT provider.
        
        Args:
            provider: Provider name (e.g., "google").
            **config: Provider-specific configuration.
        """
        self._stt_provider_name = provider
        self._stt_config = config
        self._stt = STT(provider=provider, **config)

    def set_tts_provider(self, provider: str, **config) -> None:
        """Switch to a different TTS provider.
        
        Args:
            provider: Provider name (e.g., "elevenlabs").
            **config: Provider-specific configuration.
        """
        self._tts_provider_name = provider
        self._tts_config = config
        self._tts = TTS(provider=provider, **config)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        self._speaker.stop()

    @staticmethod
    def available_stt_providers() -> list[str]:
        """List all available STT providers."""
        return list_stt_providers()

    @staticmethod
    def available_tts_providers() -> list[str]:
        """List all available TTS providers."""
        return list_tts_providers()

    @staticmethod
    def list_microphones() -> list[dict]:
        """List available microphone devices."""
        return list_audio_devices(input_only=True)

    @staticmethod
    def list_speakers() -> list[dict]:
        """List available speaker devices."""
        return list_output_devices()

    @staticmethod
    def print_devices() -> None:
        """Print all available audio devices."""
        print("\n=== INPUT DEVICES (Microphones) ===")
        print_audio_devices(input_only=True)
        print("\n=== OUTPUT DEVICES (Speakers) ===")
        print_output_devices()
