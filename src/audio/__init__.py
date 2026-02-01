"""Unified audio module for speech-to-text and text-to-speech.

Usage:
    from src.audio import WalkieAudio
    
    # Simple initialization
    audio = WalkieAudio()
    
    # Listen and transcribe
    text = audio.listen()
    
    # Speak with streaming
    audio.speak("Hello!")
    
    # Or use components directly
    from src.audio import STT, TTS, Microphone, Speaker
"""

from .walkie import WalkieAudio
from .stt import STT, STTProvider
from .tts import TTS, TTSProvider
from .microphone import Microphone, list_audio_devices, print_audio_devices
from .speaker import Speaker, list_output_devices, print_output_devices

__all__ = [
    # Main interface
    "WalkieAudio",
    # STT
    "STT",
    "STTProvider",
    "Microphone",
    "list_audio_devices",
    "print_audio_devices",
    # TTS
    "TTS",
    "TTSProvider",
    "Speaker",
    "list_output_devices",
    "print_output_devices",
]
