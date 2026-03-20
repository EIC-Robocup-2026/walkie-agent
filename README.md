# Combining all hopes and dreams

This repo is a combination of everyones work

Microphone + Silero VAD / Speaker
Implement OpenRouter API
Implement Agents (Walkie Agent / Vision Agent / Actuators Agent)
Implement Actuators Agent
Localization from Pat

TODOS:

TODO: Image Captioning from Tawan
TODO: Give optional initial todo for agent from Tawan
TODO: Implement OpenAI / Ollama from Nine
TODO: Implement with Vision Tools from Tawan
TODO: Image Embed from Pat
TODO: Vector Database from Earth
TODO: Human Face embedding from Tokyo
TODO: Follow Human from 
TODO: Get Image via Walkie SDK from Hardware Team
TODO: VQA ??

## Local STT/TTS + Wakeword

Example (offline Whisper + Piper + wakeword):

```python
from src.audio import WalkieAudio

audio = WalkieAudio(
	stt_provider="whisper",
	stt_config={"model_name": "small", "device": "cpu", "compute_type": "int8"},
	tts_provider="piper",
	tts_config={
		"voice_path": "voices/en_US-lessac-medium.onnx",
		"sample_rate": 22050,
		"output_format": "pcm_22050",
	},
	wakeword_provider="keyword",
	wakeword_config={"keywords": ["walkie"], "stt_provider": "whisper"},
)

text = audio.listen_with_wakeword(wake_timeout=60.0)
audio.speak("Hello!", stream=True)
```

### Streaming vs Non-Streaming TTS

- Streaming (lower latency, requires PCM output format):

```python
audio.speak("Hello!", stream=True)
```

- Non-streaming (wait for full audio, supports MP3 too):

```python
audio.speak("Hello!", stream=False)
```

Tip: For streaming with Piper, set `output_format` to `pcm_16000`, `pcm_22050`, or `pcm_24000`.

### Wakeword Providers

The `keyword` wakeword provider transcribes short audio chunks with STT and
matches keywords (default: `"walkie"`). Configure with:

```python
wakeword_config={
	"keywords": ["walkie", "robot"],
	"stt_provider": "whisper",
	"stt_config": {"model_name": "small", "device": "cpu", "compute_type": "int8"},
}
```