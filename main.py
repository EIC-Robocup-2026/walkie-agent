from langchain_openai import ChatOpenAI
import os
import wave
from dotenv import load_dotenv
import asyncio
from src.audio.tts.providers.elevenlabs import ElevenLabsProvider
from src.agents import create_walkie_agent
from src.audio import WalkieAudio

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
)

# Create the main Walkie agent with sub-agents for movement and vision
agent = create_walkie_agent(model)

walkie_audio = WalkieAudio(
    stt_provider="whisper",
    tts_provider="elevenlabs",
    microphone_device=3,
    stt_config={
        "model_name": "base",
        "device": "cuda",
    },
    tts_config={
        "voice_id": "fDeOZu1sNd7qahm2fV4k",
        "model_id": "eleven_v3",
    }
)

async def main():
    
    while True:
        print("Recording...")
        text = walkie_audio.listen()
        print(f"Transcription: {text}")
        
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": "1"}})
        content = result["messages"][-1].content
        
        styled_text = ElevenLabsProvider.style_text(model, content, personality="You are a super cute, warm and friendly assistant. You chuckles a lot when you are happy.")
        print(styled_text)
        
        walkie_audio.speak(styled_text)


if __name__ == "__main__":
    asyncio.run(main())
