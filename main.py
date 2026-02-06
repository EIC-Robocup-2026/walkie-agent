from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from src.audio.tts.providers.elevenlabs import ElevenLabsProvider
from src.agents import create_walkie_agent
from src.audio import WalkieAudio

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-120b:nitro",
)
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.5
#     ,  # Gemini 3.0+ defaults to 1.0
# )


walkie_audio = WalkieAudio(
    stt_provider="whisper",
    tts_provider="elevenlabs",
    # microphone_device=12,
    stt_config={
        "model_name": "base",
        "device": "cuda",
    },
    tts_config={
        "voice_id": "fDeOZu1sNd7qahm2fV4k",
        "model_id": "eleven_v3",
    }
)

# Initialize camera

# Create the main Walkie agent with sub-agents for movement and vision
agent = create_walkie_agent(model, walkie_audio)

def main():
    
    while True:
        print("Recording...")
        text = walkie_audio.listen()
        if text == "":
            continue
        print(f"Transcription: {text}")
        
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": "1"}})
        content = result["messages"][-1].content
        
        print(content)
        
        # styled_text = ElevenLabsProvider.style_text(model, content, personality="You are a super cute, warm and friendly assistant. You chuckles a lot when you are happy.")
        # print(styled_text)
        
        # walkie_audio.speak(styled_text)


if __name__ == "__main__":
    main()
