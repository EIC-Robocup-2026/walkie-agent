from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from src.audio.tts.providers.elevenlabs import ElevenLabsProvider
from src.agents import create_walkie_agent
from src.audio import WalkieAudio
from src.vision import WalkieVision
from src.db import WalkieVectorDB

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-3-flash-preview:nitro",
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

# Initialize vision (camera + caption + embedding + object detection) and optional vector DB
walkie_vision = WalkieVision(
    camera_device=0,
    caption_provider="google",
    embedding_provider="clip",
    detection_provider="yolo",
)
walkie_db = WalkieVectorDB(persist_directory="chroma_db")

# Create the main Walkie agent with sub-agents for movement and vision
agent = create_walkie_agent(
    model,
    walkie_audio,
    walkie_vision=walkie_vision,
    walkie_db=walkie_db,
)

def main():
    walkie_vision.open()
    try:
        while True:
            print("Recording...")
            text = walkie_audio.listen()
            if text == "":
                continue
            print(f"Transcription: {text}")

            result = agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                {"configurable": {"thread_id": "1"}},
            )
            content = result["messages"][-1].content

            print(content)

            # styled_text = ElevenLabsProvider.style_text(model, content, personality="...")
            # walkie_audio.speak(styled_text)
    finally:
        walkie_vision.close()


if __name__ == "__main__":
    main()
