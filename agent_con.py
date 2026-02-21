from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from walkie_sdk import WalkieRobot
from src.audio.tts.providers.elevenlabs import ElevenLabsProvider
from src.agents import create_walkie_agent
from src.audio import WalkieAudio
from src.vision import WalkieVision
from src.db import WalkieVectorDB
from src.screen import WalkieScreen
import time

load_dotenv()

# Track people until the person say Thank you. go back to the starting point.
# VQA about the current camera view and DB.
# Grab an object and put it somewhere.

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-pro:nitro",
)
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.5
#     ,  # Gemini 3.0+ defaults to 1.0
# )

ZENOH_PORT = 7447

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
robot = WalkieRobot(
    ip=robot_ip,
    camera_protocol="zenoh",
    camera_port=ZENOH_PORT,
)

walkie_audio = WalkieAudio(
    # microphone_device=13,
    stt_provider="whisper",
    tts_provider="elevenlabs",
    stt_config={
        "model_name": "base",
        "device": "cuda",
    },
    tts_config={
        "voice_id": "fDeOZu1sNd7qahm2fV4k",
        "model_id": "eleven_v3",
    }
)

for mic in walkie_audio.list_microphones():
    print(mic)

# Initialize vision (camera + caption + embedding + object detection) and optional vector DB
walkie_vision = WalkieVision(
    robot,
    caption_provider="paligemma",
    embedding_provider="clip",
    detection_provider="yolo",
    pose_provider="yolo_pose",
    preload=True,
)
walkie_db = WalkieVectorDB(persist_directory="chroma_db")

screen = WalkieScreen(fullscreen=False, screen_size=(1920, 1080))
# Create the main Walkie agent with sub-agents for movement and vision
agent = create_walkie_agent(
    model,
    walkie_audio,
    walkie_vision=walkie_vision,
    walkie_db=walkie_db,
)

def show_listening():
    screen.show_text("Listening...", font_size=128, background_color=(93, 189, 9))

def show_initializing():
    screen.show_text("Initializing...", font_size=128, background_color=(232, 179, 21))

def show_taking_action():
    screen.show_text("Taking Action...", font_size=128, background_color=(219, 62, 50))

def run_agent(text):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": "1"}})
    content = result["messages"][-1].content
    return content

# Can you follow the person in front of you

def listen():
    show_listening()
    print("Recording...")
    text = walkie_audio.listen()
    if text == "" or text.lower().startswith("you are walkie agent"): # Whisper sometimes picks up the system prompt; ignore it if it does.
        return ""
    return text

# result = run_agent("Can you find how many chairs are there in the database? Please count it.")
# print(result)

# while True:
#     pass

def main():
    show_initializing()

    time.sleep(10)
    
    while True:
        text = listen()
        # Walke work == "walkie"
        if text == "" or "walkie" not in text.lower():
            continue
        print(f"Transcription: {text}")
        show_taking_action()

        result = run_agent(text)
        print(result)
        walkie_audio.speak(result)


if __name__ == "__main__":
    main()
