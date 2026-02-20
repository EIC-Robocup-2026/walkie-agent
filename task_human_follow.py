import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from walkie_sdk import WalkieRobot

from src.agents import create_walkie_agent
from src.audio import WalkieAudio
from src.audio.tts.providers.elevenlabs import ElevenLabsProvider
from src.db import WalkieVectorDB
from src.screen import WalkieScreen
from src.vision import WalkieVision

load_dotenv()

screen = WalkieScreen()


def show_listening():
    screen.show_text("Listening...", font_size=128, background_color=(93, 189, 9))


def show_initializing():
    screen.show_text("Initializing...", font_size=128, background_color=(232, 179, 21))


def show_taking_action():
    screen.show_text("Taking Action...", font_size=128, background_color=(219, 62, 50))


def run_agent(agent, text):
    # ส่งคำสั่งให้ Agent และคืนค่าผลลัพธ์
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": "restaurant_task"}},
    )
    content = result["messages"][-1].content
    return content


def listen(walkie_audio):
    show_listening()
    print("Recording...")
    text = walkie_audio.listen()
    return text


def main():
    show_initializing()

    # --- Initialization (คงเดิมตาม main.py) ---
    model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash:nitro",
    )

    ZENOH_PORT = 7447
    robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

    robot = WalkieRobot(
        ip=robot_ip,
        camera_protocol="zenoh",
        camera_port=ZENOH_PORT,
    )

    walkie_audio = WalkieAudio(
        stt_provider="whisper",
        tts_provider="elevenlabs",
        stt_config={"model_name": "base", "device": "cuda"},
        tts_config={"voice_id": "fDeOZu1sNd7qahm2fV4k", "model_id": "eleven_v3"},
    )

    walkie_vision = WalkieVision(
        robot,
        caption_provider="paligemma",
        embedding_provider="clip",
        detection_provider="yolo",
        pose_provider="yolo_pose",
        preload=True
    )
    walkie_db = WalkieVectorDB(persist_directory="chroma_db")

    agent = create_walkie_agent(
        model,
        walkie_audio,
        walkie_vision=walkie_vision,
        walkie_db=walkie_db,
    )

    # --- SEQUENCE START ---

    # 1. [Init] -> [Counter] และบอกว่ามีอะไรอยู่บนโต๊ะบ้าง
    listen(walkie_audio)
    show_taking_action()
    result = run_agent(
        agent,
        "Please follow the person in front of you.",
    )
    print(result)
    walkie_audio.speak(result)


if __name__ == "__main__":
    main()
