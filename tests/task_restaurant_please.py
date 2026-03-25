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

    show_taking_action()
    walkie_audio.speak("I'm heading to see what's on the table")
    robot.nav.go_to(0.525, -1.637, -3.11928)
    input("NAV Correction:")
    walkie_audio.speak("I see 4 bottles. A bottle of water. A green tea. A big coke. And another bottle of water")
    input("Enter:")
    walkie_audio.speak("I'm going back to my starting point")
    robot.nav.go_to(0, 0, 0)
    input("Enter:")
    walkie_audio.speak("I'm waiting for a person to raise their hands")
    input("Enter:")
    walkie_audio.speak("I found a person raising their hands. I'm going there")
    robot.nav.go_to(2.3287, -0.4838, -1.33539)
    input("NAV Correction:")
    walkie_audio.speak("Hello! What would you like?")
    show_listening()
    input("Enter:")
    show_taking_action()
    walkie_audio.speak("Alright! I'm going to grab it for you! Just a moment please")
    robot.nav.go_to(0.525, -1.637, -3.11928)
    input("NAV Correction and ARM:")
    robot.nav.go_to(2.3287, -0.4838, -1.33539)
    walkie_audio.speak("Here is your bottle of tea. Please enjoy!")
    input("NAV Correction and ARM:")
    robot.nav.go_to(0, 0, 0)


if __name__ == "__main__":
    main()
