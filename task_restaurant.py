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

screen = WalkieScreen(fullscreen=False, screen_size=(1920, 1080))


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
        preload=True,
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
    show_taking_action()
    print("Step 1: Moving to Counter and describing objects...")
    result = run_agent(
        agent,
        "Go to the 'counter table' from the database. When you arrive, look at the table, detect all objects, and tell me what you see.",
    )
    print(result)
    walkie_audio.speak(result)

    # 2. [Counter] -> [Init] หันไปทางโต๊ะอาหาร
    show_taking_action()
    print("Step 2: Returning to Init and facing dining table...")
    run_agent(
        agent,
        "Go back to the 'start point' from the database",
    )

    # 3. รอคนยกมือ แล้วเดินไปหา
    show_taking_action()
    print("Step 3: Waiting for a raised hand...")
    result = run_agent(
        agent, "Please wait for a person to raise their hands and navigate to them."
    )
    print(result)
    walkie_audio.speak(result)

    # 4. รอรับคำสั่ง (Order)
    print("Step 4: Listening for order...")
    customer_order = listen(walkie_audio)
    print(f"Customer said: {customer_order}")

    # หุ่นยนต์ตอบกลับว่าจะไปหามาให้
    response = run_agent(
        agent,
        f"The customer said: '{customer_order}'. Respond in a friendly way that you will find it for them, then end the turn.",
    )
    walkie_audio.speak(response)

    # 5. [Init] -> [Counter] + Teleop Mode (Pick up)
    show_taking_action()
    print("Step 5: Moving to Counter for Teleop Picking...")
    run_agent(agent, "Go to the 'counter table'. Stop when you are close enough.")

    # เปิดช่วงให้คนบังคับแขน (Teleop)
    screen.show_text("TELEOP MODE: PICKING", background_color=(219, 62, 50))
    print(">>> TELEOP ACTIVE: Manually control the robot arm to pick the object.")
    input("Press Enter once the object is SECURED and you are ready to continue...")

    # 6. [Counter] -> [Table] + Teleop Mode (Place)
    show_taking_action()
    print("Step 6: Moving to Dining Table for Teleop Placing...")
    run_agent(agent, "Go to the 'dining table'. Stop when you are close.")

    screen.show_text("TELEOP MODE: PLACING", background_color=(219, 62, 50))
    print(
        ">>> TELEOP ACTIVE: Manually control the robot to place the object on the table."
    )
    input("Press Enter once the object is PLACED and you are ready to continue...")

    # 7. ถามว่าเอาอะไรอีกไหม?
    print("Step 7: Asking for additional requests...")
    walkie_audio.speak("Is there anything else I can help you with?")

    final_text = listen(walkie_audio)
    print(f"Customer response: {final_text}")

    # 8. ถ้าจบแล้วเดินกลับ [Init]
    if (
        "thank" in final_text.lower()
        or "no" in final_text.lower()
        or "nothing" in final_text.lower()
    ):
        walkie_audio.speak(
            "You're very welcome! Have a great meal. I will return to my station now."
        )
        print("Step 8: Task complete. Returning to Init...")
        run_agent(agent, "Go back to the 'start point' from the database.")
    else:
        walkie_audio.speak(
            "I understand. However, for this demo, I must return to my station. See you soon!"
        )
        run_agent(agent, "Go back to the 'start point' from the database.")

    print("Sequence Finished.")


if __name__ == "__main__":
    main()
