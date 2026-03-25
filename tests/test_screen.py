from src.screen import WalkieScreen
import time
screen = WalkieScreen()

def show_listening():
    screen.show_text("Listening...", font_size=128, background_color=(93, 189, 9))

def show_thinking():
    screen.show_text("Thinking...", font_size=128, background_color=(232, 179, 21))

def show_taking_action():
    screen.show_text("Taking Action...", font_size=128, background_color=(219, 62, 50))

show_listening()
time.sleep(1)
show_thinking()
time.sleep(1)
show_taking_action()
time.sleep(1)
screen.clear()
screen.close()