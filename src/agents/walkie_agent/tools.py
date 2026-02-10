import math
import threading
import time

from langchain_core.tools import tool

from src.audio.walkie import WalkieAudio
from ..actuators_agent import create_actuator_agent
from ..vision_agent import create_vision_agent


def create_sub_agents_tools(model, robot, walkie_vision, walkie_db):
    """Initialize sub-agents with the provided model and optional vision/db. Call this before using the tools."""
    _actuator_agent = create_actuator_agent(model, robot=robot)
    _vision_agent = create_vision_agent(model, walkie_vision=walkie_vision, walkie_db=walkie_db)

    tools = []

    @tool(parse_docstring=True)
    def control_actuators(task: str) -> str:
        """Command a movement or physical action to the Actuator Agent. Use for drive-base navigation (absolute or relative) or arm actions.

        When to use:
        - Move to map coordinates: "go to x=5, y=3" or "navigate to (2, 1) facing 90 degrees"
        - Move relative to current pose: "go forward 1 meter", "move 0.5 m to the left", "turn left 90 degrees"
        - Arm gestures or manipulation: "wave hello", "point to the left", "pick up the cup", "shake hands"
        - Check pose: "what is my current position?" or "where am I?"

        When NOT to use:
        - For seeing or recognizing something (use use_vision instead)
        - For speaking to the user (use speak instead)
        - For planning a list of steps (use write_todos if there are 3+ steps)

        Args:
            task: Natural language description of the movement or action. Be specific: absolute vs relative, units (meters, degrees), and arm action if needed.

        Returns:
            str: Result of the movement or action (success, pose, or error message).
        """
        if _actuator_agent is None:
            return "Error: Actuator agent not initialized. Please initialize sub-agents first."
        
        print(f"Control actuators: {task}")
        result = _actuator_agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        # Extract the final response from the agent
        return result["messages"][-1].content

    tools.append(control_actuators)
    
    
    if _vision_agent is not None:
        @tool(parse_docstring=True)
        def use_vision(task: str) -> str:
            """Delegate a vision or perception task to the Vision Agent. Use when you need to see, recognize, or find something in the environment.

            When to use:
            - Describe current view: "what do you see?", "describe what's in front of you", "look around and summarize"
            - People: "how many people are in view?", "detect people and their poses", "is anyone I know? check faces", "where is John?" (search by name/FaceID)
            - Objects and places: "where is the coffee mug?" (in view or in database), "find the kitchen" or "find a room with a whiteboard"
            - Text: "read the sign in front of you", "what does the it say?"

            When NOT to use:
            - For moving or turning (use control_actuators)
            - For speaking (use speak)
            - When the user only asks a general question with no need to look (e.g., "what time is it?")

            Args:
                task: Natural language description of what to look at, detect, or find. Be specific (e.g., "find the red cup" not just "find object").

            Returns:
                str: Vision result (descriptions, positions, identities, or "vision disabled" / error).
            """
            if _vision_agent is None:
                return "Error: Vision agent not initialized. Please initialize sub-agents first."
            
            print(f"Use vision: {task}")
            result = _vision_agent.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            
            # Extract the final response from the agent
            return result["messages"][-1].content
    
        tools.append(use_vision)
    
    return tools


def create_speak_tool(walkieAudio: WalkieAudio) -> str:
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak the given text out loud. This is a tool that you can use to speak to the user to give them information beforing calling other tools.
        
        When to use:
        - To give the user information before calling other tools (performing actions)
        - You MAY NOT use this tool to speak out loud the final answer to the user. Instead, return the final answer in the agent response.
        
        Args:
            text: The text to speak
        
        Returns:
            str: The result of the speech
        """
        print(f"Speaking: {text}")
        walkieAudio.speak(text)
        return "Speech completed"
    return speak

@tool(parse_docstring=True)
def think(thought: str) -> str:
    """This is a tool that you can use to think about the task at hand.
    
    Args:
        thought: The thought to think about
    
    Returns:
        str: The result of the thinking
    """
    print(f"Thinking: {thought}")
    return "Thinking completed"


FOLLOW_STOP_DISTANCE = 0.7  # meters – how close the robot approaches the person


def create_follow_person_tool(robot, walkie_vision, walkie_audio: WalkieAudio):
    """Create a tool that continuously follows the nearest person until the user says 'stop'.

    Args:
        robot: WalkieRobot instance for navigation and position helpers.
        walkie_vision: WalkieVision instance for capturing images and detecting objects.
        walkie_audio: WalkieAudio instance for listening to voice commands.

    Returns:
        A langchain tool.
    """

    @tool(parse_docstring=True)
    def follow_person() -> str:
        """Continuously follow the nearest person in view, keeping ~0.7 m distance.

        The robot will detect the biggest visible person each cycle, compute a goal
        position 0.7 m away from them, and navigate there. It keeps looping until
        the user says "stop" (detected via the microphone).

        When to use:
        - The user asks you to follow them or follow a person.
        - "follow me", "come with me", "follow that person", etc.

        When NOT to use:
        - The user just wants to move to a fixed location (use control_actuators).
        - The user wants to look at or identify a person (use use_vision).

        Returns:
            str: Summary of the follow session (stopped by user or error).
        """
        stop_event = threading.Event()

        def _listen_for_stop():
            """Background thread: listen for the word 'stop' via STT."""
            while not stop_event.is_set():
                try:
                    text = walkie_audio.listen(timeout=10.0, min_duration=1.0)
                    if text and "stop" in text.lower():
                        print(f"[follow_person] Heard stop command: '{text}'")
                        stop_event.set()
                        return
                except Exception as e:
                    print(f"[follow_person] STT listener error: {e}")
                    # Brief pause before retrying to avoid tight error loops
                    time.sleep(0.5)

        # Start the STT listener thread
        listener_thread = threading.Thread(target=_listen_for_stop, daemon=True)
        listener_thread.start()
        print("[follow_person] Started following. Say 'stop' to end.")

        iterations = 0
        try:
            while not stop_event.is_set():
                image = walkie_vision.capture()
                if image is None:
                    time.sleep(0.1)
                    continue

                objects = walkie_vision.detect_objects(image)
                persons = [obj for obj in objects if obj.class_name and obj.class_name.lower() == "person"]

                if persons:
                    # Pick the biggest person by bbox area (w * h)
                    biggest_person = max(persons, key=lambda obj: (obj.bbox[2] * obj.bbox[3]))

                    # Get world position of the person
                    target_pos = robot.tools.bboxes_to_positions([biggest_person.bbox])[0]
                    curr_pos = robot.status.get_pose()

                    tx, ty = target_pos[0], target_pos[1]
                    rx, ry = curr_pos["x"], curr_pos["y"]

                    dx = rx - tx
                    dy = ry - ty
                    dist = math.sqrt(dx**2 + dy**2)

                    if dist > 0:
                        ratio = FOLLOW_STOP_DISTANCE / dist
                        goal_x = tx + (dx * ratio)
                        goal_y = ty + (dy * ratio)

                        # Face the person
                        angle_to_person = math.atan2(-dy, -dx)
                        robot.nav.go_to(goal_x, goal_y, angle_to_person, blocking=False)

                    iterations += 1
                else:
                    # No person visible – stop moving and wait
                    robot.nav.stop()

                time.sleep(0.1)

        except Exception as e:
            robot.nav.stop()
            stop_event.set()
            listener_thread.join(timeout=2.0)
            return f"Follow person ended due to error: {e}"

        # Clean up
        robot.nav.stop()
        listener_thread.join(timeout=2.0)
        print("[follow_person] Stopped following.")
        return "Stopped following the person as requested."

    return follow_person
