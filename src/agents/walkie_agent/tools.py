from langchain_core.tools import tool

from src.audio.walkie import WalkieAudio
from ..actuators_agent import create_actuator_agent
from ..vision_agent import create_vision_agent


def create_sub_agents_tools(model, walkie_vision=None, walkie_db=None):
    """Initialize sub-agents with the provided model and optional vision/db. Call this before using the tools."""
    _actuator_agent = create_actuator_agent(model)
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
