from langchain_core.tools import tool

from src.audio.walkie import WalkieAudio
from ..actuators_agent import create_actuator_agent
from ..vision_agent import create_vision_agent


# Store agent instances to be initialized with the model
_actuator_agent = None
_vision_agent = None


def initialize_sub_agents(model):
    """Initialize sub-agents with the provided model. Call this before using the tools."""
    global _actuator_agent, _vision_agent
    _actuator_agent = create_actuator_agent(model)
    _vision_agent = create_vision_agent(model)


@tool(parse_docstring=True)
def control_actuators(task: str) -> str:
    """Command a movement or physical action task to the Actuator Agent. Use this too when you need to move the drive base (must specify absolute or relative) or use your robotic arm.
    
    Use this tool when you need to:
    - Move to a specific location (e.g., "go to coordinates x=5, y=3")
    - Move to a specific location relative to your current position (e.g., "go 2 meters to the right relatively")
    - Use your arm (e.g., "wave hello", "point to the left", "pick up the object", "shake hands")
    - Check your current position
    
    Args:
        task: A natural language description of what physical action to perform.
              Be specific about locations (absolute or relative), directions, or actions needed.
    
    Returns:
        str: The result of the movement/action task
    """
    if _actuator_agent is None:
        return "Error: Actuator agent not initialized. Please initialize sub-agents first."
    
    print(f"Control actuators: {task}")
    result = _actuator_agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })
    
    # Extract the final response from the agent
    return result["messages"][-1].content


@tool(parse_docstring=True)
def use_vision(task: str) -> str:
    """Delegate a vision or perception task to the Vision Agent.
    
    Use this tool when you need to:
    
    **People Detection & Recognition:**
    - Detect people in view and get their positions
    - Recognize poses (standing, sitting, waving, pointing, etc.)
    - Identify known individuals using face recognition (FaceID)
    - Get coordinates and tracking data for detected people
    
    **People Finding:**
    - Search for a specific person by name or FaceID
    - Find where someone was last seen
    
    **Object & Scene Finding:**
    - Search for object locations in the database
    - Find locations matching a description (e.g., "meeting room", "kitchen")
    
    **General Vision:**
    - Look at something and describe it
    - Analyze your surroundings
    - Read signs, documents, or text
    
    Args:
        task: A natural language description of what to look at or analyze.
              Be specific about what information you need.
    
    Returns:
        str: The result of the vision analysis
    
    Examples:
        - "Detect all people in view and tell me their poses"
        - "Is there anyone I recognize? Check faces"
        - "Find where John was last seen"
        - "Where is the meeting room?"
        - "Look around and describe what you see"
        - "Read the text on the sign in front of you"
    """
    if _vision_agent is None:
        return "Error: Vision agent not initialized. Please initialize sub-agents first."
    
    print(f"Use vision: {task}")
    result = _vision_agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })
    
    # Extract the final response from the agent
    return result["messages"][-1].content


def create_speak_tool(walkieAudio: WalkieAudio) -> str:
    @tool(parse_docstring=True)
    def speak(text: str) -> str:
        """Speak the given text out loud. This is a tool that you can use to speak to the user to give them information on what you are doing.
        
        Args:
            text: The text to speak
        
        Returns:
            str: The result of the speech
        """
        print(f"Speaking: {text}")
        walkieAudio.speak(text)
        return "Speech completed"
    return speak

# Export tools for use in the main agent
walkie_tools = [control_actuators, use_vision]
