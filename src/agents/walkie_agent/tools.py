from langchain_core.tools import tool
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
    """Delegate a movement or physical action task to the Actuator Agent. Use this too when you need to move or use your arms.
    
    Use this tool when you need to:
    - Move to a specific location (e.g., "go to coordinates x=5, y=3")
    - Turn to face a direction (e.g., "turn to face 90 degrees")
    - Use your arm (e.g., "wave hello", "point to the left", "pick up the object", "shake hands")
    - Check your current position
    
    Args:
        task: A natural language description of what physical action to perform.
              Be specific about locations, directions, or actions needed.
    
    Returns:
        str: The result of the movement/action task
    
    Examples:
        - "Move to coordinates x=2.5, y=1.0 facing north (90 degrees)"
        - "Wave your arm to greet the visitor"
        - "Turn around to face the entrance (heading 180 degrees)"
        - "Get your current position"
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


# Export tools for use in the main agent
walkie_tools = [control_actuators, use_vision]
