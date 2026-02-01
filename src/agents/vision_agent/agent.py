from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from .prompts import VISION_AGENT_SYSTEM_PROMPT
from .tools import (
    # General vision tools
    capture_image,
    analyze_image,
    describe_surroundings,
    # Vision Understanding - People Detection
    detect_people,
    recognize_pose,
    recognize_face,
    get_people_coordinates,
    # People Finding
    find_person,
    # Object and Scene Finding
    find_object,
    find_scene,
)


# All vision tools grouped by category
VISION_TOOLS = [
    # General vision
    capture_image,
    analyze_image,
    describe_surroundings,
    # Vision Understanding - People Detection
    detect_people,
    recognize_pose,
    recognize_face,
    get_people_coordinates,
    # People Finding
    find_person,
    # Object and Scene Finding
    find_object,
    find_scene,
]


def create_vision_agent(model):
    """Create the Vision Agent for visual perception tasks.
    
    Args:
        model: The LLM model to use for this agent
    
    Returns:
        The configured Vision agent
    """
    agent = create_agent(
        model=model,
        tools=VISION_TOOLS,
        middleware=[
            TodoListMiddleware(),
        ],
        system_prompt=VISION_AGENT_SYSTEM_PROMPT,
    )
    return agent
