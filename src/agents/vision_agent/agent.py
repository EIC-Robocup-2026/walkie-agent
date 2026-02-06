from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from vision.camera import WalkieCamera

from ..common import DisableParallelToolCallsMiddleware

from .prompts import VISION_AGENT_SYSTEM_PROMPT
from .tools import get_vision_tools


def create_vision_agent(model, walkieCamera: WalkieCamera = None):
    """Create the Vision Agent for visual perception tasks.
    
    Args:
        model: The LLM model to use for this agent
    
    Returns:
        The configured Vision agent
    """
    if walkieCamera:
        tools = get_vision_tools(walkieCamera)
    else:
        tools = []
    
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            DisableParallelToolCallsMiddleware(),
            TodoListMiddleware(),
        ],
        system_prompt=VISION_AGENT_SYSTEM_PROMPT,
    )
    return agent
