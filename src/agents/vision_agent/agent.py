"""Vision agent factory: creates agent with tools backed by WalkieVision and WalkieVectorDB."""

from langchain.agents import create_agent

from src.vision import WalkieVision
from src.db.walkie_db import WalkieVectorDB

from ..middleware import SequentialToolCallMiddleware, TodoListMiddleware
from .prompts import VISION_AGENT_SYSTEM_PROMPT
from .tools import get_vision_tools


def create_vision_agent(
    model,
    walkie_vision: WalkieVision | None = None,
    walkie_db: WalkieVectorDB | None = None,
):
    """Create the Vision Agent for visual perception tasks.

    Args:
        model: The LLM model to use for this agent.
        walkie_vision: Optional WalkieVision instance (camera + caption + embedding + detection).
                      If None, the agent is created with no vision tools.
        walkie_db: Optional WalkieVectorDB for find_object, find_scene, scan_and_remember.
                   If None, those tools still exist but will report "database not available".

    Returns:
        The configured Vision agent.
    """
    if walkie_vision is None and walkie_db is None:
        return None
    if walkie_vision is not None:
        tools = get_vision_tools(walkie_vision, walkie_db)
    else:
        tools = []

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            SequentialToolCallMiddleware(),
            TodoListMiddleware(),
        ],
        system_prompt=VISION_AGENT_SYSTEM_PROMPT,
    )
    return agent
