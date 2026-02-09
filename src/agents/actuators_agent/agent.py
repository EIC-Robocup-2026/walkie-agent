from langchain.agents import create_agent

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import create_actuators_agent_tools
from ..middleware import SequentialToolCallMiddleware, TodoListMiddleware


def create_actuator_agent(model, robot) :
    """Create the actuator agent for robot movement and arm control.
    Args:
        model: The LLM model to use for this agent
        robot: The WalkieRobot instance to use for actuator commands
    Returns:
        The configured actuator agent
    """
    
    agent = create_agent(
        model=model,
        tools=create_actuators_agent_tools(robot),
        middleware=[
            SequentialToolCallMiddleware(),
            TodoListMiddleware(),
        ],
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
    return agent