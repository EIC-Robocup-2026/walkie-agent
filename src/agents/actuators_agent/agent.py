
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import move_to_coords, move_to_heading, get_current_pose, command_arm


def create_actuator_agent(model):
    agent = create_agent(
        model=model,
        tools=[move_to_coords, move_to_heading, get_current_pose, command_arm],
        middleware=[
            TodoListMiddleware(),
        ],
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
    return agent