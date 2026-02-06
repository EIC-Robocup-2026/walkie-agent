
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import move_absolute, move_relative, get_current_pose, command_arm

from ..middleware import SequentialToolCallMiddleware, TodoListMiddleware

def create_actuator_agent(model):
    agent = create_agent(
        model=model,
        tools=[move_absolute, move_relative, get_current_pose, command_arm],
        middleware=[
            SequentialToolCallMiddleware(),
            TodoListMiddleware(),
        ],
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
    return agent