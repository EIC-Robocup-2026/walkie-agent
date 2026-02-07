from langchain.agents import create_agent

from src.agents.robot_state import RobotState

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import move_absolute, move_relative, get_current_pose, command_arm
from ..middleware import RobotStateMiddleware, SequentialToolCallMiddleware, TodoListMiddleware


def create_actuator_agent(model):
    robot_state = RobotState()
    agent = create_agent(
        model=model,
        tools=[move_absolute, move_relative, get_current_pose, command_arm],
        middleware=[
            SequentialToolCallMiddleware(),
            RobotStateMiddleware(robot_state),
            TodoListMiddleware(),
        ],
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
    return agent