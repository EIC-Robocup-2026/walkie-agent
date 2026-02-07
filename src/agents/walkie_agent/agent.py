from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from src.audio.walkie import WalkieAudio
from src.agents.robot_state import RobotState

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import walkie_tools, initialize_sub_agents, create_speak_tool, think
from ..middleware import RobotStateMiddleware, SequentialToolCallMiddleware, TodoListMiddleware

checkpointer = InMemorySaver()

def create_walkie_agent(model, walkieAudio: WalkieAudio = None, tools=[]):
    """Create the main Walkie agent with sub-agent tools.

    Args:
        model: The LLM model to use for this agent and its sub-agents
        walkieAudio: Optional WalkieAudio for the speak tool
        tools: Additional tools to add to the agent

    Returns:
        The configured Walkie agent
    """
    initialize_sub_agents(model)

    tools = walkie_tools + list(tools)
    if walkieAudio:
        tools.append(create_speak_tool(walkieAudio))
    tools.append(think)

    robot_state = RobotState()

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", 4000),
                keep=("messages", 20),
            ),
            SequentialToolCallMiddleware(),
            RobotStateMiddleware(robot_state),
            TodoListMiddleware(),
        ],
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
