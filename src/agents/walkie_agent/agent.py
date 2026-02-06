from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from src.audio.walkie import WalkieAudio

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import walkie_tools, initialize_sub_agents, create_speak_tool
from ..middleware import SequentialToolCallMiddleware, TodoListMiddleware

checkpointer = InMemorySaver()

def create_walkie_agent(model, walkieAudio: WalkieAudio = None, tools=[]):
    """Create the main Walkie agent with sub-agent tools.
    
    Args:
        model: The LLM model to use for this agent and its sub-agents
    
    Returns:
        The configured Walkie agent
    """
    # Initialize sub-agents with the same model
    initialize_sub_agents(model)
    
    tools = walkie_tools + tools
    if walkieAudio:
        tools.append(create_speak_tool(walkieAudio))
    
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            SequentialToolCallMiddleware(),
            TodoListMiddleware(
                initial_todos=[
                    {
                        "content": "Check the current position of the robot",
                        "status": "pending",
                    },
                    {
                        "content": "Move forward 1 meter",
                        "status": "pending",
                    },
                ],
            ),
        ],
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
