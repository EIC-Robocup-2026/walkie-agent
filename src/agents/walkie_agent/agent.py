from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import walkie_tools, initialize_sub_agents

checkpointer = InMemorySaver()

def create_walkie_agent(model):
    """Create the main Walkie agent with sub-agent tools.
    
    Args:
        model: The LLM model to use for this agent and its sub-agents
    
    Returns:
        The configured Walkie agent
    """
    # Initialize sub-agents with the same model
    initialize_sub_agents(model)
    
    agent = create_agent(
        model=model,
        tools=walkie_tools,
        middleware=[
            TodoListMiddleware(),
        ],
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
