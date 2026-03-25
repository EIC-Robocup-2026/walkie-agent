import langchain_community
from langchain_benchmarks import registry
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_benchmarks.tool_usage.agents import StandardAgentFactory

registry.filter(Type="ToolUsageTask")

task = registry["Tool Usage - Typewriter (26 tools)"]

env = task.create_environment()

inference_server_url = "http://localhost:8000/v1"

model = ChatOpenAI(
    model="qwen3.5-9b",
    api_key="your api key goes here",
    base_url=inference_server_url,
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}"),  # Populated from task.instructions automatically
        (
            "human",
            "{question}",
        ),  # Each evaluation example is associated with a question
        ("placeholder", "{agent_scratchpad}"),  # Space for the agent to do work
    ]
)

agent_factory = StandardAgentFactory(task, model, prompt)

from langchain import globals

globals.set_verbose(True)
agent = agent_factory()
agent.invoke({"question": "abc"})
globals.set_verbose(False)