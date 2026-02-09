from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from src.audio.walkie import WalkieAudio
from src.agents.robot_state import RobotState
from src.vision.background_detector import BackgroundObjectDetector

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import create_sub_agents_tools, create_speak_tool, think
from ..middleware import RobotStateMiddleware, SequentialToolCallMiddleware, TodoListMiddleware

checkpointer = InMemorySaver()

def create_walkie_agent(model, walkieAudio: WalkieAudio, walkie_vision, walkie_db, tools=[]):
    """Create the main Walkie agent with sub-agent tools.

    Args:
        model: The LLM model to use for this agent and its sub-agents
        walkieAudio: Optional WalkieAudio for the speak tool
        walkie_vision: Optional WalkieVision for the vision agent tools
        walkie_db: Optional WalkieVectorDB for vision agent (find_object, find_scene, scan_and_remember)
        tools: Additional tools to add to the agent

    Returns:
        The configured Walkie agent
    """
    robot = walkie_vision._camera._bot
    tools = create_sub_agents_tools(model, robot=robot, walkie_vision=walkie_vision, walkie_db=walkie_db) + tools

    if walkieAudio:
        tools.append(create_speak_tool(walkieAudio))
    tools.append(think)

    # Start background object detection (YOLO every 3s, dedup radius 1m)
    background_detector = BackgroundObjectDetector(
        vision=walkie_vision,
        db=walkie_db,
        robot=robot,
        interval=3.0,
        dedup_radius=1.0,
    )
    background_detector.start()

    robot_state = RobotState(
        robot,
        vision_enabled=True,
        background_detector=background_detector,
    )

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
