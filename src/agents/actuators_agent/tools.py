from langchain_core.tools import tool
from walkie_sdk import WalkieRobot
import os

from dotenv import load_dotenv

load_dotenv()

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
# robot = WalkieRobot(ip=robot_ip, enable_camera=False)

def _get_current_pose():
    """Get the current pose of the robot"""
    print("Getting current pose of the robot")
    # Test
    return f"Current pose of the robot: x=0.00, y=0.00, θ=0.00"
    pose = robot.status.get_pose()
    if pose is None:
        raise ValueError("Unable to get robot pose at the moment")
    return pose

@tool(parse_docstring=True)
def move_to_coords(x: float, y: float, heading: float = 0.0) -> str:
    """Move the robot to the specified coordinates x and y in meters on the map

    Args:
        x (float): The x coordinate to move to
        y (float): The y coordinate to move to
        heading (float): The heading to move to (in degrees)

    Returns:
        str: The result of the movement
    """
    print(f"Moving robot to x: {x}, y: {y}, heading: {heading}")
    
    # Test
    return f"Robot moved to x: {x}, y: {y}, heading: {heading}"
    
    result = robot.nav.go_to(x=x, y=y, heading=heading, blocking=True)
    return f"Robot navigation completed: {result}"

@tool
def move_to_heading(heading: float) -> str:
    """Move the robot to the specified heading in degrees"""
    print(f"Moving robot to heading: {heading}")
    
    # Test
    return f"Robot moved to heading: {heading}"
    # Get current pose
    pose = _get_current_pose()
    x = pose.x
    y = pose.y
    result = robot.nav.go_to(x=x, y=y, heading=heading, blocking=True)
    return f"Robot navigation completed: {result}"

@tool
def get_current_pose() -> str:
    """Get the current pose of the robot"""
    print("Getting current pose of the robot")
    # Test
    return f"Current pose of the robot: x=0.00, y=0.00, θ=0.00"
    pose = _get_current_pose()
    # Copied from Walkie SDK
    return f"Current pose of the robot: x={pose['x']:+6.2f}  y={pose['y']:+6.2f}  θ={pose['heading']:+5.2f}"

@tool
def command_arm(action: str) -> str:
    """Command the arm of the robot to perform the specified action"""
    print(f"Commanding arm to perform action: {action}")
    # Test
    return f"Arm command completed: {action}"
    # TODO: Implement arm command from Walkie SDK