import math
from langchain_core.tools import tool
from walkie_sdk import WalkieRobot
import os

from dotenv import load_dotenv

load_dotenv()

EARLY_STOP_DISTANCE = 1.5 # meters

def create_actuators_agent_tools(robot: WalkieRobot):

    def _get_current_pose():
        """Get the current pose of the robot"""
        pose = robot.status.get_pose()
        if pose is None:
            raise ValueError("Unable to get robot pose at the moment")
        return pose

    @tool(parse_docstring=True)
    def move_absolute(x: float, y: float, heading: float = 0.0, early_stop: bool = False) -> str:
        """Move the robot to a specific (x, y) position on the map, with optional heading. Use when the goal is given in map coordinates.

        Units: x, y in meters; heading in degrees (0 = forward/east, 90 = left/north).

        Args:
            x: Target x coordinate in meters (map frame).
            y: Target y coordinate in meters (map frame).
            heading: Target heading in degrees (default: keep current).
            early_stop: If True, the robot will stop moving before reaching the goal. (Useful when wanting to see an object closer/person closer instead of hitting it. e.g. when wanting to navigate towards a person.)
        Returns:
            str: Result of the navigation (success or error).
        """
        print(f"Moving robot absolutely to x: {x}, y: {y}, heading: {heading}")
        heading_rad = math.radians(heading)
        # If early_stop is True, we will stop 1.5 meters before reaching the goal.
        if early_stop:
            x = x - EARLY_STOP_DISTANCE * math.cos(heading_rad)
            y = y - EARLY_STOP_DISTANCE * math.sin(heading_rad)
        result = robot.nav.go_to(x=x, y=y, heading=heading_rad, blocking=True)
        return f"Robot moved successfully"

    @tool(parse_docstring=True)
    def move_relative(x: float, y: float, heading: float = 0.0) -> str:
        """Move the robot relative to its current pose. Use for "go forward N meters", "turn left 90 degrees", etc.

        In the robot's local frame: +x = forward, +y = left. Units: meters for x, y; degrees for heading (positive = counterclockwise).

        Args:
            x: Distance forward in meters (negative = backward).
            y: Distance left in meters (negative = right).
            heading: Change in heading in degrees (positive = turn left).

        Returns:
            str: Result of the movement (success or error).
        """
        print(f"Moving robot relatively to x: {x}, y: {y}, heading: {heading}")
        pose = _get_current_pose()
        x_cur = pose['x']
        y_cur = pose['y']
        heading_cur_rad = pose['heading']
        print(f"Current pose: x: {x_cur}, y: {y_cur}, heading: {heading_cur_rad}")
        heading_rad = math.radians(heading)
        # Transform from robot's local frame to global frame using 2D rotation
        x_global = x_cur + x * math.cos(heading_cur_rad) - y * math.sin(heading_cur_rad)
        y_global = y_cur + x * math.sin(heading_cur_rad) + y * math.cos(heading_cur_rad)
        print(f"Moving to global coordinates: x: {x_global}, y: {y_global}, heading: {heading_cur_rad + heading_rad}")
        result = robot.nav.go_to(x=x_global, y=y_global, heading=heading_cur_rad + heading_rad, blocking=True)
        return f"Robot moved successfully"

    @tool
    def get_current_pose() -> str:
        """Get the robot's current pose (x, y in meters, heading in degrees). Use before planning a relative move or to confirm position after a move."""
        print("Getting current pose of the robot")
        pose = _get_current_pose()
        # Copied from Walkie SDK
        return f"Current pose of the robot: x={pose['x']:+6.2f}  y={pose['y']:+6.2f}  θ={math.degrees(pose['heading']):+5.2f}"

    @tool
    def command_arm(action: str) -> str:
        """Command the robotic arm to perform an action. Use for gestures (e.g. wave, point) or manipulation (e.g. pick up, place). Be specific: "wave hello", "point left", "pick up the cup"."""
        print(f"Commanding arm to perform action: {action}")
        # Test
        return f"Arm command completed: {action}"
        # TODO: Implement arm command from Walkie SDK

    return move_absolute, move_relative, get_current_pose, command_arm