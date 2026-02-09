import math
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from walkie_sdk import WalkieRobot

from src.utils.teleop_handler import VRTeleopHandler

load_dotenv()

EARLY_STOP_DISTANCE = 1.5  # meters


def create_actuators_agent_tools(robot: WalkieRobot):
    teleop_handler = VRTeleopHandler(robot)

    def _get_current_pose():
        """Get the current pose of the robot"""
        pose = robot.status.get_pose()
        if pose is None:
            raise ValueError("Unable to get robot pose at the moment")
        return pose

    @tool(parse_docstring=True)
    def move_absolute(
        x: float, y: float, heading: float = 0.0, early_stop: bool = False
    ) -> str:
        """Move the robot to a specific (x, y) position on the map..."""
        print(f"Moving robot absolutely to x: {x}, y: {y}, heading: {heading}")
        heading_rad = math.radians(heading)
        if early_stop:
            x = x - EARLY_STOP_DISTANCE * math.cos(heading_rad)
            y = y - EARLY_STOP_DISTANCE * math.sin(heading_rad)
        robot.nav.go_to(x=x, y=y, heading=heading_rad, blocking=True)
        return "Robot moved successfully"

    @tool(parse_docstring=True)
    def move_relative(x: float, y: float, heading: float = 0.0) -> str:
        """Move the robot relative to its current pose..."""
        print(f"Moving robot relatively to x: {x}, y: {y}, heading: {heading}")
        pose = _get_current_pose()
        x_cur, y_cur, heading_cur_rad = pose["x"], pose["y"], pose["heading"]
        heading_rad = math.radians(heading)
        x_global = x_cur + x * math.cos(heading_cur_rad) - y * math.sin(heading_cur_rad)
        y_global = y_cur + x * math.sin(heading_cur_rad) + y * math.cos(heading_cur_rad)
        robot.nav.go_to(
            x=x_global, y=y_global, heading=heading_cur_rad + heading_rad, blocking=True
        )
        return "Robot moved successfully"

    @tool
    def get_current_pose() -> str:
        """Get the robot's current pose..."""
        pose = _get_current_pose()
        return f"Current pose of the robot: x={pose['x']:+6.2f}  y={pose['y']:+6.2f}  θ={math.degrees(pose['heading']):+5.2f}"

    @tool
    def command_arm(action: str) -> str:
        """Command the robotic arm to perform an action..."""
        return f"Arm command completed: {action}"

    @tool
    def start_vr_teleop(group_name: str = "left_arm") -> str:
        """Start the VR Teleoperation mode for real-time human control."""
        print(f"Starting VR Teleop for {group_name}")
        robot.arm.default_mode = "custom_ik"
        return teleop_handler.start()

    @tool
    def stop_vr_teleop() -> str:
        """Stop VR Teleop and return to autonomous MoveIt mode."""
        print("Stopping VR Teleop")
        robot.arm.default_mode = "moveit"
        return teleop_handler.stop()

    # ย้าย return มาไว้ล่างสุดที่เดียว และใส่ Tool ให้ครบ
    return (
        move_absolute,
        move_relative,
        get_current_pose,
        command_arm,
        start_vr_teleop,
        stop_vr_teleop,
    )
