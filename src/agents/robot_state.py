"""Robot state provider for dynamic prompt injection.

Reads current pose and optional status from the Walkie robot SDK and formats
it for inclusion in agent system prompts.
"""

from __future__ import annotations

import math
from typing import Any

# Lazy import to avoid circular imports when agents load middleware
def _get_robot():
    from src.agents.actuators_agent.tools import robot
    return robot


class RobotState:
    """Provides current robot state for injection into agent prompts.

    Reads pose from the shared WalkieRobot instance. Placeholder fields
    (vision_enabled, battery_level, arm_status) are for future use when
    the SDK or other sensors expose them.
    """

    def __init__(
        self,
        robot: Any | None = None,
        *,
        vision_enabled: bool = True,
    ) -> None:
        """Initialize the robot state provider.

        Args:
            robot: Optional WalkieRobot instance. If None, the singleton
                from actuators_agent.tools is used.
            vision_enabled: Whether vision/camera is currently enabled (placeholder).
        """
        self._robot = robot
        self.vision_enabled = vision_enabled

    def get_pose(self) -> dict[str, float] | None:
        """Get current pose from the robot SDK.

        Returns:
            Dict with keys x, y, heading (radians), or None if unavailable.
        """
        r = self._robot if self._robot is not None else _get_robot()
        try:
            return r.status.get_pose()
        except Exception:
            return None

    def format_for_prompt(self) -> str:
        """Format current robot state as a text block for system prompts.

        Returns:
            A formatted section suitable for appending to the system message.
        """
        lines = ["## Robot State"]

        pose = self.get_pose()
        if pose is not None:
            x, y = pose.get("x", 0), pose.get("y", 0)
            heading_rad = pose.get("heading", 0)
            heading_deg = math.degrees(heading_rad)
            lines.append(f"- Position: x={x:+.2f} m, y={y:+.2f} m")
            lines.append(f"- Heading: {heading_deg:+.1f} deg")
        else:
            lines.append("- Position: unknown (pose unavailable)")

        lines.append(f"- Vision: {'enabled' if self.vision_enabled else 'disabled'}")

        return "\n".join(lines)
