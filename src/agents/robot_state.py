"""Robot state provider for dynamic prompt injection.

Reads current pose and optional status from the Walkie robot SDK and formats
it for inclusion in agent system prompts.  Optionally includes the latest
visible-objects snapshot from the background object detector.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from walkie_sdk import WalkieRobot

if TYPE_CHECKING:
    from src.vision.background_detector import BackgroundObjectDetector


class RobotState:
    """Provides current robot state for injection into agent prompts.

    Reads pose from the shared WalkieRobot instance and, when a
    ``BackgroundObjectDetector`` is attached, includes the latest
    visible-objects list so the agent knows what is in front of it
    without calling a tool.
    """

    def __init__(
        self,
        robot: WalkieRobot,
        *,
        vision_enabled: bool = True,
        background_detector: BackgroundObjectDetector | None = None,
    ) -> None:
        """Initialize the robot state provider.

        Args:
            robot: WalkieRobot instance for reading pose.
            vision_enabled: Whether vision/camera is currently enabled.
            background_detector: Optional background detector whose
                ``visible_objects`` will be included in the prompt.
        """
        self._robot = robot
        self.vision_enabled = vision_enabled
        self._background_detector = background_detector

    def get_pose(self) -> dict[str, float] | None:
        """Get current pose from the robot SDK.

        Returns:
            Dict with keys x, y, heading (radians), or None if unavailable.
        """
        try:
            return self._robot.status.get_pose()
        except Exception:
            return None

    def _format_visible_objects(self) -> list[str]:
        """Format the latest visible objects for the prompt.

        Returns:
            Lines to append to the prompt (may be empty if no detector).
        """
        if self._background_detector is None:
            return []

        objects = self._background_detector.visible_objects
        lines: list[str] = ["## Visible Objects (auto-detected)"]

        if not objects:
            lines.append("No objects detected in current view.")
            return lines

        for idx, obj in enumerate(objects, 1):
            name = obj.get("class_name", "unknown")
            conf = obj.get("confidence", 0.0)
            pos = obj.get("position")
            if pos is not None:
                lines.append(
                    f"{idx}. {name} (confidence: {conf:.2f}) "
                    f"at position (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f})"
                )
            else:
                lines.append(f"{idx}. {name} (confidence: {conf:.2f}) -- position unavailable")

        return lines

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

        # Append visible objects from background detector
        visible_lines = self._format_visible_objects()
        if visible_lines:
            lines.append("")  # blank line separator
            lines.extend(visible_lines)

        return "\n".join(lines)
