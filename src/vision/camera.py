"""Camera input for capturing images and video frames using Walkie SDK."""

import cv2
import numpy as np
from PIL import Image
from walkie_sdk.robot import WalkieRobot


class Camera:
    """Camera interface for capturing images using Walkie SDK.
    
    Provides methods for capturing single frames and returning images 
    in various formats (numpy array, PIL Image, bytes).
    """

    def __init__(
        self,
        robot: WalkieRobot,
    ) -> None:
        """Initialize camera with Walkie SDK connection.
        
        Args:
            ip: Robot IP address.
            ros_protocol: Protocol for ROS commands ("rosbridge" or "zenoh").
            camera_protocol: Protocol for camera ("zenoh" or other supported).
            ros_port: Port for ROS protocol.
            camera_port: Port for camera protocol.
        """
        self._bot: WalkieRobot = robot

    def capture(self) -> np.ndarray:
        """Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array in BGR format.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        
        frame = self._bot.camera.get_frame()
        if frame is None:
            raise RuntimeError("Failed to get frame from robot camera.")
        
        return frame

    def capture_rgb(self) -> np.ndarray:
        """Capture a single frame in RGB format.
        
        Returns:
            Frame as numpy array in RGB format.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
        # WalkieRobot returns BGR, convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def capture_pil(self) -> Image.Image:
        """Capture a single frame as PIL Image.
        
        Returns:
            Frame as PIL Image in RGB format.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame_rgb = self.capture_rgb()
        return Image.fromarray(frame_rgb)

    def capture_jpeg(self, quality: int = 95) -> bytes:
        """Capture a single frame as JPEG bytes.
        
        Args:
            quality: JPEG quality (0-100).
            
        Returns:
            Frame as JPEG-encoded bytes.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def capture_png(self) -> bytes:
        """Capture a single frame as PNG bytes.
        
        Returns:
            Frame as PNG-encoded bytes.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
        _, buffer = cv2.imencode(".png", frame)
        return buffer.tobytes()

    def __enter__(self) -> "Camera":
        """Context manager entry - opens the camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the camera."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures camera connection is released."""
        self.close()
