"""Camera input for capturing images and video frames."""

import cv2
import numpy as np
from PIL import Image


def list_cameras(max_check: int = 10) -> list[dict]:
    """List available camera devices.
    
    Args:
        max_check: Maximum number of device indices to check.
        
    Returns:
        List of camera info dicts with id, name, width, height, and fps.
    """
    cameras = []
    
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cameras.append({
                "id": i,
                "name": f"Camera {i}",
                "width": width,
                "height": height,
                "fps": fps,
            })
            cap.release()
    
    return cameras


def print_cameras(max_check: int = 10) -> None:
    """Print available cameras in a readable format.
    
    Args:
        max_check: Maximum number of device indices to check.
    """
    cameras = list_cameras(max_check)
    print("Available cameras:")
    print("-" * 60)
    for cam in cameras:
        print(f"  [{cam['id']}] {cam['name']}")
        print(f"      Resolution: {cam['width']}x{cam['height']}, FPS: {cam['fps']}")
    print("-" * 60)


class Camera:
    """Camera interface for capturing images and video frames.
    
    Provides methods for capturing single frames, continuous streaming,
    and returning images in various formats (numpy array, PIL Image, bytes).
    """

    def __init__(
        self,
        device: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        """Initialize camera.
        
        Args:
            device: Camera device index. Use list_cameras() to see options.
            width: Desired frame width. If None, uses camera default.
            height: Desired frame height. If None, uses camera default.
            fps: Desired frames per second. If None, uses camera default.
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        
        self._cap: cv2.VideoCapture | None = None
        self._is_open = False

    def open(self) -> None:
        """Open the camera device.
        
        Raises:
            RuntimeError: If camera cannot be opened.
        """
        if self._is_open:
            return
        
        self._cap = cv2.VideoCapture(self.device)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.device}")
        
        # Set resolution if specified
        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Update actual values
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        self._is_open = True

    def close(self) -> None:
        """Close the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def is_open(self) -> bool:
        """Check if camera is currently open.
        
        Returns:
            True if camera is open.
        """
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def capture(self) -> np.ndarray:
        """Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array in BGR format (OpenCV default).
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        if not self.is_open():
            raise RuntimeError("Camera is not open. Call open() first.")
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        
        return frame

    def capture_rgb(self) -> np.ndarray:
        """Capture a single frame in RGB format.
        
        Returns:
            Frame as numpy array in RGB format.
            
        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
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
        """Destructor - ensures camera is released."""
        self.close()
