"""Vision module for camera input and image processing.

Usage:
    from src.vision import WalkieCamera
    
    # Simple initialization
    camera = WalkieCamera()
    camera.open()
    
    # Capture a frame
    frame = camera.capture_rgb()
    
    # Or use as context manager
    with WalkieCamera() as camera:
        pil_image = camera.capture_pil()
    
    # List available cameras
    from src.vision import list_cameras, print_cameras
    
    # Image captioning
    from src.vision import ImageCaption
    
    captioner = ImageCaption(provider="google", model="gemini-2.5-flash")
    text = captioner.caption(image_bytes, prompt="What is in this image?")
"""

from .camera import WalkieCamera, list_cameras, print_cameras
from .image_caption import ImageCaption, ImageCaptionProvider

__all__ = [
    # Camera
    "WalkieCamera",
    "list_cameras",
    "print_cameras",
    # Image Captioning
    "ImageCaption",
    "ImageCaptionProvider",
]
