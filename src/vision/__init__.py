"""Vision module for camera input and image processing.

Usage:
    from src.vision import WalkieCamera

    # Simple initialization
    camera = WalkieCamera()
    camera.open()
    frame = camera.capture_rgb()

    # Unified vision (camera + caption + embedding + object detection)
    from src.vision import WalkieVision
    with WalkieVision() as vision:
        desc = vision.describe()
        room, conf = vision.classify_scene(["kitchen", "living room"])

    # Image captioning
    from src.vision import ImageCaption
    captioner = ImageCaption(provider="google", model="gemini-2.5-flash")
    text = captioner.caption(image_bytes, prompt="What is in this image?")
"""

from .camera import Camera, list_cameras, print_cameras
from .embedding import Embedding, EmbeddingProvider
from .image_caption import ImageCaption, ImageCaptionProvider
from .object_detection import DetectedObject, ObjectDetection, ObjectDetectionProvider
from .walkie import WalkieVision

__all__ = [
    # Camera
    "Camera",
    "list_cameras",
    "print_cameras",
    # Image Captioning
    "ImageCaption",
    "ImageCaptionProvider",
    # Embedding
    "Embedding",
    "EmbeddingProvider",
    # Object Detection
    "DetectedObject",
    "ObjectDetection",
    "ObjectDetectionProvider",
    # Unified
    "WalkieVision",
]
