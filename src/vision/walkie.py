"""WalkieVision - Unified vision interface for camera, captioning, embedding, and object detection.

Usage:
    from src.vision import WalkieVision

    vision = WalkieVision(camera_device=0)
    vision.open()
    image = vision.capture()
    desc = vision.describe(image)
    objects = vision.detect_objects(image)
    vision.close()

    # Or with context manager
    with WalkieVision() as vision:
        image = vision.capture()
        room = vision.classify_scene(image, ["kitchen", "living room", "bedroom"])
"""

from __future__ import annotations

from typing import Any

from PIL import Image
from walkie_sdk import WalkieRobot

from .camera import Camera
from .embedding import Embedding
from .image_caption import ImageCaption
from .object_detection import ObjectDetection
from .object_detection.base import DetectedObject


class WalkieVision:
    """Unified vision interface combining camera, captioning, embedding, and object detection."""

    def __init__(
        self,
        robot: WalkieRobot | None = None,
        camera_device: int | None = None,
        caption_provider: str = "paligemma",
        embedding_provider: str = "clip",
        detection_provider: str = "sam",
        caption_config: dict[str, Any] | None = None,
        embedding_config: dict[str, Any] | None = None,
        detection_config: dict[str, Any] | None = None,
        preload: bool = False,
    ) -> None:
        """Initialize WalkieVision with camera and providers.

        Provide *robot* for robot camera access **or** *camera_device* for a
        local webcam (useful for testing without a robot).  If neither is
        given the camera will be ``None`` and ``capture()`` will raise.

        Args:
            robot: WalkieRobot instance for camera access.
            camera_device: Local webcam device index (e.g. 0). Mutually
                exclusive with *robot*.
            caption_provider: Image captioning provider (e.g., "google").
            embedding_provider: Embedding provider (e.g., "clip").
            detection_provider: Object detection provider (e.g., "sam").
            caption_config: Config for ImageCaption.
            embedding_config: Config for Embedding.
            detection_config: Config for ObjectDetection.
            preload: If True, eagerly load all provider model weights into
                memory during initialization. This avoids cold-start latency
                on the first inference call.
        """
        if robot is not None:
            self._camera = Camera(robot=robot)
        elif camera_device is not None:
            self._camera = Camera(device=camera_device)
            self._camera.open()
        else:
            self._camera = None

        self._caption = ImageCaption(
            provider=caption_provider,
            **(caption_config or {}),
        )
        self._embedding = Embedding(
            provider=embedding_provider,
            **(embedding_config or {}),
        )
        self._detection = ObjectDetection(
            provider=detection_provider,
            **(detection_config or {}),
        )

        if preload:
            self.preload_models()

    @property
    def camera(self) -> Camera:
        """Get the camera instance."""
        return self._camera

    @property
    def caption(self) -> ImageCaption:
        """Get the ImageCaption instance."""
        return self._caption

    @property
    def embedding(self) -> Embedding:
        """Get the Embedding instance."""
        return self._embedding

    @property
    def detection(self) -> ObjectDetection:
        """Get the ObjectDetection instance."""
        return self._detection

    def preload_models(self) -> None:
        """Eagerly load all provider model weights into memory.

        Call this to avoid cold-start latency on the first inference call.
        Providers that are already loaded or API-based (no local model) will
        no-op safely.
        """
        self._caption.load_model()
        self._embedding.load_model()
        self._detection.load_model()

    def __enter__(self) -> WalkieVision:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def open(self) -> None:
        """Open resources (camera, etc.).

        Called automatically when used as a context manager.
        """
        if self._camera is not None:
            self._camera.open()

    def close(self) -> None:
        """Release resources (camera, etc.).

        Called automatically when used as a context manager.
        """
        if self._camera is not None:
            self._camera.close()

    def capture(self) -> Image.Image:
        """Capture a single frame as PIL Image (RGB)."""
        if self._camera is None:
            raise RuntimeError(
                "Camera is not initialized. Provide a 'robot' or 'camera_device' argument."
            )
        return self._camera.capture_pil()

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        """Return a text description (caption) of the given image.

        Args:
            image: PIL Image to describe.
            prompt: Optional prompt to guide the captioning.

        Returns:
            Caption string.
        """
        return self._caption.caption(image, prompt=prompt)
    
    def caption_batch(self, images: list[Image.Image], prompts: list[str] | None = None) -> list[str]:
        """Return text descriptions (captions) for a batch of images.

        Args:
            images: List of PIL Images to describe.
            prompts: Optional list of prompts to guide the captioning.
        Returns:
            List of caption strings.
        """
        return self._caption.caption_batch(images, prompts=prompts)

    def detect_objects(self, image: Image.Image) -> list[DetectedObject]:
        """Detect/segment objects in the given image.

        Args:
            image: PIL Image to run detection on.

        Returns:
            List of DetectedObject instances.
        """
        return self._detection.detect(image)

    def detect_and_embed_objects(
        self,
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        """Detect objects in the given image and compute CLIP embedding for each crop.

        Args:
            image: PIL Image to run detection and embedding on.

        Returns:
            List of dicts with keys: object_index, bbox, area_ratio, cropped_image,
            embedding, and optionally caption.
        """
        detected = self._detection.detect(image)
        results: list[dict[str, Any]] = []
        for i, obj in enumerate(detected):
            emb = self._embedding.embed_image(obj.cropped_image)
            item: dict[str, Any] = {
                "object_index": i,
                "bbox": obj.bbox,
                "area_ratio": obj.area_ratio,
                "cropped_image": obj.cropped_image,
                "embedding": emb,
            }
            if obj.class_id is not None:
                item["class_id"] = obj.class_id
            if obj.class_name is not None:
                item["class_name"] = obj.class_name
            if obj.confidence is not None:
                item["confidence"] = obj.confidence
            results.append(item)
        return results

    def embed_image(self, image: Image.Image) -> list[float]:
        """Compute embedding for an image."""
        return self._embedding.embed_image(image)

    def embed_text(self, text: str) -> list[float]:
        """Compute embedding for a text."""
        return self._embedding.embed_text(text)

    def classify_scene(
        self,
        image: Image.Image,
        categories: list[str],
    ) -> tuple[str, float]:
        """Classify the given image into one of the given categories (e.g. room types).

        Uses CLIP text-image similarity. Categories should be short labels
        (e.g. ["kitchen", "living room", "bedroom"]).

        Args:
            image: PIL Image to classify.
            categories: List of category labels.

        Returns:
            (best_category_name, confidence in [0, 1]).
        """
        img_emb = self._embedding.embed_image(image)
        text_embs = self._embedding.embed_texts(categories)
        best_idx = 0
        best_sim = -1.0
        for idx, text_emb in enumerate(text_embs):
            sim = self._embedding.similarity(img_emb, text_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        # CLIP logits are often in a range that softmax normalizes; for "confidence"
        # we use a simple linear scaling from [-1,1] to [0,1]: (sim + 1) / 2
        confidence = (best_sim + 1.0) / 2.0
        return categories[best_idx], confidence
