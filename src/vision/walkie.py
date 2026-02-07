"""WalkieVision - Unified vision interface for camera, captioning, embedding, and object detection.

Usage:
    from src.vision import WalkieVision

    vision = WalkieVision(camera_device=0)
    vision.open()
    desc = vision.describe()
    objects = vision.detect_objects()
    vision.close()

    # Or with context manager
    with WalkieVision() as vision:
        room = vision.classify_scene(["kitchen", "living room", "bedroom"])
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .camera import Camera
from .embedding import Embedding
from .image_caption import ImageCaption
from .object_detection import ObjectDetection
from .object_detection.base import DetectedObject


class WalkieVision:
    """Unified vision interface combining camera, captioning, embedding, and object detection."""

    def __init__(
        self,
        camera_device: int = 0,
        caption_provider: str = "google",
        embedding_provider: str = "clip",
        detection_provider: str = "sam",
        caption_config: dict[str, Any] | None = None,
        embedding_config: dict[str, Any] | None = None,
        detection_config: dict[str, Any] | None = None,
        camera_width: int | None = None,
        camera_height: int | None = None,
        camera_fps: float | None = None,
    ) -> None:
        """Initialize WalkieVision with camera and providers.

        Args:
            camera_device: Camera device index.
            caption_provider: Image captioning provider (e.g., "google").
            embedding_provider: Embedding provider (e.g., "clip").
            detection_provider: Object detection provider (e.g., "sam").
            caption_config: Config for ImageCaption.
            embedding_config: Config for Embedding.
            detection_config: Config for ObjectDetection.
            camera_width: Optional camera width.
            camera_height: Optional camera height.
            camera_fps: Optional camera FPS.
        """
        self._camera = Camera(
            device=camera_device,
            width=camera_width,
            height=camera_height,
            fps=int(camera_fps) if camera_fps is not None else None,
        )

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

    def open(self) -> None:
        """Open the camera."""
        self._camera.open()

    def close(self) -> None:
        """Release the camera."""
        self._camera.close()

    def is_open(self) -> bool:
        """Return True if the camera is open."""
        return self._camera.is_open()

    def __enter__(self) -> WalkieVision:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def capture(self) -> Image.Image:
        """Capture a single frame as PIL Image (RGB)."""
        return self._camera.capture_pil()

    def describe(self, prompt: str | None = None) -> str:
        """Capture current view and return a text description (caption)."""
        image = self.capture()
        return self._caption.caption(image, prompt=prompt)

    def detect_objects(self) -> list[DetectedObject]:
        """Capture current view and detect/segment objects."""
        image = self.capture()
        return self._detection.detect(image)

    def detect_and_embed_objects(
        self,
    ) -> list[dict[str, Any]]:
        """Capture, detect objects, and compute CLIP embedding for each crop.

        Returns:
            List of dicts with keys: object_index, bbox, area_ratio, cropped_image,
            embedding, and optionally caption.
        """
        image = self.capture()
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
        categories: list[str],
    ) -> tuple[str, float]:
        """Classify current view into one of the given categories (e.g. room types).

        Uses CLIP text-image similarity. Categories should be short labels
        (e.g. ["kitchen", "living room", "bedroom"]).

        Returns:
            (best_category_name, confidence in [0, 1]).
        """
        image = self.capture()
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

    @staticmethod
    def list_cameras(max_check: int = 10) -> list[dict]:
        """List available camera devices."""
        from .camera import list_cameras
        return list_cameras(max_check=max_check)

    @staticmethod
    def print_cameras(max_check: int = 10) -> None:
        """Print available cameras."""
        from .camera import print_cameras
        print_cameras(max_check=max_check)
