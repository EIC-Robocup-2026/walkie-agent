"""SAM (Segment Anything Model) object detection provider."""

import os
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from ..base import DetectedObject, ObjectDetectionProvider


def _download_file(url: str, filepath: str) -> None:
    """Download file from url to filepath if not present."""
    if os.path.exists(filepath):
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


class SAMObjectDetectionProvider(ObjectDetectionProvider):
    """Object detection via Segment Anything Model (SAM) ViT-B."""

    DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"
    DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize SAM provider.

        Args:
            config: Optional keys:
                - checkpoint_path: Path to .pth file (default: sam_vit_b_01ec64.pth)
                - checkpoint_url: URL to download checkpoint if missing
                - device: "cuda" or "cpu" (default: auto)
                - min_area_ratio: Filter out objects smaller than this (default: 0.005)
                - max_area_ratio: Filter out objects larger than this (default: 0.50)
                - max_objects: Maximum number of objects to return (default: 10)
                - crop_padding: Pixels to add around crop (default: 50)
                - points_per_side: SAM grid (default: 32)
                - pred_iou_thresh: SAM threshold (default: 0.90)
                - stability_score_thresh: SAM threshold (default: 0.95)
                - min_mask_region_area: SAM min region (default: 200)
        """
        checkpoint_path = config.get("checkpoint_path", self.DEFAULT_CHECKPOINT)
        checkpoint_url = config.get("checkpoint_url", self.DEFAULT_URL)
        _download_file(checkpoint_url, checkpoint_path)

        device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._min_area_ratio = float(config.get("min_area_ratio", 0.005))
        self._max_area_ratio = float(config.get("max_area_ratio", 0.50))
        self._max_objects = int(config.get("max_objects", 10))
        self._crop_padding = int(config.get("crop_padding", 50))

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.get("points_per_side", 32),
            pred_iou_thresh=config.get("pred_iou_thresh", 0.90),
            stability_score_thresh=config.get("stability_score_thresh", 0.95),
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=config.get("min_mask_region_area", 200),
        )
        self._model_name = "sam_vit_b"

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Generate masks, filter by area, return cropped objects."""
        img_rgb = np.array(image)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        h, w = img_rgb.shape[0], img_rgb.shape[1]
        total_area = h * w

        masks = self._mask_generator.generate(img_rgb)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        results: list[DetectedObject] = []
        for mask_data in masks:
            if len(results) >= self._max_objects:
                break
            area_ratio = mask_data["area"] / total_area
            if area_ratio < self._min_area_ratio or area_ratio > self._max_area_ratio:
                continue
            seg = mask_data["segmentation"]
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            pad = self._crop_padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_rgb = img_rgb[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop_rgb)
            bbox = (x1, y1, x2, y2)
            results.append(
                DetectedObject(
                    mask=None,  # Could store mask coords if needed
                    bbox=bbox,
                    area_ratio=area_ratio,
                    cropped_image=crop_pil,
                )
            )
        return results

    def get_model_name(self) -> str:
        return self._model_name
