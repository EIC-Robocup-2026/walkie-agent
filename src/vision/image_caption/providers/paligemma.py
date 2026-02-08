"""PaliGemma image captioning provider (local inference via HuggingFace Transformers)."""

import io
from typing import Any, Union

import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from ..base import ImageCaptionProvider


class PaliGemmaImageCaptionProvider(ImageCaptionProvider):
    """PaliGemma image captioning provider using HuggingFace Transformers (local)."""

    DEFAULT_MODEL = "google/paligemma-3b-ft-cococap-448"
    DEFAULT_PROMPT = "caption en\n"
    DEFAULT_MAX_NEW_TOKENS = 100

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize PaliGemma image captioning provider.

        Args:
            config: Provider configuration with optional keys:
                - model: HuggingFace model id (default: "google/paligemma-3b-ft-cococap-448")
                - device: "cuda" or "cpu" (default: "cuda" if available)
                - default_prompt: Task prompt for PaliGemma (default: "caption en\\n")
                - max_new_tokens: Max tokens to generate (default: 100)
        """
        self.model_name = config.get("model", self.DEFAULT_MODEL)
        device_str = config.get("device", "cuda")
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.default_prompt = config.get("default_prompt", self.DEFAULT_PROMPT)
        self.max_new_tokens = config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)

        self.processor = PaliGemmaProcessor.from_pretrained(self.model_name)
        self.model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )

    def _to_pil(self, image: Union[bytes, Image.Image]) -> Image.Image:
        """Convert image to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(io.BytesIO(image)).convert("RGB")

    def caption(
        self,
        image: Union[bytes, Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Generate a caption or description for an image.

        Args:
            image: The image to caption, either as bytes or PIL Image.
            prompt: Optional task prompt (e.g. "caption en\\n"). If None, uses default_prompt.

        Returns:
            The generated caption/description as a string.
        """
        if prompt is None:
            prompt = self.default_prompt

        pil_image = self._to_pil(image)
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt) :].strip()

    def caption_batch(
        self,
        images: list[Union[bytes, Image.Image]],
        prompt: str | None = None,
    ) -> list[str]:
        """Generate captions for multiple images in a single batched forward pass.
        
        Args:
            images: List of images to caption (bytes or PIL Image).
            prompt: Optional task prompt; if None, uses default_prompt.
            
        Returns:
            List of caption strings, one per image, in the same order as images.
        """
        if not images:
            return []

        if prompt is None:
            prompt = self.default_prompt

        pil_images = [self._to_pil(img) for img in images]
        prompts = [prompt] * len(pil_images)

        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded_outputs = self.processor.batch_decode(
            output, skip_special_tokens=True
        )
        return [d[len(prompt) :].strip() for d in decoded_outputs]

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats."""
        return ["jpeg", "png"]

    def get_default_prompt(self) -> str:
        """Get the default prompt used when none is provided."""
        return self.default_prompt
