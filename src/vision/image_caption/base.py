"""Base class for Image Captioning providers."""

from abc import ABC, abstractmethod
from typing import Union

from PIL import Image


class ImageCaptionProvider(ABC):
    """Abstract base class for image captioning providers.
    
    Implement this class to add a new image captioning provider.
    """

    @abstractmethod
    def caption(
        self,
        image: Union[bytes, Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Generate a caption or description for an image.
        
        Args:
            image: The image to caption, either as bytes or PIL Image.
            prompt: Optional prompt to guide the captioning (e.g., "What is in this image?").
                   If None, uses the provider's default prompt.
            
        Returns:
            The generated caption/description as a string.
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.
        
        Returns:
            List of format identifiers (e.g., ['jpeg', 'png', 'webp', 'gif']).
        """
        pass

    def get_default_prompt(self) -> str:
        """Get the default prompt used when none is provided.
        
        Returns:
            Default prompt string.
        """
        return "Describe this image in detail."
