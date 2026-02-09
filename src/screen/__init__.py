"""Screen display module for showing images and text on a fullscreen window.

Usage:
    from src.screen import WalkieScreen

    # Open a fullscreen display
    screen = WalkieScreen()

    # Show text on a coloured background
    screen.show_text("Hello, World!", font_size=64, background_color=(0, 120, 255))

    # Show an image (file path, PIL Image, or numpy array)
    screen.show_image("photo.jpg", background_color=(30, 30, 30))

    # Clear to black
    screen.clear()

    # Close when done
    screen.close()
"""

from .walkie import WalkieScreen

__all__ = [
    "WalkieScreen",
]
