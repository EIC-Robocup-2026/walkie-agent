"""WalkieScreen - Fullscreen display interface using OpenCV.

Usage:
    from src.screen import WalkieScreen

    screen = WalkieScreen()

    # Show text on a blue background
    screen.show_text("Hello!", background_color=(0, 120, 255))

    # Show an image from a file
    screen.show_image("photo.jpg")

    # Clear to black
    screen.clear()

    # Close when done
    screen.close()
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .renderer import render_image_frame, render_solid_frame, render_text_frame

# Default resolution used when the actual screen size cannot be detected.
_DEFAULT_WIDTH = 1920
_DEFAULT_HEIGHT = 1080


class WalkieScreen:
    """Fullscreen display window driven by OpenCV.

    A dedicated daemon thread runs the display loop, calling
    ``cv2.waitKey()`` to keep the window responsive.  Public methods
    update a shared frame buffer that the display thread picks up on its
    next iteration.
    """

    def __init__(
        self,
        window_name: str = "Walkie",
        fullscreen: bool = True,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialise and open the display window.

        Args:
            window_name: Name of the OpenCV window.
            fullscreen: Whether to open in fullscreen mode.
            screen_size: Explicit (width, height) override.  When *None*
                the size is auto-detected (falls back to 1920x1080).
        """
        self._window_name = window_name
        self._fullscreen = fullscreen

        # Screen dimensions (resolved once the window is created)
        self._screen_size: tuple[int, int] = screen_size or (
            _DEFAULT_WIDTH,
            _DEFAULT_HEIGHT,
        )

        # Shared state between the main thread and the display thread
        self._lock = threading.Lock()
        self._frame: np.ndarray = render_solid_frame(self._screen_size)
        self._running = True

        # Start the display loop in a daemon thread
        self._thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="WalkieScreen",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Display an image centred on a coloured background.

        The image is scaled to fit the screen while preserving its aspect
        ratio.

        Args:
            image: A file path (str / Path), PIL Image, or numpy array.
            background_color: RGB background colour tuple.
        """
        frame = render_image_frame(image, self._screen_size, background_color)
        self._update_frame(frame)

    def show_text(
        self,
        text: str,
        font_size: int = 48,
        color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """Display text centred on a coloured background.

        Long text is automatically word-wrapped to fit the screen.
        Newline characters in *text* are respected.

        Args:
            text: The text to display.
            font_size: Font size in pixels.
            color: RGB text colour.
            background_color: RGB background colour.
        """
        frame = render_text_frame(
            text,
            self._screen_size,
            font_size=font_size,
            color=color,
            bg_color=background_color,
        )
        self._update_frame(frame)

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """Clear the screen to a solid colour.

        Args:
            color: RGB colour to fill the screen with.
        """
        frame = render_solid_frame(self._screen_size, color)
        self._update_frame(frame)

    def close(self) -> None:
        """Close the display window and stop the background thread."""
        self._running = False
        self._thread.join(timeout=2.0)

    @property
    def screen_size(self) -> tuple[int, int]:
        """Return the current screen size as (width, height)."""
        return self._screen_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_frame(self, frame: np.ndarray) -> None:
        """Thread-safe frame buffer update."""
        with self._lock:
            self._frame = frame

    def _display_loop(self) -> None:
        """Background loop that keeps the OpenCV window alive."""
        # Create the window and show an initial frame *before* setting
        # fullscreen – some Linux window managers ignore the property
        # unless a frame has already been presented.
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        with self._lock:
            frame = self._frame
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)

        if self._fullscreen:
            cv2.setWindowProperty(
                self._window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            cv2.waitKey(1)

        while self._running:
            with self._lock:
                frame = self._frame

            cv2.imshow(self._window_name, frame)

            # waitKey keeps the window responsive; 30 ms ≈ ~33 fps cap
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC to close
                self._running = False
                break

        cv2.destroyWindow(self._window_name)
