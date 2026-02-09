"""Rendering helpers for the screen module.

Produces numpy frames (BGR format, suitable for cv2.imshow) for text,
images, and solid-color backgrounds.  Text rendering uses Pillow for
proper Unicode support, TrueType fonts, and automatic word wrapping.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_solid_frame(
    screen_size: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a solid-color frame.

    Args:
        screen_size: (width, height) of the output frame.
        color: RGB color tuple.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    width, height = screen_size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Convert RGB -> BGR for OpenCV
    frame[:] = color[::-1]
    return frame


def render_text_frame(
    text: str,
    screen_size: tuple[int, int],
    font_size: int = 48,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render centred, word-wrapped text onto a coloured background.

    Uses Pillow for high-quality text rendering, then converts to a BGR
    numpy array for OpenCV.

    Args:
        text: The text to render (newlines are respected).
        screen_size: (width, height) of the output frame.
        font_size: Font size in pixels.
        color: RGB text colour.
        bg_color: RGB background colour.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    width, height = screen_size

    # Create Pillow image in RGB
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    font = _load_font(font_size)

    # Wrap text to fit within screen (with padding)
    padding = int(width * 0.08)
    max_text_width = width - 2 * padding
    wrapped = _wrap_text(draw, text, font, max_text_width)

    # Measure total text block height
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Centre the text block
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.multiline_text(
        (x, y),
        wrapped,
        fill=color,
        font=font,
        align="center",
    )

    # Convert Pillow RGB -> numpy BGR
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def render_image_frame(
    image: str | Path | Image.Image | np.ndarray,
    screen_size: tuple[int, int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render an image centred on a coloured background, scaled to fit.

    The image is scaled to fit within the screen dimensions while
    preserving its aspect ratio, then placed in the centre of the
    background canvas.

    Args:
        image: A file path (str/Path), PIL Image, or numpy array (BGR).
        screen_size: (width, height) of the output frame.
        bg_color: RGB background colour.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    width, height = screen_size
    bgr_img = _to_bgr(image)

    # Scale to fit while preserving aspect ratio
    img_h, img_w = bgr_img.shape[:2]
    scale = min(width / img_w, height / img_h)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=interp)

    # Create background canvas and place the image in the centre
    canvas = render_solid_frame(screen_size, bg_color)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, falling back to the default bitmap font."""
    # Common Linux font paths
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)

    # Last resort: Pillow's built-in default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    """Word-wrap *text* so that no line exceeds *max_width* pixels.

    Existing newlines in the input are preserved.
    """
    paragraphs = text.split("\n")
    wrapped_lines: list[str] = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_lines.append("")
            continue

        words = paragraph.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word

        if current_line:
            wrapped_lines.append(current_line)

    return "\n".join(wrapped_lines)


def _to_bgr(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Convert various image types to a BGR numpy array."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    if isinstance(image, (str, Path)):
        path = str(image)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return frame

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Expected str, Path, PIL.Image, or numpy array."
    )
