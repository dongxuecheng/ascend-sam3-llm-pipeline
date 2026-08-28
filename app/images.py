"""Validate images before admission; keep the original upload unchanged."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.domain import PreparedImage


class InvalidImage(ValueError):
    pass


class ImageTooLarge(InvalidImage):
    pass


FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}


def prepare_image(data: bytes, max_bytes: int, max_pixels: int) -> PreparedImage:
    if not data:
        raise InvalidImage("Image is empty")
    if len(data) > max_bytes:
        raise ImageTooLarge("Image exceeds MAX_IMAGE_BYTES")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in FORMATS:
                raise InvalidImage("Only JPEG, PNG and single-frame WebP are supported")
            extension, mime = FORMATS[source.format]
            if source.width * source.height > max_pixels:
                raise ImageTooLarge("Image exceeds MAX_IMAGE_PIXELS")
            if getattr(source, "n_frames", 1) != 1:
                raise InvalidImage("Animated or multi-frame images are not supported")
            source.load()  # Reject truncated data before returning 202.
            normalized = source.getexif().get(274, 1) not in (None, 1)
            inference = data
            width, height = source.size
            if normalized:
                # Both models receive the same orientation and coordinates. Preserve
                # uploaded bytes separately, and normalize only the inference image.
                upright = ImageOps.exif_transpose(source).convert("RGB")
                buffer = io.BytesIO()
                upright.save(buffer, format="PNG")
                inference = buffer.getvalue()
                if len(inference) > max_bytes:
                    raise ImageTooLarge("Orientation-normalized image exceeds MAX_IMAGE_BYTES")
                mime = "image/png"
                width, height = upright.size
            return PreparedImage(data, inference, extension, mime, width, height, normalized)
    except Image.DecompressionBombError as exc:
        raise ImageTooLarge("Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, InvalidImage):
            raise
        raise InvalidImage("Image could not be decoded") from exc
