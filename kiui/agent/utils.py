import base64
import io
from pathlib import Path
from typing import Literal

KIA_DIR_NAME = ".kia"


def get_kia_dir(cwd: str | Path | None = None) -> Path:
    """Return the .kia directory for the given working directory, creating it if needed."""
    base = Path(cwd) if cwd else Path.cwd()
    kia_dir = base / KIA_DIR_NAME
    kia_dir.mkdir(parents=True, exist_ok=True)
    return kia_dir


def get_text_content_dict(text: str) -> dict:
    """Get the text content dict for a message."""
    return {"type": "text", "text": text}


def get_image_content_dict(image_path: str, detail: Literal["low", "high"] = "low") -> dict:
    """Get the image content dict for a message."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{load_image_as_jpeg_base64(image_path)}",
            "detail": detail,
        },
    }


def load_image_as_jpeg_base64(input_path: str, resolution: int = 512, quality: int = 85) -> str:
    """Load an image, resize to fit within *resolution* px, and return as base64 JPEG."""
    from PIL import Image

    img = Image.open(input_path)

    orig_w, orig_h = img.size
    if orig_w > orig_h:
        width = resolution
        height = int(orig_h * resolution / orig_w)
    else:
        height = resolution
        width = int(orig_w * resolution / orig_h)

    img = img.resize((width, height), Image.LANCZOS)

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
