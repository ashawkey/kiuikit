import base64
import io
from PIL import Image
from typing import Literal

def get_text_content_dict(text: str) -> dict:
    """Get the text content dict for a message."""
    return {
        "type": "text",
        "text": text,
    }


def get_image_content_dict(image_path: str, detail: Literal["low", "high"] = "low") -> dict:
    """Get the image content dict for a message."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{load_image_as_jpeg_base64(image_path)}",
            "detail": detail,
        },
    }


def load_image_as_jpeg_base64(input_path, resolution=512, quality=85):
    img = Image.open(input_path)
    # resize the longer side to resolution and keep the aspect ratio
    width, height = img.size
    if width > height:
        width = resolution
        height = int(height * resolution / width)
    else:
        height = resolution
        width = int(width * resolution / height)
    # resize the image
    # print(f"[INFO] resizing image to {width}x{height}")
    img = img.resize((width, height), Image.LANCZOS)
    # use a buffer to store the image as jpeg
    buffer = io.BytesIO()
    img = img.convert('RGB')
    img.save(buffer, format="JPEG", quality=quality)
    jpeg_bytes = buffer.getvalue()
    buffer.seek(0)
    # base64 encode the jpeg bytes
    base64_jpeg_bytes = base64.b64encode(jpeg_bytes).decode("utf-8")
    return base64_jpeg_bytes


def parse_tool_docstring(docstring: str) -> str:
    """Parse a formatted docstring of a tool function.
    The function should be strictly formatted as follows:
    ```
    def function_name(param1: type1, param2: type2, ...):
        \"\"\" <one-sentence description>
        <more detailed description and usage>
        Args: # if no args, still keep this line and leave a None placeholder
            <param1>: <param1_description>
            <param2>: <param2_description>
            ...
        Examples: # optional
            <example_1>
            ...
        \"\"\"
        <function_implementation>
    ```
    Returns:
        description: the description of the tool function
        parameters: a dictionary of the parameters of the tool function
    """
    description, args_examples = docstring.strip().split("Args:")
    if "Examples:" not in args_examples:
        args = args_examples
        examples = None
    else:
        args, examples = args_examples.strip().split("Examples:")
    description = description.strip()

    parameters: dict[str, str] = {}
    args = args.split("\n")
    for arg in args:
        # split at the first colon
        if ":" not in arg:
            continue
        param_name, param_description = arg.strip().split(":", 1)
        parameters[param_name] = param_description
    
    return description, parameters