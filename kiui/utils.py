import os
import glob
import tqdm
import cv2
import json
import varname
from PIL import Image

import torch
import numpy as np

from rich.console import Console

# inspect array like object x and report stats
def lo(*xs, verbose=0):

    console = Console()

    def _lo(x, name):

        if isinstance(x, np.ndarray):
            # general stats
            text = ""
            text += f"[orange1]Array {name}[/orange1] {x.shape} {x.dtype} ∈ [{x.min()}, {x.max()}]"
            if verbose >= 1:
                text += f" μ = {x.mean()} σ = {x.std()}"
            # detect abnormal values
            if np.isnan(x).any():
                text += "[red] NaN![/red]"
            if np.isinf(x).any():
                text += "[red] Inf![/red]"
            console.print(text)

            # show values if shape is small or verbose is high
            if x.size < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)

        elif torch.is_tensor(x):
            # general stats
            text = ""
            text += f"[orange1]Tensor {name}[/orange1] {x.shape} {x.dtype} {x.device} ∈ [{x.min().item()}, {x.max().item()}]"
            if verbose >= 1:
                text += f" μ = {x.mean().item()} σ = {x.std().item()}"
            # detect abnormal values
            if torch.isnan(x).any():
                text += "[red] NaN![/red]"
            if torch.isinf(x).any():
                text += "[red] Inf![/red]"
            console.print(text)

            # show values if shape is small or verbose is high
            if x.numel() < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)

        else:  # other type, just print them
            console.print(f"[orange1]{type(x)} {name}[/orange1] {x}")

    # inspect names
    for i, x in enumerate(xs):
        _lo(x, varname.argname(f"xs[{i}]"))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, x):
    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def read_image(path, mode="float", order="RGB"):

    if mode == "pil":
        return Image.open(path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # cvtColor
    if order == "RGB":
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mode
    if "float" in mode:
        return img.astype(np.float32) / 255
    elif "tensor" in mode:
        return torch.from_numpy(img.astype(np.float32) / 255)
    else:  # uint8
        return img


def write_image(path, img, order="RGB"):

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # cvtColor
    if order == "RGB":
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):

    from torch.hub import download_url_to_file, get_dir
    from urllib.parse import urlparse

    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def is_image_file(f):
    return os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]


def batch_process_image(
    process_fn, path, out_path, color_order="RGB", out_format=None, **kwargs
):
    if os.path.isdir(path):
        img_paths = glob.glob(os.path.join(path, "*"))
        img_paths = [f for f in img_paths if is_image_file(f)]
    else:
        img_paths = [path]

    if not is_image_file(out_path):
        os.makedirs(out_path, exist_ok=True)

    for img_path in tqdm.tqdm(img_paths):
        try:
            if not is_image_file(out_path):
                img_out_path = os.path.join(out_path, os.path.basename(img_path))
            else:
                img_out_path = out_path

            if out_format is not None:
                img_out_path = os.path.splitext(img_out_path)[0] + "." + out_format

            img = read_image(img_path, mode="uint8", order=color_order)
            res = process_fn(img, **kwargs)
            write_image(img_out_path, res, order=color_order)

        except Exception as e:
            print(f"[Error] when processing {img_path} --> {img_out_path}")
            print(e)
