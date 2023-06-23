import os
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
            text = ''
            text += f"[orange1]Array {name}[/orange1] {x.shape} {x.dtype} ∈ [{x.min()}, {x.max()}]"
            if verbose >= 1:
                text += f" μ = {x.mean()} σ = {x.std()}"
            # detect abnormal values
            if np.isnan(x).any():
                text += '[red] NaN![/red]'
            if np.isinf(x).any():
                text += '[red] Inf![/red]'
            console.print(text)
            
            # show values if shape is small or verbose is high
            if x.size < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)
        
        elif torch.is_tensor(x):
            # general stats
            text = ''
            text += f"[orange1]Tensor {name}[/orange1] {x.shape} {x.dtype} {x.device} ∈ [{x.min().item()}, {x.max().item()}]"
            if verbose >= 1:
                text += f" μ = {x.mean().item()} σ = {x.std().item()}"
            # detect abnormal values
            if torch.isnan(x).any():
                text += '[red] NaN![/red]'
            if torch.isinf(x).any():
                text += '[red] Inf![/red]'
            console.print(text)
            
            # show values if shape is small or verbose is high
            if x.numel() < 50 or verbose >= 2:
                # np.set_printoptions(precision=4)
                print(x)
            
        else: # other type, just print them
            console.print(f'[orange1]{type(x)} {name}[/orange1] {x}')

    # inspect names
    for i, x in enumerate(xs):
        _lo(x, varname.argname(f'xs[{i}]'))


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, x):
    with open(path, 'w') as f:
        json.dump(x, f, indent=2)
    

def read_image(path, mode='float'):

    if mode == 'pil':
        return Image.open(path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # cvtColor
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # mode
    if 'float' in mode:
        return img.astype(np.float32) / 255
    elif 'tensor' in mode:
        return torch.from_numpy(img.astype(np.float32) / 255)
    else: # uint8
        return img


def write_image(path, img):

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # cvtColor
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
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
        model_dir = os.path.join(hub_dir, 'checkpoints')

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