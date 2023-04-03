import torch
import numpy as np
import cv2
import json
import varname
from PIL import Image

from rich.console import Console
from rich.text import Text

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