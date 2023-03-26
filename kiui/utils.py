import torch
import numpy as np
import cv2
import json

from rich.console import Console
from rich.text import Text

# inspect array like object x and report stats
def lo(x, verbose=0, prefix=''):
    console = Console()

    if isinstance(x, np.ndarray):
        # general stats
        text = Text(prefix)
        text.append(f"Array {x.shape} {x.dtype} ∈ [{x.min()}, {x.max()}]")
        if verbose >= 1:
            text.append(f" μ = {x.mean()} σ = {x.std()}")
        # detect abnormal values
        if np.isnan(x).any():
            text.append(' NaN!', style='red')
        if np.isinf(x).any():
            text.append(' Inf!', style='red')
        console.print(text)
        
        # show values if shape is small or verbose is high
        if x.size < 50 or verbose >= 2:
            # np.set_printoptions(precision=4)
            print(x)
    
    elif torch.is_tensor(x):
        # general stats
        text = Text(prefix)
        text.append(f"Tensor {x.shape} {x.dtype} {x.device} ∈ [{x.min().item()}, {x.max().item()}]")
        if verbose >= 1:
            text.append(f" μ = {x.mean().item()} σ = {x.std().item()}")
        # detect abnormal values
        if torch.isnan(x).any():
            text.append(' NaN!', style='red')
        if torch.isinf(x).any():
            text.append(' Inf!', style='red')
        console.print(text)
        
        # show values if shape is small or verbose is high
        if x.numel() < 50 or verbose >= 2:
            # np.set_printoptions(precision=4)
            print(x)
        
    else: # other type, just print them
        print(x)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, x):
    with open(path, 'w') as f:
        json.dump(x, f, indent=2)
    

def read_image(path, as_float=False):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if as_float:
        img = img.astype(np.float32) / 255

    return img


def write_image(path, img):

    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, img)