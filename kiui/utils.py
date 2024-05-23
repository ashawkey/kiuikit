import os
import sys
import glob
import tqdm
import json
import pickle
import varname
from objprint import objstr
from rich.console import Console

import cv2
from PIL import Image

import numpy as np
import torch

from kiui.typing import *
from kiui.env import is_imported

def lo(*xs, verbose=0):
    """inspect array like objects and report statistics.

    Args:
        xs (Any): array like objects to inspect.
        verbose (int, optional): level of verbosity, set to 1 to report mean and std, 2 to print the content. Defaults to 0.
    """

    console = Console()

    def _lo(x, name):

        if isinstance(x, np.ndarray):
            # general stats
            text = ""
            text += f"[orange1]Array {name}[/orange1] {x.shape} {x.dtype}"
            if x.size > 0:
                text += f" ∈ [{x.min()}, {x.max()}]"
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
            text += f"[orange1]Tensor {name}[/orange1] {x.shape} {x.dtype} {x.device}"
            if x.numel() > 0:
                text += f" ∈ [{x.min().item()}, {x.max().item()}]"
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
            console.print(f"[orange1]{type(x)} {name}[/orange1] {objstr(x)}")

    # inspect names
    for i, x in enumerate(xs):
        try:
            name = varname.argname(f"xs[{i}]", func=lo)
        except:
            name = f"UNKNOWN"
        _lo(x, name)


def seed_everything(seed=42, verbose=False, strict=False):
    """auto set seed for random, numpy and torch.

    Args:
        seed (int, optional): random seed. Defaults to 42.
        verbose (bool, optional): whether to report each seed setting. Defaults to False.
        strict (bool, optional): whether to use strict deterministic mode for better torch reproduction. Defaults to False.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)

    if is_imported('random'):
        import random # still need to import it here
        random.seed(seed)
        if verbose: print(f'[INFO] set random.seed = {seed}')
    else:
        if verbose: print(f'[INFO] random not imported, skip setting seed')

    # assume numpy is imported as np
    if is_imported('np'):
        import numpy as np
        np.random.seed(seed)
        if verbose: print(f'[INFO] set np.random.seed = {seed}')
    else:
        if verbose: print(f'[INFO] numpy not imported, skip setting seed')
        
    if is_imported('torch'):
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if verbose: print(f'[INFO] set torch.manual_seed = {seed}')

        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            if verbose: print(f'[INFO] set strict deterministic mode for torch.')
    else:
        if verbose: print(f'[INFO] torch not imported, skip setting seed')


def read_json(path):
    """load a json file.

    Args:
        path (str): path to json file.

    Returns:
        dict: json content.
    """
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, x):
    """write a json file.

    Args:
        path (str): path to write json file.
        x (dict): dict to write.
    """
    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def read_pickle(path):
    """read a pickle file.

    Args:
        path (str): path to pickle file.

    Returns:
        Any: pickle content.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path, x):
    """write a pickle file.

    Args:
        path (str): path to write pickle file.
        x (Any): content to write.
    """
    with open(path, "wb") as f:
        pickle.dump(x, f)


def read_image(
    path: str, 
    mode: Literal["float", "uint8", "pil", "torch", "tensor"] = "float", 
    order: Literal["RGB", "RGBA", "BGR", "BGRA"] = "RGB",
):
    """read an image file into various formats and color mode.

    Args:
        path (str): path to the image file.
        mode (Literal["float", "uint8", "pil", "torch", "tensor"], optional): returned image format. Defaults to "float".
            float: float32 numpy array, range [0, 1];
            uint8: uint8 numpy array, range [0, 255];
            pil: PIL image;
            torch/tensor: float32 torch tensor, range [0, 1];
        order (Literal["RGB", "RGBA", "BGR", "BGRA"], optional): channel order. Defaults to "RGB".
    
    Note:
        By default this function will convert RGBA image to white-background RGB image. Use ``order="RGBA"`` to keep the alpha channel.

    Returns:
        Union[np.ndarray, PIL.Image, torch.Tensor]: the image array.
    """

    if mode == "pil":
        return Image.open(path).convert(order)

    if path.endswith('.exr'):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # cvtColor
    if len(img.shape) == 3: # ignore if gray scale
        if order in ["RGB", "RGBA"]:
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # mix background
        if img.shape[-1] == 4 and 'A' not in order:
            img = img.astype(np.float32) / 255
            img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])

    # mode
    if mode == "uint8":
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        return img
    elif mode == "float":
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        return img
    elif mode in ["tensor", "torch"]:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        return torch.from_numpy(img)
    else:
        raise ValueError(f"Unknown read_image mode {mode}")


def write_image(
        path: str, 
        img: Union[Tensor, np.ndarray, Image.Image], 
        order: Literal["RGB", "BGR"] = "RGB",
    ):
    """write an image to various formats.

    Args:
        path (str): path to write the image file.
        img (Union[torch.Tensor, np.ndarray, PIL.Image.Image]): image to write.
        order (str, optional): channel order. Defaults to "RGB".
    """

    if isinstance(img, Image.Image):
        img.save(path)
        return

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 4:
        if img.shape[0] > 1:
            raise ValueError(f'only support saving a single image! current image: {img.shape}')
        img = img[0]
        
    if len(img.shape) == 3:
        # cvtColor
        if order == "RGB":
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dir_path = os.path.dirname(path)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """

    from torch.hub import download_url_to_file, get_dir
    from urllib.parse import urlparse

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
        print(f'[INFO] Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def is_format(f: str, format: Sequence[str]):
    """if a file's extension is in a set of format

    Args:
        f (str): file name.
        format (Sequence[str]): set of extensions (both '.jpg' or 'jpg' is ok).

    Returns:
        bool: if the file's extension is in the set.
    """
    ext = os.path.splitext(f)[1].lower() # include the dot
    return ext in format or ext[1:] in format

def batch_process_files(
    process_fn, path, out_path, 
    overwrite=False,
    in_format=[".jpg", ".jpeg", ".png"],
    out_format=None,
    image_mode='uint8',
    image_color_order="RGB",
    **kwargs
):
    """simple function wrapper to batch processing files.

    Args:
        process_fn (Callable): process function.
        path (str): path to a file or a directory containing the files to process.
        out_path (str): output path of a file or a directory.
        overwrite (bool, optional): whether to overwrite existing results. Defaults to False.
        in_format (list, optional): input file formats. Defaults to [".jpg", ".jpeg", ".png"].
        out_format (str, optional): output file format. Defaults to None.
        image_mode (str, optional): for images, the mode to read. Defaults to 'uint8'.
        image_color_order (str, optional): for images, the color order. Defaults to "RGB".
    """
   
    if os.path.isdir(path):
        file_paths = glob.glob(os.path.join(path, "*"))
        file_paths = [f for f in file_paths if is_format(f, in_format)]
    else:
        file_paths = [path]

    if os.path.dirname(out_path) != '':
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for file_path in tqdm.tqdm(file_paths):
        try:
            
            if len(file_paths) == 1:
                file_out_path = out_path
            else:
                file_out_path = os.path.join(out_path, os.path.basename(file_path))
            
            if out_format is not None:
                file_out_path = os.path.splitext(file_out_path)[0] + out_format

            if os.path.exists(file_out_path) and not overwrite:
                print(f"[INFO] ignoring {file_path} --> {file_out_path}")
                continue
            
            # dispatch loader
            if is_format(file_path, ['.jpg', '.jpeg', '.png']):
                input = read_image(file_path, mode=image_mode, order=image_color_order)
            elif is_format(file_path, ['.ply', '.obj', '.glb', '.gltf']):
                from kiui.mesh import Mesh
                input = Mesh.load(file_path)
            else:
                with open(file_path, "r") as f:
                    input = f.read()
            
            # process
            output = process_fn(input, **kwargs)

            # dispatch writer
            if is_format(file_out_path, ['.jpg', '.jpeg', '.png']):
                write_image(file_out_path, output, order=image_color_order)
            elif is_format(file_out_path, ['.ply', '.obj', '.glb', '.gltf']):
                output.write(file_out_path)
            elif is_format(file_out_path, ['.npy']):
                np.save(file_out_path, output)
            else:
                with open(file_out_path, "w") as f:
                    f.write(output)

        except Exception as e:
            print(f"[ERROR] when processing {file_path} --> {file_out_path}")
            print(e)