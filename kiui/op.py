import cv2
import torch
import numpy as np

from kiui.typing import *
from kiui.grid_put import grid_put

# torch / numpy math utils
def dot(x: Union[Tensor, ndarray], y: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)

def make_divisible(x: int, m: int = 8):
    """make an int x divisible by m.

    Args:
        x (int): x
        m (int, optional): m. Defaults to 8.

    Returns:
        int: x + (m - x % m)
    """
    return int(x + (m - x % m))

def trunc_rev_sigmoid(x: Tensor, eps=1e-6) -> Tensor:
    """inversion of sigmoid function.

    Args:
        x (Tensor): x
        eps (float, optional): eps. Defaults to 1e-6.

    Returns:
        Tensor: log(x / (1 - x))
    """
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

# torch image scaling
def scale_img_nhwc(x: Tensor, size: Sequence[int], mag='bilinear', min='bilinear') -> Tensor:
    """image scaling helper.

    Args:
        x (Tensor): input image, float [N, H, W, C]
        size (Sequence[int]): target size, tuple of [H', W']
        mag (str, optional): upscale interpolation mode. Defaults to 'bilinear'.
        min (str, optional): downscale interpolation mode. Defaults to 'bilinear'.

    Returns:
        Tensor: rescaled image, float [N, H', W', C]
    """
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x: Tensor, size: Sequence[int], mag='bilinear', min='bilinear') -> Tensor:
    """image scaling helper.

    Args:
        x (Tensor): input image, float [H, W, C]
        size (Sequence[int]): target size, tuple of [H', W']
        mag (str, optional): upscale interpolation mode. Defaults to 'bilinear'.
        min (str, optional): downscale interpolation mode. Defaults to 'bilinear'.

    Returns:
        Tensor: rescaled image, float [H', W', C]
    """
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x: Tensor, size: Sequence[int], mag='bilinear', min='bilinear') -> Tensor:
    """image scaling helper.

    Args:
        x (Tensor): input image, float [N, H, W]
        size (Sequence[int]): target size, tuple of [H', W']
        mag (str, optional): upscale interpolation mode. Defaults to 'bilinear'.
        min (str, optional): downscale interpolation mode. Defaults to 'bilinear'.

    Returns:
        Tensor: rescaled image, float [N, H', W']
    """
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x: Tensor, size: Sequence[int], mag='bilinear', min='bilinear') -> Tensor:
    """image scaling helper.

    Args:
        x (Tensor): input image, float [H, W]
        size (Sequence[int]): target size, tuple of [H', W']
        mag (str, optional): upscale interpolation mode. Defaults to 'bilinear'.
        min (str, optional): downscale interpolation mode. Defaults to 'bilinear'.

    Returns:
        Tensor: rescaled image, float [H', W']
    """
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]


def uv_padding(image: Union[Tensor, ndarray], mask: Union[Tensor, ndarray], padding: int = 2, backend: Literal['knn', 'cv2'] = 'knn'):
    """padding the uv-space texture image to avoid seam artifacts.

    Args:
        image (Union[Tensor, ndarray]): texture image, float, [H, W, C] in [0, 1].
        mask (Union[Tensor, ndarray]): valid uv region, bool, [H, W].
        padding (int, optional): padding size into the unmasked region. Defaults to 2.
        backend (Literal[&#39;knn&#39;, &#39;cv2&#39;], optional): algorithm backend, knn is faster. Defaults to 'knn'.

    Returns:
        Union[Tensor, ndarray]: padded texture image. float, [H, W, C].
    """
    

    if torch.is_tensor(image):
        image_input = image.detach().cpu().numpy()
    else:
        image_input = image

    if torch.is_tensor(mask):
        mask_input = mask.detach().cpu().numpy()
    else:
        mask_input = mask
    
    # padding backend
    if backend == 'knn':

        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        inpaint_region = binary_dilation(mask_input, iterations=padding)
        inpaint_region[mask_input] = 0

        search_region = mask_input.copy()
        not_search_region = binary_erosion(search_region, iterations=2)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        inpaint_image = image_input.copy()
        inpaint_image[tuple(inpaint_coords.T)] = inpaint_image[tuple(search_coords[indices[:, 0]].T)]

    elif backend == 'cv2':
        # kind of slow
        inpaint_image = cv2.inpaint(
            (image_input * 255).astype(np.uint8),
            (~mask_input * 255).astype(np.uint8),
            padding,
            cv2.INPAINT_TELEA,
        ).astype(np.float32) / 255

    if torch.is_tensor(image):
        inpaint_image = torch.from_numpy(inpaint_image).to(image)
    
    return inpaint_image
