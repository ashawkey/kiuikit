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

def normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim). alias of safe_normalize.

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

def inverse_sigmoid(x: Tensor, eps=1e-6) -> Tensor:
    """inversion of sigmoid function.

    Args:
        x (Tensor): x
        eps (float, optional): eps. Defaults to 1e-6.

    Returns:
        Tensor: log(x / (1 - x))
    """
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def inverse_softplus(x: Tensor) -> Tensor:
    """inversion of softplus function.
    
    Args:
        x (Tensor): x
    
    Returns:
        Tensor: log(exp(x) - 1)
    """
    # a numerically stable equation (ref: https://github.com/pytorch/pytorch/issues/72759)
    return x + torch.log(-torch.expm1(-x))

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


def uv_padding(image: Union[Tensor, ndarray], mask: Union[Tensor, ndarray], padding: Optional[int] = None, backend: Literal['knn', 'cv2'] = 'knn'):
    """padding the uv-space texture image to avoid seam artifacts in mipmaps.

    Args:
        image (Union[Tensor, ndarray]): texture image, float, [H, W, C] in [0, 1].
        mask (Union[Tensor, ndarray]): valid uv region, bool, [H, W].
        padding (int, optional): padding size into the unmasked region. Defaults to 0.1 * max(H, W).
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
    
    if padding is None:
        H, W = image_input.shape[:2]
        padding = int(0.1 * max(H, W))
    
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


def recenter(image: ndarray, mask: ndarray, border_ratio: float = 0.2) -> ndarray:
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """
    
    return_int = False
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
        return_int = True
    
    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    if return_int:
        result = (result * 255).astype(np.uint8)

    return result
