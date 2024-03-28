import time
import torch
import numpy as np
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from kiui.typing import *
from kiui.utils import lo, write_image


def map_color(value: ndarray, cmap_name: str="viridis", vmin: float=None, vmax: float=None):
    """ map a 1D array to continuous color.

    Args:
        value (ndarray): array of float, [N]
        cmap_name (str, optional): color map name, ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#classes-of-colormaps. Defaults to "viridis".
        vmin (float, optional): min value. Defaults to None.
        vmax (float, optional): max value. Defaults to None.

    Returns:
        ndarray: array of color, [N, 3] in [0, 1]
    """
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]

    if vmin is None:
        vmin = value.min()
    if vmax is None:
        vmax = value.max()
    value = (value - vmin) / (vmax - vmin)  # range in [0, 1]
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def plot_image(*xs, normalize=False, save=False, prefix='kiui_vis_plot_image'):
    """ sequentially plot provided images, optionally save to current dir.
    
    Args:
        xs (Sequence[Union[torch.Tensor, numpy.ndarray]]): can be uint8 or float32.
            [B, 4/3/1, H, W], [B, H, W, 4/3/1], [4/3/1, H, W], [H, W, 4/3/1], [H, W] torch.Tensor or numpy.ndarray
        normalize (bool, optional): whether to renormalize the image to [0, 1]. Defaults to False.
        save (bool, optional): whether to save the image to current dir (in case the plot cannot be showed, like in vscode remote). Defaults to False.
        prefix (str, optional): image save name prefix if save=True.
    """

    _cnt = 0
    _signature = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

    def _plot_image(image):

        nonlocal _cnt

        lo(image)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # empirially to channel-last
        if len(image.shape) == 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose(1, 2, 0)

        # normalize
        if normalize:
            image = (image - image.min(axis=0, keepdims=True)) / (
                image.max(axis=0, keepdims=True)
                - image.min(axis=0, keepdims=True)
                + 1e-8
            )

        if save:
            _path = f'{prefix}_{_signature}_{_cnt}.png'
            _cnt += 1
            write_image(_path, image.astype(np.float32))
            print(f'[INFO] write image to {_path}')
        else:
            plt.imshow(image.astype(np.float32))
            plt.show()

    for x in xs:
        if len(x.shape) == 4:
            for i in range(x.shape[0]):
                _plot_image(x[i])
        else: # 3 or 2
            _plot_image(x)


def plot_matrix(*xs):
    """ visualize some 2D matrix, different from ``kiui.vis.plot_image``, this will keep the original range and plot channel-by-channel.
    
    Args:
        xs (Sequence[Union[torch.Tensor, numpy.ndarray]]): [B, C, H, W], [C, H, W], or [H, W] torch.Tensor or numpy.ndarray
    """
    
    def _plot_matrix(matrix):

        lo(matrix)

        if isinstance(matrix, torch.Tensor):
            if len(matrix.shape) == 3:
                matrix = matrix.permute(1, 2, 0).squeeze()
            matrix = matrix.detach().cpu().numpy()

        if len(matrix.shape) == 3:
            # per channel
            for i in range(matrix.shape[-1]):
                plt.matshow(matrix[..., i])
                plt.show()
        else:
            plt.matshow(matrix.astype(np.float32))
            plt.show()

    for x in xs:
        if len(x.shape) == 4:
            for i in range(x.shape[0]):
                _plot_matrix(x[i])
        else: # 3 or 2
            _plot_matrix(x)


def plot_pointcloud(pc, color=None):
    """plot point cloud.

    Args:
        pc (ndarray): point cloud positions, float [N, 3].
        color (ndarray, optional): point cloud colors, float/uint8 [N, 3/4]. Defaults to None.
    
    Note:
        This function requires a desktop (cannot be forwarded by ssh)!
    """
    
    lo(pc)

    if color is not None:
        lo(color)
        if color.dtype == np.float32:
            color = (color * 255).astype(np.uint8)

    if color is None or color.shape[-1] == 3:
        # use o3d as it's better to control
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])

    else:
        import trimesh

        pc = trimesh.PointCloud(pc, color)
        # axis
        axes = trimesh.creation.axis(axis_length=4)
        # sphere
        box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
        box.colors = np.array([[128, 128, 128]] * len(box.entities))

        trimesh.Scene([pc, axes, box]).show()


def plot_poses(poses, size=0.1, bound=1, points=None, mesh=None, opengl=True):
    """plot camera poses.

    Args:
        poses (ndarray): camera poses, float [N, 4, 4].
        size (float, optional): line width. Defaults to 0.1.
        bound (int, optional): bounding box bound. Defaults to 1.
        points (ndarray, optional): also draw point clouds, float [M, 3]. Defaults to None.
        mesh (trimesh.Trimesh, optional): also draw mesh. Defaults to None.
        opengl (bool, optional): use OpenGL camera convention. Defaults to True.
    
    Note:
        This function requires a desktop (cannot be forwarded by ssh)!
    """

    lo(poses)

    if torch.is_tensor(poses):
        poses = poses.detach().cpu().numpy()

    import trimesh

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2 * bound] * 3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2] * 3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2] * (-1 if opengl else 1)
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2] * (-1 if opengl else 1)
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2] * (-1 if opengl else 1)
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2] * (-1 if opengl else 1)

        # construct 3D paths
        frame = np.array([
            [pos, a],
            [pos, b],
            [pos, c],
            [pos, d],
            [a, b],
            [b, c],
            [c, d],
            [d, a],
            [pos, pos + pose[:3, 2] * (-1 if opengl else 1) * 3], # point to target
        ])
        frame = trimesh.load_path(frame)
        objects.append(frame)

        right_line = np.array([[pos, pos + pose[:3, 0] * size]])
        right_line = trimesh.load_path(right_line)
        right_line.colors = np.array([[255, 0, 0, 255]]).repeat(len(right_line.entities), axis=0)
        objects.append(right_line)

        up_line = np.array([[pos, pos + pose[:3, 1] * size]])
        up_line = trimesh.load_path(up_line)
        up_line.colors = np.array([[0, 255, 0, 255]]).repeat(len(up_line.entities), axis=0)
        objects.append(up_line)

        forward_line = np.array([[pos, pos + pose[:3, 2] * size]])
        forward_line = trimesh.load_path(forward_line)
        forward_line.colors = np.array([[0, 0, 255, 255]]).repeat(len(forward_line.entities), axis=0)
        objects.append(forward_line)

    if points is not None:

        lo(points)

        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255  # blue
        colors[:, 3] = 30  # transparent
        objects.append(trimesh.PointCloud(points, colors))

    if mesh is not None:
        objects.append(mesh)

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()
