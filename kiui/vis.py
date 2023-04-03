import torch
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import trimesh
from .utils import lo

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]

    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def plot_image(*xs, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W] torch.Tensor
    #    [H, W, 3] or [H, W] numpy.ndarray

    def _plot_image(x):

        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3:
                x = x.permute(1,2,0).squeeze()
            x = x.detach().cpu().numpy()

        lo(x)

        x = x.astype(np.float32)
        
        # renormalize
        if renormalize:
            x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

        plt.imshow(x)
        plt.show()
    
    for x in xs:
        _plot_image(x)


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]

    lo(pc)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # if color is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])

    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))

    trimesh.Scene([pc, axes, box]).show()
    

def plot_poses(poses, size=0.05, bound=1, points=None):
    # poses: [B, 4, 4]

    lo(poses)

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255 # blue
        colors[:, 3] = 30 # transparent
        objects.append(trimesh.PointCloud(points, colors))

    # tmp: verify mesh matches the points
    # mesh = trimesh.load('trial_garden_colmap/mesh_stage0/mesh.ply')
    # objects.append(mesh)

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()
