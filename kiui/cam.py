import numpy as np
from .op import *

# camera convention (OpenGL): 
# right-hand, x right, y up, z forward
# elevation in (-90, 90), azimuth in (-180, 180)
# ref: https://note.kiui.moe/vision/camera_intrinsics_exintrics/

# look at
def look_at(campos, target):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    forward_vector = safe_normalize(target - campos)
    up_vector = np.array([[0, 1, 0]], dtype=np.float32)
    right_vector = safe_normalize(np.cross(forward_vector, up_vector))
    up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def circle_pose(elevation, azimuth, radius=1, is_degree=True, target=None):
    # radius: [N,] or scalar
    # elevation: [N,], in (-90, 90)
    # azimuth: [N,], in (-180, 180), from +z to +x is (0, 90)
    # return: [N, 4, 4], camera pose matrix
    N = elevation.shape[0]
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([1, 3], dtype=np.float32)
    campos = np.stack([x, y, z], axis=-1) + target # [N, 3]
    T = np.eye(4, dtype=np.float32)[None, ...].repeat(N, axis=0)
    T[:, :3, :3] = look_at(campos, target)
    T[:, :3, 3] = campos
    return T
