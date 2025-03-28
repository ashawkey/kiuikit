import numpy as np
from scipy.spatial.transform import Rotation

from kiui.op import safe_normalize
from kiui.typing import *

# convert between different world coordinate systems
def convert(
    pose, 
    target: Literal['unity', 'blender', 'opencv', 'colmap', 'opengl', 'unreal'] = 'unity', 
    original: Literal['unity', 'blender', 'opencv', 'colmap', 'opengl', 'unreal'] = 'opengl',
):
    """A method to convert between different world coordinate systems.

    Args:
        pose (np.ndarray): camera pose, float [4, 4].
        target (Literal[&#39;unity&#39;, &#39;blender&#39;, &#39;opencv&#39;, &#39;colmap&#39;, &#39;opengl&#39;, &#39;unreal&#39;], optional): from convention. Defaults to 'unity'.
        original (Literal[&#39;unity&#39;, &#39;blender&#39;, &#39;opencv&#39;, &#39;colmap&#39;, &#39;opengl&#39;, &#39;unreal&#39;], optional): to convention. Defaults to 'opengl'.

    Returns:
        np.ndarray: converted camera pose, float [4, 4].
    """
    
    if original == 'opengl':
        if target == 'unity':
            pose[2] *= -1
        elif target == 'blender':
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1:3] *= -1
        elif target == 'unreal':
            pose[[1, 2]] = pose[[2, 1]]
    elif original == 'unity':
        if target == 'opengl':
            pose[2] *= -1
        elif target == 'blender':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1] *= -1
        elif target == 'unreal':
            pose[[1, 2]] = pose[[2, 1]]
            pose[1] *= -1
    elif original == 'blender':
        if target == 'opengl':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unity':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unreal':
            pose[1] *= -1
    elif original in ['opencv', 'colmap']:
        if target == 'opengl':
            pose[1:3] *= -1
        elif target == 'unity':
            pose[1] *= -1
        elif target == 'blender':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unreal':
            pose[[1, 2]] = pose[[2, 1]]
            pose[:2] *= -1
    elif original == 'unreal':
        if target == 'opengl':
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unity':
            pose[[1, 2]] = pose[[2, 1]]
            pose[2] *= -1
        elif target == 'blender':
            pose[1] *= -1
        elif target in ['opencv', 'colmap']:
            pose[[1, 2]] = pose[[2, 1]]
            pose[:2] *= -1
    return pose


def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """
   
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    """construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Args:
        elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
        azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
        radius (float, optional): camera radius. Defaults to 1.
        is_degree (bool, optional): if the angles are in degree. Defaults to True.
        target (np.ndarray, optional): look at target position. Defaults to None.
        opengl (bool, optional): whether to use OpenGL camera convention. Defaults to True.

    Returns:
        np.ndarray: the camera pose matrix, float [4, 4]
    """
    
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


def undo_orbit_camera(T, is_degree=True):
    """ undo an orbital camera pose matrix to elevation & azimuth

    Args:
        T (np.ndarray): camera pose matrix, float [4, 4], must be an orbital camera targeting at (0, 0, 0)!
        is_degree (bool, optional): whether to return angles in degree. Defaults to True.

    Returns:
        Tuple[float]: elevation, azimuth, and radius.
    """
    
    campos = T[:3, 3]
    radius = np.linalg.norm(campos)
    elevation = np.arcsin(-campos[1] / radius)
    azimuth = np.arctan2(campos[0], campos[2])
    if is_degree:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)
    return elevation, azimuth, radius

# perspective matrix
def get_perspective(fovy, aspect=1, near=0.01, far=1000):
    """construct a perspective matrix from fovy.

    Args:
        fovy (float): field of view in degree along y-axis.
        aspect (int, optional): aspect ratio. Defaults to 1.
        near (float, optional): near clip plane. Defaults to 0.01.
        far (int, optional): far clip plane. Defaults to 1000.

    Returns:
        np.ndarray: perspective matrix, float [4, 4]
    """
    # fovy: field of view in degree.
    
    y = np.tan(np.deg2rad(fovy) / 2)
    return np.array(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -(2 * far * near) / (far - near),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def get_rays(pose, h, w, fovy, opengl=True, normalize_dir=True):
    """ construct rays origin and direction from a camera pose.

    Args:
        pose (np.ndarray): camera pose, float [4, 4]
        h (int): image height
        w (int): image width
        fovy (float): field of view in degree along y-axis.
        opengl (bool, optional): whether to use the OpenGL camera convention. Defaults to True.
        normalize_dir (bool, optional): whether to normalize the ray directions. Defaults to True.

    Returns:
        Tuple[np.ndarray]: rays_o and rays_d, both are float [h, w, 3]
    """
    # pose: [4, 4]
    # fov: in degree
    # opengl: camera front view convention

    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x = x.reshape(-1)
    y = y.reshape(-1)

    cx = w * 0.5
    cy = h * 0.5

    # objaverse rendering has fixed focal of 560 at resolution 512 --> fov = 49.1 degree
    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = np.stack([
        (x - cx + 0.5) / focal,
        (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
        np.ones_like(x) * (-1.0 if opengl else 1.0),
    ], axis=-1) # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = np.expand_dims(pose[:3, 3], 0).repeat(rays_d.shape[0], 0)  # [hw, 3]

    if normalize_dir:
        rays_d = safe_normalize(rays_d)

    rays_o = rays_o.reshape(h, w, 3)
    rays_d = rays_d.reshape(h, w, 3)

    return rays_o, rays_d

class OrbitCamera:
    """ An orbital camera class.
    """
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        """init function

        Args:
            W (int): image width
            H (int): image height
            r (float, optional): camera radius. Defaults to 2.
            fovy (float, optional): camera field of view in degree along y-axis. Defaults to 60.
            near (float, optional): near clip plane. Defaults to 0.01.
            far (float, optional): far clip plane. Defaults to 100.
        """
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = Rotation.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        """get the field of view in radians along x-axis

        Returns:
            float: field of view in radians along x-axis
        """
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        """get the camera position

        Returns:
            np.ndarray: camera position, float [3]
        """
        return self.pose[:3, 3]


    @property
    def pose(self):
        """get the camera pose matrix (cam2world)

        Returns:
            np.ndarray: camera pose, float [4, 4]
        """
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    
    @property
    def view(self):
        """get the camera view matrix (world2cam, inverse of cam2world)

        Returns:
            np.ndarray: camera view, float [4, 4]
        """
        return np.linalg.inv(self.pose)

    
    @property
    def perspective(self):
        """get the perspective matrix

        Returns:
            np.ndarray: camera perspective, float [4, 4]
        """
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        """get the camera intrinsics

        Returns:
            np.ndarray: intrinsics (fx, fy, cx, cy), float [4]
        """
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    
    @property
    def mvp(self):
        """get the MVP (model-view-perspective) matrix.

        Returns:
            np.ndarray: camera MVP, float [4, 4]
        """
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        """ rotate along camera up/side axis!

        Args:
            dx (float): delta step along x (up).
            dy (float): delta step along y (side).
        """
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        """scale the camera.

        Args:
            delta (float): delta step.
        """
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        """pan the camera.

        Args:
            dx (float): delta step along x.
            dy (float): delta step along y.
            dz (float, optional): delta step along x. Defaults to 0.
        """
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

    def from_angle(self, elevation, azimuth, is_degree=True):
        """set the camera pose from elevation & azimuth angle.

        Args:
            elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
            azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
            is_degree (bool, optional): whether the angles are in degree. Defaults to True.
        """
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)
        x = self.radius * np.cos(elevation) * np.sin(azimuth)
        y = - self.radius * np.sin(elevation)
        z = self.radius * np.cos(elevation) * np.cos(azimuth)
        campos = np.array([x, y, z])  # [N, 3]
        rot_mat = look_at(campos, np.zeros([3], dtype=np.float32))
        self.rot = Rotation.from_matrix(rot_mat)