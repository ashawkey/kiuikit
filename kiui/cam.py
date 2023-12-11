import numpy as np
from .op import safe_normalize
from scipy.spatial.transform import Rotation
from typing import Literal

''' common world coordinate system conventions

   OpenGL          OpenCV           Blender        Unity             
Right-handed       Colmap                        Left-handed  

     +y                +z           +z  +y         +y  +z                                               
     |                /             |  /           |  /                                               
     |               /              | /            | /                                                   
     |______+x      /______+x       |/_____+x      |/_____+x                                          
    /               |                                                                                        
   /                |                                                                                                  
  /                 |                                                                                         
 +z                 +y                                                                                           

A common color code: x = red, y = green, z = blue (XYZ=RGB)
Left/right-handed notation: Thumb = right (x), Index = up (y), Middle = forward (z).
'''

# convert between different world coordinate systems
def convert(
    pose, 
    target: Literal['unity', 'blender', 'opencv', 'colmap', 'opengl'] = 'unity', 
    original: Literal['unity', 'blender', 'opencv', 'colmap', 'opengl'] = 'opengl',
):
    # pose: [4, 4]
    # target/original: 'unity', 'blender', 'opencv', 'colmap', 'opengl'
    if original == 'opengl':
        if target == 'unity':
            pose[2] *= -1
        elif target == 'blender':
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1:3] *= -1
    elif original == 'unity':
        if target == 'opengl':
            pose[2] *= -1
        elif target == 'blender':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
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
    elif original in ['opencv', 'colmap']:
        if target == 'opengl':
            pose[1:3] *= -1
        elif target == 'unity':
            pose[1] *= -1
        elif target == 'blender':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
    return pose


''' camera pose matrix
[[Right_x, Up_x, Forward_x, Position_x],
 [Right_y, Up_y, Forward_y, Position_y],
 [Right_z, Up_z, Forward_z, Position_z],
 [0,       0,    0,         1         ]]

The xyz follows corresponding world coordinate system.
However, the three directions (right, up, forward) can be defined differently:
(1) forward can be (camera --> target) or (target --> camera).
(2) up can align with the world-up-axis (y) or world-down-axis (-y).
(3) right can also be left, depending on it's (up cross forward) or (forward cross up).

Two common camera coordinate conventions:

   OpenGL                OpenCV       
   Blender               Colmap       

     up  target          forward & target
     |  /                /         
     | /                /          
     |/_____right      /______right   
    /                  |           
   /                   |           
  /                    |           
forward                up          

A common color code: right = red, up = green, forward = blue (XYZ=RUF=RGB)

But many datasets are just very confusing and combine different conventions together.
You may check a few poses to make sure what the convention they are using...
'''

# our camera convention:
# world coordinate is OpenGL/right-handed, +x = right, +y = up, +z = forward
# camera coordinate is OpenGL (forward is target --> campos).
# elevation in (-90, 90), from +y (-90) --> -y (+90)
# azimuth in (-180, 180), from +z (0/-360) --> +x (90/-270) --> -z (180/-180) --> -x (270/-90) --> +z (360/0)

# construct rotation matrix by look-at
def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
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


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
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

# orbit pose to elevation & azimuth
def undo_orbit_camera(T, is_degree=True):
    # T: [4, 4], camera pose matrix
    # return: elevation, azimuth, radius
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


def get_rays(pose, h, w, fovy, opengl=False):
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

    rays_o = rays_o.reshape(h, w, 3)
    rays_d = safe_normalize(rays_d).reshape(h, w, 3)

    return rays_o, rays_d


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
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
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
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

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
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
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    # model-view-perspective
    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

    def from_angle(self, elevation, azimuth, is_degree=True):
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)
        x = self.radius * np.cos(elevation) * np.sin(azimuth)
        y = - self.radius * np.sin(elevation)
        z = self.radius * np.cos(elevation) * np.cos(azimuth)
        campos = np.array([x, y, z])  # [N, 3]
        rot_mat = look_at(campos, np.zeros([3], dtype=np.float32))
        self.rot = Rotation.from_matrix(rot_mat)