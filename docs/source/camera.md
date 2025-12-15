# Camera

### World coordinate systems
```
OpenGL/MAYA       OpenCV/Colmap     Blender      Unity/DirectX     Unreal   
Right-handed      Right-handed    Right-handed    Left-handed    Left-handed

     +y                +z           +z  +y         +y  +z          +z
     |                /             |  /           |  /             |
     |               /              | /            | /              |
     |______+x      /______+x       |/_____+x      |/_____+x        |______+x
    /               |                                              /
   /                |                                             / 
  /                 |                                            /      
 +z                 +y                                          +y
```

A common color code: right = <span style="color:red">red</span>., up = <span style="color:green">green</span>, forward = <span style="color:blue">blue</span> (XYZ=RUF=RGB).

> **Left/right-handed notation**: roll your left/right palm from x to y, and your thumb should point to z.

### Our camera convention
* world coordinate is OpenGL/right-handed, `+x = right, +y = up, +z = forward`
* camera coordinate is OpenGL (forward is `target --> campos`).
* elevation in (-90, 90), from `+y (-90) --> -y (+90)`
* azimuth in (-180, 180), from `+z (0/-360) --> +x (90/-270) --> -z (180/-180) --> -x (270/-90) --> +z (360/0)`


### Camera pose conventions

We call the camera to world (c2w) transformation matrix strictly as `pose`.
The inversion of it is the world to camera (w2c) transformation matrix is called `extrinsics` (or the `view` transform in graphics).

A camera pose matrix is in the form of:
```
[[Right_x, Up_x, Forward_x, Position_x],
 [Right_y, Up_y, Forward_y, Position_y],
 [Right_z, Up_z, Forward_z, Position_z],
 [0,       0,    0,         1         ]]
```

The xyz follows corresponding world coordinate system.
However, the three directions (right, up, forward) can be defined differently:
* forward can be `camera --> target` or `target --> camera`.
* up can align with the world-up-axis (`y`) or world-down-axis (`-y`).
* right can also be left, depending on it's (`up cross forward`) or (`forward cross up`).

This leads to two common camera conventions:
```
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
```

These two conventions can be converted by negating the forward and up directions:
```python
pose # camera2world, [4, 4]
# convert between opengl and colmap convention
pose[:, 1] *= -1 # up
pose[:, 2] *= -1 # forward
```
Note that camera convention is NOT dependent on the world coordinate system!

#### Why there are different camera conventions?
_It's an unfortunate outcome of different 3D world coordinate systems and 2D image coordinate systems._
For a 2D image / array / matrix, the convention is usually:
```
       0      1      2
  +---------------------> +x/j
0 | (0, 0) (0, 1) (0, 2)
1 | (1, 0) (1, 1) (1, 2)
2 | (2, 0) (2, 1) (2, 2)
  v
 +y/i
```
Now assume we use OpenGL world coordinate system and OpenGL camera convention:
```
     +y & up        
     |  target       
     | /        
     |/______+x & right
    /          
   /           
  /            
 +z & forward             
```
The unit camera (Identity rotation matrix) is perfectly aligned with the 3D world coordinate system.
However, if we want to construct the camera rays (camera center --> target), the rays should point to `-z`, and the `+y` axis is also misaligned with image coordinate system (where `+y` points down).
So we need to negate both y and z axes during the ray construction (check `kiui.cam.get_rays`):
```python
rays_d = np.stack([
    (x - cx + 0.5) / focal,
    (y - cy + 0.5) / focal * (-1.0),
    np.ones_like(x) * (-1.0),
], axis=-1) # [hw, 3]
```

On the other hand, if we use OpenCV world coordinate system and OpenCV camera convention:
```
         +z & forward & target
        /
       /
      /_______+x & right
      |
      |
      |
     +y & up           
```
You can see the unit camera is perfectly aligned with the 2D image convention.
And we don't need to negate any axes during the ray construction:
```python
rays_d = np.stack([
    (x - cx + 0.5) / focal,
    (y - cy + 0.5) / focal,
    np.ones_like(x),
], axis=-1) # [hw, 3]
```

#### Given a poorly-documented dataset of cameras (and depth maps), how to make sure its conventions?

1. check if the camera "pose" is pose (camera2world) or extrinsics (world2camera), many datasets are misusing the term "pose" to mean extrinsics.
2. check if the depth map is the distance to camera center or to the camera plane. If it's to the camera center, we need to normalize the `rays_d` to unit length. Otherwise, just use the above raw rays (`|z|` is always 1).
3. simply assume a camera convention (e.g., OpenGL) and use it to construct the rays, and project depth to world-space point cloud for visualization.
4. Check if point cloud of static objects are drifting (they shouldn't if everything is correct). If they are, you are using the wrong camera convention. Change to the other convention and it should be correct!

This is a complete script:
```python
import os
import re
import cv2  
import kiui 
import glob
import tqdm
import numpy as np
from plyfile import PlyData, PlyElement

def glob_sorted(path: str):
    paths = glob.glob(path)
    def _natural_key(name: str):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]
    return sorted(
        paths,
        key=lambda p: _natural_key(os.path.splitext(os.path.basename(p))[0]),
    )

def export_ply_with_color(point_cloud: np.ndarray, colors: np.ndarray, file_path: str) -> None:
    assert point_cloud.shape[0] == colors.shape[0]
    vertices = np.zeros(
        point_cloud.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"] = point_cloud[:, 0].astype(np.float32)
    vertices["y"] = point_cloud[:, 1].astype(np.float32)
    vertices["z"] = point_cloud[:, 2].astype(np.float32)
    vertices["red"] = colors[:, 0].astype(np.uint8)
    vertices["green"] = colors[:, 1].astype(np.uint8)
    vertices["blue"] = colors[:, 2].astype(np.uint8)
    el = PlyElement.describe(vertices, "vertex")
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    PlyData([el]).write(file_path)


def load_data(path: str, target_size: tuple[int, int] = (320, 180)):
    
    # load camera
    cam_params = kiui.read_json(path)
    width, height = int(cam_params['renderProductResolution'][0]), int(cam_params['renderProductResolution'][1])
    proj = np.array(cam_params["cameraProjection"]).reshape(4, 4).T # [4, 4], perspective projection (T is for col-major to row-major)
    w2c = np.array(cam_params["cameraViewTransform"]).reshape(4, 4).T # [4, 4], world2cam 

    # get intrinsics from projection matrix
    fx = proj[0, 0] * (width / 2.0)
    fy = proj[1, 1] * (height / 2.0)
    cx = (1.0 - proj[0, 2]) * (width / 2.0)
    cy = (1.0 - proj[1, 2]) * (height / 2.0)

    c2w = np.linalg.inv(w2c)

    # load depth
    path_depth = path.replace("camera_params", "distance_to_camera").replace(".json", ".exr")
    depth = kiui.read_image(path_depth, mode="float")
    depth = depth.astype(np.float32) # [H, W]

    # downsample depth and intrinsics for visualization
    d_width, d_height = int(target_size[0]), int(target_size[1])
    d_depth = cv2.resize(depth, (d_width, d_height), interpolation=cv2.INTER_NEAREST)
    scale_x = float(d_width) / float(width)
    scale_y = float(d_height) / float(height)
    d_fx = fx * scale_x
    d_fy = fy * scale_y
    d_cx = cx * scale_x
    d_cy = cy * scale_y

    # create world space point cloud

    ### IMPORTANT: toggle camera convention 
    # here the camera is OpenGL convention actually

    # either toggle the c2w convention
    # c2w[:, 1] *= -1
    # c2w[:, 2] *= -1

    # or toggle the rays construction (but don't do both!).
    OPENGL_RAYS = True

    x, y = np.meshgrid(np.arange(d_width), np.arange(d_height), indexing="xy")
    x = x.reshape(-1)
    y = y.reshape(-1)
    rays_d = np.stack([
        (x - d_cx + 0.5) / d_fx,
        (y - d_cy + 0.5) / d_fy * (-1.0 if OPENGL_RAYS else 1.0),
        np.ones_like(x) * (-1.0 if OPENGL_RAYS else 1.0),
    ], axis=-1) # [hw, 3]

    rays_d = rays_d @ c2w[:3, :3].T  # [hw, 3]
    rays_o = np.expand_dims(c2w[:3, 3], 0).repeat(rays_d.shape[0], 0)  # [hw, 3]

    # since depth is to camera center, we need to normalize the rays_d
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = rays_o.reshape(d_height, d_width, 3)
    rays_d = rays_d.reshape(d_height, d_width, 3)

    valid_mask = np.isfinite(d_depth) & (d_depth > 0)
    points = rays_d[valid_mask] * d_depth[valid_mask][:, None] + rays_o[valid_mask] # [N, 3] in world space

    # load rgb for color visualization
    path_rgb = path.replace("camera_params", "ldr_color").replace(".json", ".png")
    rgb = kiui.read_image(path_rgb, mode="float") # [H, W, 3]
    d_rgb = cv2.resize(rgb, (d_width, d_height), interpolation=cv2.INTER_AREA)
    d_rgb = d_rgb[valid_mask]  # [N, 3]

    return c2w, points, d_rgb

def c2w_to_pointcloud(c2ws: list[np.ndarray], point_per_axis: int = 5, axis_length: float = 1.0):
    # visualize the camera poses as point clouds, c2w is [4, 4]

    # debug: add the identity pose to visualize
    c2ws.append(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]))

    axis_points = []
    axis_colors = []
    # Right, Up, Forward unit directions are columns of rotation in c2w
    for c2w in c2ws:
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        dirs = [
            (R[:, 0], np.array([1, 0, 0], dtype=np.float32)),   # Right (X) - Red
            (R[:, 1], np.array([0, 1, 0], dtype=np.float32)),   # Up (Y) - Green
            (R[:, 2], np.array([0, 0, 1], dtype=np.float32)),   # Forward (Z) - Blue
        ]
        for d, col in dirs:
            for k in range(point_per_axis):
                p = t + d * (axis_length * (k / (point_per_axis - 1)))
                axis_points.append(p)
                axis_colors.append(col)

    axis_points_np = np.stack(axis_points, axis=0).astype(np.float32)
    axis_colors_np = np.stack(axis_colors, axis=0).astype(np.float32)
    return axis_points_np, axis_colors_np

camera_paths = glob_sorted("camera_params/*.json")

all_c2ws = []
all_points = []
all_colors = []
for path in tqdm.tqdm(camera_paths):
    c2w, points, rgb = load_data(path, target_size=(320, 180))
    all_c2ws.append(c2w)
    all_points.append(points)
    all_colors.append(rgb)

# visualize the camera poses as point clouds
axis_points, axis_colors = c2w_to_pointcloud(all_c2ws, point_per_axis=5, axis_length=1.0)
all_points.append(axis_points)
all_colors.append(axis_colors)

# concatenate all the points and colors
all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)
all_colors = np.clip(all_colors * 255.0, 0.0, 255.0).astype(np.uint8)

# save colored point cloud (e.g., check with meshlab)
export_ply_with_color(all_points, all_colors, "world_pointcloud.ply")
```

### API

.. note::
   the camera API is designed to be `numpy` based and un-batched!

.. automodule:: kiui.cam
   :members: