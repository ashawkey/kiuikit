import os
import sys
import tqdm
import glob
import math
import random
import argparse
import numpy as np
from contextlib import contextmanager

### blender env 
import bpy
from mathutils import Vector, Matrix

# print(bpy.app.version_string)

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)

def reset_scene():
    # delete everything that isn't part of a camera
    for obj in bpy.data.objects:
        if obj.type not in ["CAMERA"]:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def setup_rendering(args):
    ### render parameters
    bpy.context.scene.render.engine = args.engine

    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.resolution_percentage = 100

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    ### use nodes system for rendering
    bpy.context.scene.use_nodes = True

    ### render nodes (if you want to render more than a shaded image)

    # nodes = bpy.context.scene.node_tree.nodes
    # links = bpy.context.scene.node_tree.links
    # nodes.clear()
    # render_layers = nodes.new("CompositorNodeRLayers")

    # depth
    # bpy.context.view_layer.use_pass_z = True
    # depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    # depth_file_output.label = "Depth Output"
    # depth_file_output.base_path = ""
    # depth_file_output.file_slots[0].use_node_format = True
    # depth_file_output.format.file_format = "OPEN_EXR"
    # depth_file_output.format.color_depth = "16"
    # links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])

    # normal 
    # bpy.context.view_layer.use_pass_normal = True
    # scale_node = nodes.new(type="CompositorNodeMixRGB")
    # scale_node.blend_type = "MULTIPLY"
    # scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    # links.new(render_layers.outputs["Normal"], scale_node.inputs[1])
    # bias_node = nodes.new(type="CompositorNodeMixRGB")
    # bias_node.blend_type = "ADD"
    # bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    # links.new(scale_node.outputs[0], bias_node.inputs[1])

    # normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    # normal_file_output.label = "Normal Output"
    # normal_file_output.base_path = ""
    # normal_file_output.file_slots[0].use_node_format = True
    # normal_file_output.format.file_format = "PNG"
    # normal_file_output.format.color_mode = "RGBA"
    # links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # albedo
    # bpy.context.view_layer.use_pass_diffuse_color = True
    # alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    # links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    # links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    # albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    # albedo_file_output.label = "Albedo Output"
    # albedo_file_output.base_path = ""
    # albedo_file_output.file_slots[0].use_node_format = True
    # albedo_file_output.format.file_format = "PNG"
    # albedo_file_output.format.color_mode = "RGBA"
    # links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

    # NOTE: blender cannot render metallic and roughness as image...

    ### render engine
    # EEVEE will use OpenGL, CYCLES will use GPU + CUDA
    if bpy.context.scene.render.engine == 'CYCLES':
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 64 # 128
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 3
        bpy.context.scene.cycles.transmission_bounces = 3
        bpy.context.scene.cycles.filter_width = 0.01
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.tile_size = 8192

        bpy.context.preferences.addons["cycles"].preferences.get_devices()

        # set which GPU to use
        for i, device in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
            if i == args.gpu:
                device.use = True
                print(f'[INFO] using device {i}: {device}')
            else:
                device.use = False

        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

# a brute-force way to remove the annoying planes under the mesh
# this can be a little dangerous to delete really meaningful part...
def clean_scene_meshes():
    for obj in bpy.data.objects:
        # print(obj)
        if isinstance(obj.data, (bpy.types.Mesh)):
            # typical for a two-triangle plane
            if len(obj.data.vertices) <= 6:
                bpy.data.objects.remove(obj, do_unlink=True)
        

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    resolution_x_in_px = bpy.context.scene.render.resolution_x
    resolution_y_in_px = bpy.context.scene.render.resolution_y
    scale = bpy.context.scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K

def load_hdri(hdri_path):
    
    # use world nodes
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    world_nodes.clear()

    node_world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    node_background = world_nodes.new(type='ShaderNodeBackground')
    node_hdri = world_nodes.new('ShaderNodeTexEnvironment')
    
    world_links.new(node_hdri.outputs["Color"], node_background.inputs["Color"])
    world_links.new(node_background.outputs["Background"], node_world_output.inputs["Surface"])

    node_hdri.image = bpy.data.images.load(hdri_path)

def load_object(mesh):
    if mesh.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=mesh, merge_vertices=True)
    elif mesh.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh)
    else:
        raise ValueError(f"Unsupported file type: {mesh}")

def get_scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def get_scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def normalize_scene(bound=0.9):
    # bound: normalize to [-bound, bound]
    bbox_min, bbox_max = get_scene_bbox()
    scale = 2 * bound / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = get_scene_bbox()
    offset = - (bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def main(args):

    setup_rendering(args)

    # reset scene
    reset_scene()

    # load the object
    name = os.path.basename(args.mesh).split(".")[0]
    os.makedirs(os.path.join(args.outdir, name), exist_ok=True)
    load_object(args.mesh)

    # clever clean scene
    clean_scene_meshes()

    # normalize objects to [-b, b]^3
    normalize_scene(bound=args.bound)

    # load random hdri
    hdri_paths = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/blender_lights/*.exr'))
    random_hdri_path = random.choice(hdri_paths)
    load_hdri(random_hdri_path)

    # orbit camera
    cam = bpy.context.scene.objects["Camera"]
    cam.data.angle = np.deg2rad(args.fovy)

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # place cameras (following zero123++ v1.2 6 views)
    # azimuth in [0, 2pi], elevation in [-pi/2, pi/2]
    # for objaverse, most objects' front view is 270 azimuth!
    azimuths = np.deg2rad(np.array([300, 0, 60, 120, 180, 240]))
    elevations = np.deg2rad(np.array([-20, 10, -20, 10, -20, 10]))
    
    # get camera positions in blender coordinate system
    x = args.radius * np.cos(azimuths) * np.cos(elevations)
    y = args.radius * np.sin(azimuths) * np.cos(elevations)
    z = - args.radius * np.sin(elevations)
    cam_pos = np.stack([x,y,z], axis=-1)

    cam_poses = []

    for i in tqdm.trange(len(azimuths)):
        # set camera
        cam.location = cam_pos[i]
        bpy.context.view_layer.update()

        # pose matrix (c2w)
        c2w = np.eye(4)
        t, R = cam.matrix_world.decompose()[0:2]
        c2w[:3, :3] = np.asarray(R.to_matrix()) # [3, 3]
        c2w[:3, 3] = np.asarray(t)

        # blender to opengl
        c2w_opengl = c2w.copy()
        c2w_opengl[1] *= -1
        c2w_opengl[[1, 2]] = c2w_opengl[[2, 1]]

        cam_poses.append(c2w_opengl)

        # render image
        render_file_path = os.path.join(args.outdir, name, f"{i:03d}")
        bpy.context.scene.render.filepath = render_file_path
        # depth_file_output.file_slots[0].path = render_file_path + "_depth"
        # normal_file_output.file_slots[0].path = render_file_path + "_normal"
        # albedo_file_output.file_slots[0].path = render_file_path + "_albedo"

        if os.path.exists(render_file_path) and not args.overwrite: 
            continue

        with stdout_redirected(): # suppress tons of rendering logs
            bpy.ops.render.render(write_still=True)

    # write camera
    if args.save_camera:
        K = get_calibration_matrix_K_from_blender(cam)
        cam_poses = np.stack(cam_poses, 0)
        np.savez(os.path.join(args.outdir, name, 'cameras.npz'), K=K, poses=cam_poses)

    # save blend file for debugging
    if args.save_blend:
        blend_path = os.path.join(args.outdir, name, f"{name}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--outdir", type=str, default='./')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--engine", type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])

    # saving parameters
    parser.add_argument('--save_blend', action='store_true')
    parser.add_argument('--save_camera', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    # rendering parameters
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bound", type=float, default=0.95)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--fovy", type=float, default=49.1)

    args = parser.parse_args()

    main(args)