import os
import sys
import tqdm
import glob
import math
import random
import argparse
import numpy as np
from contextlib import contextmanager, nullcontext

### blender env 
import bpy
from mathutils import Vector, Matrix

# print('=== BPY VERSION ===', bpy.app.version_string)

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
    ### ref obj to return
    refs = {}

    ### render parameters
    bpy.context.scene.render.engine = args.engine

    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.resolution_percentage = 100

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    ### camera
    cam = bpy.context.scene.objects["Camera"]

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    refs['cam'] = cam
    refs['cam_constraint'] = cam_constraint

    ### shader world nodes
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    world_nodes.clear()

    node_world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    node_background = world_nodes.new(type='ShaderNodeBackground')
    node_hdri = world_nodes.new('ShaderNodeTexEnvironment')
    
    world_links.new(node_hdri.outputs["Color"], node_background.inputs["Color"])
    world_links.new(node_background.outputs["Background"], node_world_output.inputs["Surface"])

    refs['node_hdri'] = node_hdri

    ### compositor nodes 
    bpy.context.scene.use_nodes = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers") # render layers

    ### setting up PBR using custom AOV pass
    # ref: https://www.reddit.com/r/blender/comments/kbno51/custom_render_passes_from_the_shader_editor_with/
    # ref: https://blender.stackexchange.com/questions/23436/control-cycles-eevee-material-nodes-and-material-properties-using-python
    # NOTE: we have really NO error handling here, the input must be correct.
    if args.pbr:
        ### add custom AOV pass in view layer
        bpy.context.view_layer.aovs.add()
        bpy.context.view_layer.active_aov.name = 'albedo'
        bpy.context.view_layer.aovs.add()
        bpy.context.view_layer.active_aov.name = 'metallicroughness'
        
        ### add object shader AOV output node
        # loop all materials
        for mat in bpy.data.materials:
            try:
                mat.use_nodes = True # assert
                mat_nodes = mat.node_tree.nodes
                mat_links = mat.node_tree.links
                node_bsdf = mat_nodes['Principled BSDF'] # assert all these exist...

                # albedo
                node_aov_albedo = mat_nodes.new(type='ShaderNodeOutputAOV')
                node_aov_albedo.name = 'albedo' # link to view layer
                if len(node_bsdf.inputs['Base Color'].links) == 0:
                    # handle pure value case (no texture image)
                    node_singleton_albedo = mat_nodes.new(type='ShaderNodeRGB')
                    node_singleton_albedo.outputs['Color'].default_value = node_bsdf.inputs['Base Color'].default_value
                    mat_links.new(node_singleton_albedo.outputs['Color'], node_aov_albedo.inputs['Color'])
                else:
                    link_albedo = node_bsdf.inputs['Base Color'].links[0]
                    mat_links.new(link_albedo.from_node.outputs[link_albedo.from_socket.name], node_aov_albedo.inputs['Color'])

                # metallic-roughness
                node_aov_metallicroughness = mat_nodes.new(type='ShaderNodeOutputAOV')
                node_aov_metallicroughness.name = 'metallicroughness' # link to view layer
                node_combine_color = mat_nodes.new(type='ShaderNodeCombineRGB')
                if len(node_bsdf.inputs['Metallic'].links) == 0:
                    node_singleton_metallic = mat_nodes.new(type='ShaderNodeValue')
                    node_singleton_metallic.outputs['Value'].default_value = node_bsdf.inputs['Metallic'].default_value
                    mat_links.new(node_singleton_metallic.outputs['Value'], node_combine_color.inputs["B"])
                else:
                    link_metallic = node_bsdf.inputs['Metallic'].links[0]
                    mat_links.new(link_metallic.from_node.outputs[link_metallic.from_socket.name], node_combine_color.inputs["B"])
                if len(node_bsdf.inputs['Roughness'].links) == 0:
                    node_singleton_roughness = mat_nodes.new(type='ShaderNodeValue')
                    node_singleton_roughness.outputs['Value'].default_value = node_bsdf.inputs['Roughness'].default_value
                    mat_links.new(node_singleton_roughness.outputs['Value'], node_combine_color.inputs["G"])
                else:
                    link_roughness = node_bsdf.inputs['Roughness'].links[0]
                    mat_links.new(link_roughness.from_node.outputs[link_roughness.from_socket.name], node_combine_color.inputs["G"])

                mat_links.new(node_combine_color.outputs["Image"], node_aov_metallicroughness.inputs['Color'])
            except Exception as e:
                print(f'[ERROR] failed to set up PBR AOV for material {mat}: {e}')

        ### add compositor output node
        node_albedo_alpha = nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers.outputs["albedo"], node_albedo_alpha.inputs["Image"])
        links.new(render_layers.outputs["Alpha"], node_albedo_alpha.inputs["Alpha"])
        node_albedo = nodes.new(type="CompositorNodeOutputFile")
        node_albedo.label = "Albedo Output"
        node_albedo.base_path = "/"
        node_albedo.file_slots[0].use_node_format = True
        node_albedo.format.file_format = "PNG"
        node_albedo.format.color_mode = "RGBA"
        links.new(node_albedo_alpha.outputs["Image"], node_albedo.inputs[0])
        refs['node_albedo'] = node_albedo

        node_metallicroughness = nodes.new(type="CompositorNodeOutputFile")
        node_metallicroughness.label = "MetallicRoughness Output"
        node_metallicroughness.base_path = "/"
        node_metallicroughness.file_slots[0].use_node_format = True
        node_metallicroughness.format.file_format = "PNG"
        node_metallicroughness.format.color_mode = "RGB"
        links.new(render_layers.outputs["metallicroughness"], node_metallicroughness.inputs[0])
        refs['node_metallicroughness'] = node_metallicroughness

    ## depth
    if args.depth:
        bpy.context.view_layer.use_pass_z = True
        node_depth = nodes.new(type="CompositorNodeOutputFile")
        node_depth.label = "Depth Output"
        node_depth.base_path = "/" # use absolute save path
        node_depth.file_slots[0].use_node_format = True
        node_depth.format.file_format = "OPEN_EXR"
        node_depth.format.color_depth = "16"
        links.new(render_layers.outputs["Depth"], node_depth.inputs[0])

        refs['node_depth'] = node_depth

    ## normal 
    if args.normal:
        bpy.context.view_layer.use_pass_normal = True
        # rescale [-1, 1] * 0.5 + 0.5 to [0, 1]
        node_normal_scale = nodes.new(type="CompositorNodeMixRGB")
        node_normal_scale.blend_type = "MULTIPLY"
        node_normal_scale.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs["Normal"], node_normal_scale.inputs[1])
        node_normal_bias = nodes.new(type="CompositorNodeMixRGB")
        node_normal_bias.blend_type = "ADD"
        node_normal_bias.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(node_normal_scale.outputs[0], node_normal_bias.inputs[1])
        node_normal = nodes.new(type="CompositorNodeOutputFile")
        node_normal.label = "Normal Output"
        node_normal.base_path = "/"
        node_normal.file_slots[0].use_node_format = True
        node_normal.format.file_format = "OPEN_EXR"
        node_normal.format.color_depth = "16"
        links.new(node_normal_bias.outputs[0], node_normal.inputs[0])

        refs['node_normal'] = node_normal

    ### render engine
    # EEVEE will use OpenGL, CYCLES will use GPU + CUDA/OPTIX
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

        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device

    return refs


# a brute-force way to remove the annoying planes under the mesh
# this can be a little dangerous to delete really meaningful part...
def clean_scene_meshes():
    for obj in bpy.data.objects:
        # print(obj)
        if isinstance(obj.data, (bpy.types.Mesh)):
            # typical for a two-triangle plane
            if len(obj.data.vertices) <= 6:
                bpy.data.objects.remove(obj, do_unlink=True)


# remove all animations, which will disturb normalization
def clear_animation():
    for obj in bpy.data.objects:
        obj.animation_data_clear()


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


def load_object(mesh):
    if mesh.lower().endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=mesh, merge_vertices=True)
    elif mesh.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh)
    elif mesh.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh, forward_axis="Z", up_axis="Y")
    elif mesh.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=mesh, forward_axis="Z", up_axis="Y")
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

MAT_ID = 0

def create_simple_material(color, roughness=1, mat_name=None):
    global MAT_ID

    if mat_name is None:
        mat_name = f"assigned_mat_{MAT_ID:06d}"
        MAT_ID += 1

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    tree = mat.node_tree

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = [0, 0, 0, 1]
    # tree.nodes["Principled BSDF"].inputs['Specular IOR Level'].default_value = 0.5
    # tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    # tree.nodes["Principled BSDF"].inputs['Transmission Weight'].default_value = 0
    # tree.nodes["Principled BSDF"].inputs['Coat Roughness'].default_value = 0

    return mat

def create_default_materials():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            surface_mat = create_simple_material((0.10, 0.20, 0.80, 1), roughness=0.5, mat_name="surface_mat") # blue surface, color is a tuple of 4 float (in [0, 1])
            obj.data.materials.append(surface_mat) # 0, default material for surface
            

# enable wireframe rendering
# parameters are hard-coded for now
def enable_wireframe():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_add(type='WIREFRAME')
            bpy.context.object.modifiers["Wireframe"].use_replace = False # do not replace the original mesh
            bpy.context.object.modifiers["Wireframe"].thickness = 0.01 # thickness of wireframe
            bpy.context.object.modifiers["Wireframe"].use_even_offset = False # otherwise lead to spikes...
            # use a different color for surface and wireframe
            surface_mat = create_simple_material((0.10, 0.20, 0.80, 1), roughness=0.5, mat_name="surface_mat") # blue surface, color is a tuple of 4 float (in [0, 1])
            obj.data.materials.append(surface_mat) # 0, default material for surface
            wireframe_mat = create_simple_material((1, 1, 1, 1), roughness=1, mat_name="wireframe_mat") # white wireframe
            obj.data.materials.append(wireframe_mat) # 1
            bpy.context.object.modifiers["Wireframe"].material_offset = 1


def disable_wireframe():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_remove(modifier="Wireframe")
            except:
                pass


def main(args):

    # reset scene
    reset_scene()

    # load the object
    name = os.path.basename(args.mesh).split(".")[0]
    os.makedirs(os.path.join(args.outdir, name), exist_ok=True)
    load_object(args.mesh)
    
    # set up rendering
    refs = setup_rendering(args)
    cam = refs['cam']

    # clever clean scene
    # clean_scene_meshes()

    # normalize objects to [-b, b]^3
    clear_animation()
    normalize_scene(bound=args.bound)

    # load random hdri if not specified
    if args.hdri_path is None:
        hdri_paths = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/blender_lights/*.exr'))
        args.hdri_path = random.choice(hdri_paths)
    
    print(f'[INFO] using hdri: {args.hdri_path}')
    refs['node_hdri'].image = bpy.data.images.load(args.hdri_path)

    # enable wireframe rendering
    if args.wireframe:
        enable_wireframe()
    else:
        create_default_materials()

    # orbit camera
    cam.data.angle = np.deg2rad(args.fovy)
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    refs['cam_constraint'].target = empty

    # place cameras (following zero123++ v1.2, front view + 6 side views)
    # azimuth in [0, 2pi], elevation in [-pi/2, pi/2]
    # for objaverse, most objects' front view is 270 azimuth!
    # azimuths = np.deg2rad(np.array([270, 300, 0, 60, 120, 180, 240]))
    # elevations = np.deg2rad(np.array([0, -20, 10, -20, 10, -20, 10]))

    # standard 6 views with 2 additional views
    azimuths = np.deg2rad(np.array([270, 0, 90, 180, 270, 270, 315, 135]))
    elevations = np.deg2rad(np.array([0, 0, 0, 0, -89.99, 89.99, -45, 45]))
    
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
        render_file_path = os.path.join(args.outdir, name, f"{i:03d}" if not args.wireframe else f"{i:03d}_wireframe")
        render_file_path = os.path.abspath(render_file_path) # relative path leads to problems for depth/normal/albedo nodes...
        
        bpy.context.scene.render.filepath = render_file_path

        if args.depth:
            refs['node_depth'].file_slots[0].path = render_file_path + "_depth"
        if args.normal:
            refs['node_normal'].file_slots[0].path = render_file_path + "_normal"
        if args.pbr:
            refs['node_albedo'].file_slots[0].path = render_file_path + "_albedo"
            refs['node_metallicroughness'].file_slots[0].path = render_file_path + "_mr"

        if os.path.exists(render_file_path) and not args.overwrite: 
            continue
        
        with nullcontext() if args.verbose else stdout_redirected():
            bpy.ops.render.render(write_still=True)

    print(f'[INFO] finished rendering {len(azimuths)} images')

    # write camera
    if args.camera:
        K = get_calibration_matrix_K_from_blender(cam)
        cam_poses = np.stack(cam_poses, 0)
        np.savez(os.path.join(args.outdir, name, 'cameras.npz'), K=K, poses=cam_poses)

    # save blend file for debugging
    if args.blend:
        blend_path = os.path.join(args.outdir, name, f"{name}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--outdir", type=str, default='./')
    parser.add_argument("--engine", type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])
    parser.add_argument("--device", type=str, default='OPTIX', choices=['OPTIX', 'CUDA', 'OPENCL'])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--verbose', action='store_true')

    # saving parameters
    parser.add_argument('--blend', action='store_true')
    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--pbr', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    # rendering parameters
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--hdri_path", type=str, default=None, help='path to hdri (exr), if not provided, random hdri will be used')
    parser.add_argument("--wireframe", action='store_true', help='enable wireframe rendering')
    parser.add_argument("--bound", type=float, default=0.9)
    parser.add_argument("--radius", type=float, default=4.5)
    parser.add_argument("--fovy", type=float, default=30)

    args = parser.parse_args()

    main(args)
