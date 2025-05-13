import os
import sys
import time
from argparse import ArgumentParser

import bpy
from quad_remesher_1_2 import QREMESHER_OT_remesh as theOp
from quad_remesher_1_2.qr_operators import doRemeshing_Start, doRemeshing_Finish

# ONLY tested on Linux blender 4.1 + qremesher 1.2
# Usage: 
# 1. make sure quad remesher is installed and activated (trial version is also ok).
# 2. call by "blender -b -P remesh.py -- --mesh ./input.glb --output ./output.glb --target_face 5000"
    
def clear_scene():
    bpy.ops.object.select_all(action='DESELECT') # unselect all
    for mesh in bpy.data.meshes: # meshes
        bpy.data.meshes.remove(mesh)
    for obj in bpy.data.objects: # all the nodes in right-top
        bpy.data.objects.remove(obj)

def import_scene(filepath):
    # import as scene (may contain multiple meshes)
    if filepath.lower().endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif filepath.lower().endswith('.glb') or filepath.lower().endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError('Unsupported file format')

def get_mesh_objects():
    meshes = []
    for obj in bpy.data.objects:
        if isinstance(obj.data, (bpy.types.Mesh)):
            meshes.append(obj)
    return meshes

def join_scene():
    # only select mesh objects!
    mesh_objs = get_mesh_objects()
    for obj in mesh_objs:
        obj.select_set(True)
    # need a major active object (all the others will join into this)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    # call join
    bpy.ops.object.join()

def print_scene():
    print('----------------')
    for obj in bpy.data.objects:
        print(f"{obj}: data: {obj.data}")
    print('----------------')


def main(mesh_path, output_path, target_count=5000):

    # clear default cube
    clear_scene()
    
    # import scene
    import_scene(mesh_path)

    # join all meshes into one
    join_scene()

    # select the only and first mesh now
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objs = get_mesh_objects()
    mesh_objs[0].select_set(True)

    # set quadremesher parameters
    bpy.context.scene.qremesher.target_count = target_count
    bpy.context.scene.qremesher.adaptive_size = 50
    bpy.context.scene.qremesher.adapt_quad_count = True
    bpy.context.scene.qremesher.autodetect_hard_edges = True

    # call remesh op and wait until finishing
    doRemeshing_Start(theOp, bpy.context)
    while True:
        retval, log, _ = theOp.progressData.get_progress_status()
        # print(f'[INFO] {retval}: {log}')
        if retval == 2: # SUCCESS -> import the result
            doRemeshing_Finish(theOp, bpy.context)
            break
        else:
            time.sleep(0.2)
                
    # delete the old mesh
    for obj in mesh_objs:
        bpy.data.objects.remove(obj)

    # export mesh as glb
    bpy.ops.export_scene.gltf(export_format='GLB', filepath=output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target_face", type=int, default=5000)
    opt = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(opt.mesh, opt.output, opt.target_face)
