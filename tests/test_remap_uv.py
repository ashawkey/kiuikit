import kiui
from kiui.mesh import Mesh
from kiui.mesh_utils import clean_mesh

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str)
parser.add_argument("--mesh2", type=str)
parser.add_argument("--out", type=str, default='out.glb')
opt = parser.parse_args()

# load input mesh
mesh = Mesh.load(opt.mesh)

# must have texture to remap uv
assert mesh.albedo is not None

# load target mesh
mesh2 = Mesh.load(opt.mesh2)

# no texture for this mesh
assert mesh2.albedo is None

# remap uv
mesh2.albedo = mesh.albedo
mesh2.vt = mesh.remap_uv(mesh2.v)

# write
mesh2.write(opt.out)