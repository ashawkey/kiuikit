"""
This code is heavily borrowed from: https://github.com/Profactor/continuous-remeshing
"""
import os
import tqdm
import tyro
import trimesh
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
import torch_scatter

import kiui
from kiui.mesh import Mesh
from kiui.cam import orbit_camera, get_perspective


def prepend_dummies(
    vertices: torch.Tensor,  # V,D
    faces: torch.Tensor,  # F,3 long
):
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V, D = vertices.shape
    vertices = torch.concat(
        (torch.full((1, D), fill_value=torch.nan, device=vertices.device), vertices),
        dim=0,
    )
    faces = torch.concat(
        (torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces + 1), dim=0
    )
    return vertices, faces


def remove_dummies(
    vertices: torch.Tensor,  # V,D - first vertex all nan and unreferenced
    faces: torch.Tensor,  # F,3 long - first face all zeros
):
    """remove dummy elements added with prepend_dummies()"""
    return vertices[1:], faces[1:] - 1


def calc_edges(
    faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
    with_edge_to_face: bool = False,
):
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |
    e0---->-o
    """

    F = faces.shape[0]

    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F,3,2
    full_edges = face_edges.reshape(F * 3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(
        input=sorted_edges, sorted=True, return_inverse=True, dim=0
    )  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros(
        (E, 2, 2), dtype=torch.long, device=faces.device
    )  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(
        torch.arange(0, F, device=faces.device), torch.arange(0, 3, device=faces.device)
    )  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(
        dim=0,
        index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
        src=scatter_src,
    )  # E,LR=2,S=2
    edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face


def calc_edge_length(
    vertices: torch.Tensor,  # V,3 first may be dummy
    edges: torch.Tensor,  # E,2 long, lower vertex index first, (0,0) for unused
):  # E
    full_vertices = vertices[edges]  # E,2,3
    a, b = full_vertices.unbind(dim=1)  # E,3
    return torch.norm(a - b, p=2, dim=-1)


def calc_face_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # F,3 long, first face may be all zero
    normalize: bool = False,
):  # F,3
    """
       n
       |
       c0     corners ordered counterclockwise when
      / \     looking onto surface (in neg normal direction)
    c1---c2
    """
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = F.normalize(face_normals, eps=1e-6, dim=1)  # TODO inplace?
    return face_normals  # F,3


def calc_vertex_normals(
    vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
    faces: torch.Tensor,  # M,3 long, first face may be all zero
    face_normals: torch.Tensor = None,  # M,3, not normalized
):  # M,3
    M = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)

    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device
    )  # V,C=3,3
    vertex_normals.scatter_add_(
        dim=0,
        index=faces[:, :, None].expand(M, 3, 3),
        src=face_normals[:, None, :].expand(M, 3, 3),
    )
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return F.normalize(vertex_normals, eps=1e-6, dim=1)


def calc_face_ref_normals(
    faces: torch.Tensor,  # F,3 long, 0 for unused
    vertex_normals: torch.Tensor,  # V,3 first unused
    normalize: bool = False,
):  # F,3
    """calculate reference normals for face flip detection"""
    full_normals = vertex_normals[faces]  # F,C=3,3
    ref_normals = full_normals.sum(dim=1)  # F,3
    if normalize:
        ref_normals = F.normalize(ref_normals, eps=1e-6, dim=1)
    return ref_normals


def pack(
    vertices: torch.Tensor,  # V,3 first unused and nan
    faces: torch.Tensor,  # F,3 long, 0 for unused
):  # (vertices,faces), keeps first vertex unused
    """removes unused elements in vertices and faces"""
    V = vertices.shape[0]

    # remove unused faces
    used_faces = faces[:, 0] != 0
    used_faces[0] = True
    faces = faces[used_faces]  # sync

    # remove unused vertices
    used_vertices = torch.zeros(V, 3, dtype=torch.bool, device=vertices.device)
    used_vertices.scatter_(
        dim=0, index=faces, value=True, reduce="add"
    )  # TODO int faster?
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices]  # sync

    # update used faces
    ind = torch.zeros(V, dtype=torch.long, device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] = torch.arange(0, V1, device=vertices.device)  # sync
    faces = ind[faces]

    return vertices, faces


def split_edges(
    vertices: torch.Tensor,  # V,3 first unused
    faces: torch.Tensor,  # F,3 long, 0 for unused
    edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
    face_to_edge: torch.Tensor,  # F,3 long 0 for unused
    splits,  # E bool
    pack_faces: bool = True,
):  # (vertices,faces)
    #   c2                    c2               c...corners = faces
    #    . .                   . .             s...side_vert, 0 means no split
    #    .   .                 .N2 .           S...shrunk_face
    #    .     .               .     .         Ni...new_faces
    #   s2      s1           s2|c2...s1|c1
    #    .        .            .     .  .
    #    .          .          . S .      .
    #    .            .        . .     N1    .
    #   c0...(s0=0)....c1    s0|c0...........c1
    #
    # pseudo-code:
    #   S = [s0|c0,s1|c1,s2|c2] example:[c0,s1,s2]
    #   split = side_vert!=0 example:[False,True,True]
    #   N0 = split[0]*[c0,s0,s2|c2] example:[0,0,0]
    #   N1 = split[1]*[c1,s1,s0|c0] example:[c1,s1,c0]
    #   N2 = split[2]*[c2,s2,s1|c1] example:[c2,s2,s1]

    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item()  # sync

    if S == 0:
        return vertices, faces

    edge_vert = torch.zeros_like(splits, dtype=torch.long)  # E
    edge_vert[splits] = torch.arange(
        V, V + S, dtype=torch.long, device=vertices.device
    )  # E 0 for no split, sync
    side_vert = edge_vert[face_to_edge]  # F,3 long, 0 for no split
    split_edges = edges[splits]  # S sync

    # vertices
    split_vertices = vertices[split_edges].mean(dim=1)  # S,3
    vertices = torch.concat((vertices, split_vertices), dim=0)

    # faces
    side_split = side_vert != 0  # F,3
    shrunk_faces = torch.where(side_split, side_vert, faces)  # F,3 long, 0 for no split
    new_faces = side_split[:, :, None] * torch.stack(
        (faces, side_vert, shrunk_faces.roll(1, dims=-1)), dim=-1
    )  # F,N=3,C=3
    faces = torch.concat((shrunk_faces, new_faces.reshape(F * 3, 3)))  # 4F,3
    if pack_faces:
        mask = faces[:, 0] != 0
        mask[0] = True
        faces = faces[mask]  # F',3 sync

    return vertices, faces


def collapse_edges(
    vertices: torch.Tensor,  # V,3 first unused
    faces: torch.Tensor,  # F,3 long 0 for unused
    edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
    priorities: torch.Tensor,  # E float
    stable: bool = False,  # only for unit testing
):  # (vertices,faces)
    V = vertices.shape[0]

    # check spacing
    _, order = priorities.sort(stable=stable)  # E
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vert_rank = torch.zeros(V, dtype=torch.long, device=vertices.device)  # V
    edge_rank = rank  # E
    for i in range(3):
        torch_scatter.scatter_max(
            src=edge_rank[:, None].expand(-1, 2).reshape(-1),
            index=edges.reshape(-1),
            dim=0,
            out=vert_rank,
        )
        edge_rank, _ = vert_rank[edges].max(dim=-1)  # E
    candidates = edges[(edge_rank == rank).logical_and_(priorities > 0)]  # E',2

    # check connectivity
    vert_connections = torch.zeros(V, dtype=torch.long, device=vertices.device)  # V
    vert_connections[candidates[:, 0]] = 1  # start
    edge_connections = vert_connections[edges].sum(dim=-1)  # E, edge connected to start
    vert_connections.scatter_add_(
        dim=0,
        index=edges.reshape(-1),
        src=edge_connections[:, None].expand(-1, 2).reshape(-1),
    )  # one edge from start
    vert_connections[candidates] = 0  # clear start and end
    edge_connections = vert_connections[edges].sum(
        dim=-1
    )  # E, one or two edges from start
    vert_connections.scatter_add_(
        dim=0,
        index=edges.reshape(-1),
        src=edge_connections[:, None].expand(-1, 2).reshape(-1),
    )  # one or two edges from start
    collapses = candidates[
        vert_connections[candidates[:, 1]] <= 2
    ]  # E" not more than two connections between start and end

    # mean vertices
    vertices[collapses[:, 0]] = vertices[collapses].mean(dim=1)  # TODO dim?

    # update faces
    dest = torch.arange(0, V, dtype=torch.long, device=vertices.device)  # V
    dest[collapses[:, 1]] = dest[collapses[:, 0]]
    faces = dest[faces]  # F,3 TODO optimize?
    c0, c1, c2 = faces.unbind(dim=-1)
    collapsed = (c0 == c1).logical_or_(c1 == c2).logical_or_(c0 == c2)
    faces[collapsed] = 0

    return vertices, faces


def calc_face_collapses(
    vertices: torch.Tensor,  # V,3 first unused
    faces: torch.Tensor,  # F,3 long, 0 for unused
    edges: torch.Tensor,  # E,2 long 0 for unused, lower vertex index first
    face_to_edge: torch.Tensor,  # F,3 long 0 for unused
    edge_length: torch.Tensor,  # E
    face_normals: torch.Tensor,  # F,3
    vertex_normals: torch.Tensor,  # V,3 first unused
    min_edge_length: torch.Tensor = None,  # V
    area_ratio=0.5,  # collapse if area < min_edge_length**2 * area_ratio
    shortest_probability=0.8,
):  # E edges to collapse
    E = edges.shape[0]
    F = faces.shape[0]

    # face flips
    ref_normals = calc_face_ref_normals(faces, vertex_normals, normalize=False)  # F,3
    face_collapses = (face_normals * ref_normals).sum(dim=-1) < 0  # F

    # small faces
    if min_edge_length is not None:
        min_face_length = min_edge_length[faces].mean(dim=-1)  # F
        min_area = min_face_length**2 * area_ratio  # F
        face_collapses.logical_or_(face_normals.norm(dim=-1) < min_area * 2)  # F
        face_collapses[0] = False

    # faces to edges
    face_length = edge_length[face_to_edge]  # F,3

    if shortest_probability < 1:
        # select shortest edge with shortest_probability chance
        randlim = round(2 / (1 - shortest_probability))
        rand_ind = torch.randint(0, randlim, size=(F,), device=faces.device).clamp_max_(2)  # selected edge local index in face
        sort_ind = torch.argsort(face_length, dim=-1, descending=True)  # F,3
        local_ind = sort_ind.gather(dim=-1, index=rand_ind[:, None])
    else:
        local_ind = torch.argmin(face_length, dim=-1)[:, None]  # F,1 0...2 shortest edge local index in face

    edge_ind = face_to_edge.gather(dim=1, index=local_ind)[:, 0]  # F 0...E selected edge global index
    edge_collapses = torch.zeros(E, dtype=torch.long, device=vertices.device)
    edge_collapses.scatter_add_(
        dim=0, index=edge_ind, src=face_collapses.long()
    )  # TODO legal for bool?

    return edge_collapses.bool()


def flip_edges(
    vertices: torch.Tensor,  # V,3 first unused
    faces: torch.Tensor,  # F,3 long, first must be 0, 0 for unused
    edges: torch.Tensor,  # E,2 long, first must be 0, 0 for unused, lower vertex index first
    edge_to_face: torch.Tensor,  # E,[left,right],[face,side]
    with_border: bool = True,  # handle border edges (D=4 instead of D=6)
    with_normal_check: bool = True,  # check face normal flips
    stable: bool = False,  # only for unit testing
):
    V = vertices.shape[0]
    E = edges.shape[0]
    device = vertices.device
    vertex_degree = torch.zeros(V, dtype=torch.long, device=device)  # V long
    vertex_degree.scatter_(dim=0, index=edges.reshape(E * 2), value=1, reduce="add")
    neighbor_corner = (edge_to_face[:, :, 1] + 2) % 3  # go from side to corner
    neighbors = faces[edge_to_face[:, :, 0], neighbor_corner]  # E,LR=2
    edge_is_inside = neighbors.all(dim=-1)  # E

    if with_border:
        # inside vertices should have D=6, border edges D=4, so we subtract 2 for all inside vertices
        # need to use float for masks in order to use scatter(reduce='multiply')
        vertex_is_inside = torch.ones(V, 2, dtype=torch.float32, device=vertices.device)  # V,2 float
        src = edge_is_inside.type(torch.float32)[:, None].expand(E, 2)  # E,2 float
        vertex_is_inside.scatter_(dim=0, index=edges, src=src, reduce="multiply")
        vertex_is_inside = vertex_is_inside.prod(dim=-1, dtype=torch.long)  # V long
        vertex_degree -= 2 * vertex_is_inside  # V long

    neighbor_degrees = vertex_degree[neighbors]  # E,LR=2
    edge_degrees = vertex_degree[edges]  # E,2
    #
    # loss = Sum_over_affected_vertices((new_degree-6)**2)
    # loss_change = Sum_over_neighbor_vertices((degree+1-6)**2-(degree-6)**2)
    #                   + Sum_over_edge_vertices((degree-1-6)**2-(degree-6)**2)
    #             = 2 * (2 + Sum_over_neighbor_vertices(degree) - Sum_over_edge_vertices(degree))
    #
    loss_change = 2 + neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1)  # E
    candidates = torch.logical_and(loss_change < 0, edge_is_inside)  # E
    loss_change = loss_change[candidates]  # E'
    if loss_change.shape[0] == 0:
        return

    edges_neighbors = torch.concat((edges[candidates], neighbors[candidates]), dim=-1)  # E',4
    _, order = loss_change.sort(descending=True, stable=stable)  # E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0, len(rank), device=rank.device)
    vertex_rank = torch.zeros((V, 4), dtype=torch.long, device=device)  # V,4
    torch_scatter.scatter_max(
        src=rank[:, None].expand(-1, 4), index=edges_neighbors, dim=0, out=vertex_rank
    )
    vertex_rank, _ = vertex_rank.max(dim=-1)  # V
    neighborhood_rank, _ = vertex_rank[edges_neighbors].max(dim=-1)  # E'
    flip = rank == neighborhood_rank  # E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |
        #   e0---->-cr
        v = vertices[edges_neighbors]  # E",4,3
        v = v - v[:, 0:1]  # make relative to e0
        e1 = v[:, 1]
        cl = v[:, 2]
        cr = v[:, 3]
        n = torch.cross(e1, cl) + torch.cross(cr, e1)  # sum of old normal vectors
        flip.logical_and_(
            torch.sum(n * torch.cross(cr, cl), dim=-1) > 0
        )  # first new face
        flip.logical_and_(
            torch.sum(n * torch.cross(cl - e1, cr - e1), dim=-1) > 0
        )  # second new face

    flip_edges_neighbors = edges_neighbors[flip]  # E",4
    flip_edge_to_face = edge_to_face[candidates, :, 0][flip]  # E",2
    flip_faces = flip_edges_neighbors[:, [[0, 3, 2], [1, 2, 3]]]  # E",2,3
    faces.scatter_(
        dim=0,
        index=flip_edge_to_face.reshape(-1, 1).expand(-1, 3),
        src=flip_faces.reshape(-1, 3),
    )


@torch.no_grad()
def remesh(
    vertices_etc: torch.Tensor,  # V,D
    faces: torch.Tensor,  # F,3 long
    min_edgelen: torch.Tensor,  # V
    max_edgelen: torch.Tensor,  # V
    flip: bool,
    max_vertices=1e6,
):
    # dummies
    vertices_etc, faces = prepend_dummies(vertices_etc, faces)
    vertices = vertices_etc[:, :3]  # V,3
    nan_tensor = torch.tensor([torch.nan], device=min_edgelen.device)
    min_edgelen = torch.concat((nan_tensor, min_edgelen))
    max_edgelen = torch.concat((nan_tensor, max_edgelen))

    # collapse
    edges, face_to_edge = calc_edges(faces)  # E,2 F,3
    edge_length = calc_edge_length(vertices, edges)  # E
    face_normals = calc_face_normals(vertices, faces, normalize=False)  # F,3
    vertex_normals = calc_vertex_normals(vertices, faces, face_normals)  # V,3
    face_collapse = calc_face_collapses(
        vertices,
        faces,
        edges,
        face_to_edge,
        edge_length,
        face_normals,
        vertex_normals,
        min_edgelen,
        area_ratio=0.5,
    )
    shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0)  # e[0,1] 0...ok, 1...edgelen=0
    priority = face_collapse.float() + shortness
    vertices_etc, faces = collapse_edges(vertices_etc, faces, edges, priority)

    # split
    if vertices.shape[0] < max_vertices:
        edges, face_to_edge = calc_edges(faces)  # E,2 F,3
        vertices = vertices_etc[:, :3]  # V,3
        edge_length = calc_edge_length(vertices, edges)  # E
        splits = edge_length > max_edgelen[edges].mean(dim=-1)
        vertices_etc, faces = split_edges(
            vertices_etc, faces, edges, face_to_edge, splits, pack_faces=False
        )

    vertices_etc, faces = pack(vertices_etc, faces)
    vertices = vertices_etc[:, :3]

    if flip:
        edges, _, edge_to_face = calc_edges(faces, with_edge_to_face=True)  # E,2 F,3
        flip_edges(vertices, faces, edges, edge_to_face, with_border=False)

    return remove_dummies(vertices_etc, faces)


def lerp_unbiased(a: torch.Tensor, b: torch.Tensor, weight: float, step: int):
    """lerp with adam's bias correction"""
    c_prev = 1 - weight ** (step - 1)
    c = 1 - weight**step
    a_weight = weight * c_prev / c
    b_weight = (1 - weight) / c
    a.mul_(a_weight).add_(b, alpha=b_weight)

@dataclass
class Options:
    # input mesh path
    mesh: str
    # training itersations
    iters: int = 300
    # output mesh path
    out_path: str = 'out.glb'

    ## params for rendering
    # render radius
    radius: float = 2.5
    # render fov 
    fovy: float = 49.1 
    # render resolution
    resolution: int = 512
    # nvdiffrast context
    force_cuda_rast: bool = False
    # use fixed 16 cameras and cache gt renders, faster
    fix_cameras: bool = True 

    ## params for remeshing
    # smallest and largest allowed reference edge length (controls final triangle sizes!)
    edge_len_lims: Tuple[float, ...] = (0.01, 0.15) 
    # learning rate
    lr: float = 0.3 
    # betas[0:2] are the same as in Adam, betas[2] may be used to time-smooth the relative velocity nu
    betas: Tuple[float, ...] = (0.8, 0.8, 0) 
    # optional spatial smoothing for m1,m2,nu, values between 0 (no smoothing) and 1 (max. smoothing)
    gammas: Tuple[float, ...] = (0, 0, 0) 
    # reference velocity for edge length controller
    nu_ref: float = 0.3 
    # edge length tolerance for split and collapse
    edge_len_tol: float = 0.5 
    # gain value for edge length controller
    gain: float = 0.2 
    # for laplacian smoothing/regularization
    laplacian_weight: float = 0.02 
    # learning rate ramp, actual ramp width is ramp/(1-betas[0])
    ramp: float = 1 
    # gradients are clipped to m1.abs()*grad_lim
    grad_lim: float = 10.0 
    # larger intervals are faster but with worse mesh quality
    remesh_interval: float = 1 
    # set to False to use a global scalar reference edge length instead    
    local_edgelen: bool = True

class Remesher(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        # hyper-parameters
        self.opt = opt
        self.with_gammas = any(g != 0 for g in self.opt.gammas)
        self.step = 0

        # only support cuda
        self.device = torch.device("cuda")

        # load input mesh
        self.mesh = Mesh.load(opt.mesh, bound=0.9, wotex=True, device=self.device)

        # init primitive (icosphere now)
        vertices, self.faces = self.init_primitives(level=2, bound=0.5)
        V = vertices.shape[0]

        # fixed persp
        self.proj = torch.from_numpy(get_perspective(opt.fovy)).float().to(self.device)

        # nvdiffrast context
        if not opt.force_cuda_rast:
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        # parameters
        self.vertices_etc = torch.zeros([V, 9], device=self.device)
        self.vertices = self.vertices_etc[:, :3]
        self.m2 = self.vertices_etc[:, 3]
        self.nu = self.vertices_etc[:, 4]
        self.m1 = self.vertices_etc[:, 5:8]
        self.ref_len = self.vertices_etc[:, 8]
        self.smooth = self.vertices_etc[:, :8] if self.with_gammas else self.vertices_etc[:, :3]
        
        self.vertices.copy_(vertices)
        self.vertices.requires_grad_()
        self.ref_len.fill_(opt.edge_len_lims[1])

    def init_primitives(self, level=2, bound=0.5):
        sphere = trimesh.creation.icosphere(subdivisions=level, radius=bound)
        verts = torch.tensor(sphere.vertices, device=self.device, dtype=torch.float32)
        faces = torch.tensor(sphere.faces, device=self.device, dtype=torch.long)
        return verts, faces

    @torch.no_grad()
    def render_gt(self, poses, resolution=512):
        v = self.mesh.v
        f = self.mesh.f.int()
        vn = self.mesh.vn * 0.5 + 0.5

        v_cam = F.pad(v, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0) @ torch.inverse(poses).permute(0, 2, 1)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (resolution, resolution))

        image, _ = dr.interpolate(vn.unsqueeze(0), rast, f)  # [1, H, W, 3]
        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous()  # [1, H, W, 1]

        image = torch.cat([image, alpha], dim=-1)  # [1, H, W, 4]
        image = dr.antialias(image, rast, v_clip, f)  # [1, H, W, 4]

        return image

    def render(self, poses, resolution=512):
        v = self.vertices
        f = self.faces

        face_normals = calc_face_normals(v, f, normalize=False)  # F,3
        vn = calc_vertex_normals(v, f, face_normals)  # V,3
        vn = vn * 0.5 + 0.5
        f = f.int()

        v_cam = F.pad(v, pad=(0, 1), mode="constant", value=1.0).unsqueeze(0) @ torch.inverse(poses).permute(0, 2, 1)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (resolution, resolution))

        image, _ = dr.interpolate(vn.unsqueeze(0), rast, f)  # [1, H, W, 3]
        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous()  # [1, H, W, 1]

        image = torch.cat([image, alpha], dim=-1)  # [1, H, W, 4]
        image = dr.antialias(image, rast, v_clip, f)  # [1, H, W, 4]

        return image

    @torch.no_grad()
    def optimize_step(self, eps=1e-8):
        self.step += 1

        # spatial smoothing
        edges, _ = calc_edges(self.faces)  # E,2
        E = edges.shape[0]
        edge_smooth = self.smooth[edges]  # E,2,S
        neighbor_smooth = torch.zeros_like(self.smooth)  # V,S
        torch_scatter.scatter_mean(
            src=edge_smooth.flip(dims=[1]).reshape(E * 2, -1),
            index=edges.reshape(E * 2, 1),
            dim=0,
            out=neighbor_smooth,
        )

        # apply optional smoothing of m1,m2,nu
        if self.opt.gammas[0]:
            self.m1.lerp_(neighbor_smooth[:, 5:8], self.opt.gammas[0])
        if self.opt.gammas[1]:
            self.m2.lerp_(neighbor_smooth[:, 3], self.opt.gammas[1])
        if self.opt.gammas[2]:
            self.nu.lerp_(neighbor_smooth[:, 4], self.opt.gammas[2])

        # add laplace smoothing to gradients
        laplace = self.vertices - neighbor_smooth[:, :3]
        grad = torch.addcmul(
            self.vertices.grad,
            laplace,
            self.nu[:, None],
            value=self.opt.laplacian_weight,
        )

        # gradient clipping
        if self.step > 1:
            grad_lim = self.m1.abs().mul_(self.opt.grad_lim)
            grad.clamp_(min=-grad_lim, max=grad_lim)

        # moment updates
        lerp_unbiased(self.m1, grad, self.opt.betas[0], self.step)
        lerp_unbiased(self.m2, (grad**2).sum(dim=-1), self.opt.betas[1], self.step)

        velocity = self.m1 / self.m2[:, None].sqrt().add_(eps)  # V,3
        speed = velocity.norm(dim=-1)  # V

        if self.opt.betas[2]:
            lerp_unbiased(self.nu, speed, self.opt.betas[2], self.step)  # V
        else:
            self.nu.copy_(speed)  # V

        # update vertices
        ramped_lr = self.opt.lr * min(1, self.step * (1 - self.opt.betas[0]) / self.opt.ramp)
        self.vertices.add_(velocity * self.ref_len[:, None], alpha=-ramped_lr)

        # update target edge length
        if self.step % self.opt.remesh_interval == 0:
            if self.opt.local_edgelen:
                len_change = 1 + (self.nu - self.opt.nu_ref) * self.opt.gain
            else:
                len_change = 1 + (self.nu.mean() - self.opt.nu_ref) * self.opt.gain
            self.ref_len *= len_change
            self.ref_len.clamp_(*self.opt.edge_len_lims)

    def remesh(self, flip: bool = True):
        min_edge_len = self.ref_len * (1 - self.opt.edge_len_tol)
        max_edge_len = self.ref_len * (1 + self.opt.edge_len_tol)

        self.vertices_etc, self.faces = remesh(self.vertices_etc, self.faces, min_edge_len, max_edge_len, flip)

        self.vertices = self.vertices_etc[:, :3]
        self.m2 = self.vertices_etc[:, 3]
        self.nu = self.vertices_etc[:, 4]
        self.m1 = self.vertices_etc[:, 5:8]
        self.ref_len = self.vertices_etc[:, 8]
        self.smooth = self.vertices_etc[:, :8] if self.with_gammas else self.vertices_etc[:, :3]

        self.vertices.requires_grad_()

    def get_cameras(self, n_ele=4, n_azi=4):

        if self.opt.fix_cameras:
            eles = np.linspace(-60, 60, n_ele)
            azis = np.linspace(-180, 180, n_azi)
        else:
            eles = np.random.randint(-89, 89, size=(n_ele,))
            azis = np.random.randint(-180, 180, size=(n_azi,))

        poses = []
        for ele in eles:
            for azi in azis:
                poses.append(orbit_camera(ele, azi, self.opt.radius))

        poses = np.stack(poses, axis=0)  # [N, 4, 4]
        poses = torch.from_numpy(poses.astype(np.float32)).to(self.device)

        return poses

    def fit(self, iters=300):

        if self.opt.fix_cameras:
            # cache gt renders
            poses = self.get_cameras()
            images_gt = self.render_gt(poses, self.opt.resolution)

        pbar = tqdm.trange(iters)
        for i in pbar:

            if not self.opt.fix_cameras:
                # random orbital camera pose
                poses = self.get_cameras()
                images_gt = self.render_gt(poses, self.opt.resolution)

            images = self.render(poses, self.opt.resolution)

            # loss
            loss = F.l1_loss(images, images_gt)
            loss.backward()

            # step
            self.optimize_step()

            # remesh
            self.remesh()

            # zero grad
            self.vertices.grad = None

            pbar.set_description(f"loss={loss.item():.6f} v={self.vertices.shape} f={self.faces.shape}")

    def export(self, path):
        mesh = Mesh(v=self.vertices, f=self.faces)
        mesh.write(path)


if __name__ == "__main__":
    opt = tyro.cli(Options)

    remesher = Remesher(opt)
    remesher.fit(opt.iters)
    remesher.export(opt.out_path)
