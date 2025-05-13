import torch
import numpy as np

from kiui.op import safe_normalize

import pymeshlab as pml
from importlib.metadata import version

PML_VER = version('pymeshlab') 

# the code assumes the latest 2023.12 version, but we can patch older versions
if PML_VER.startswith('0.2'):
    # monkey patch for 0.2 (only the used functions in this file!)
    pml.MeshSet.meshing_decimation_quadric_edge_collapse = pml.MeshSet.simplification_quadric_edge_collapse_decimation
    pml.MeshSet.meshing_isotropic_explicit_remeshing = pml.MeshSet.remeshing_isotropic_explicit_remeshing
    pml.MeshSet.meshing_remove_unreferenced_vertices = pml.MeshSet.remove_unreferenced_vertices
    pml.MeshSet.meshing_merge_close_vertices = pml.MeshSet.merge_close_vertices
    pml.MeshSet.meshing_remove_duplicate_faces = pml.MeshSet.remove_duplicate_faces
    pml.MeshSet.meshing_remove_null_faces = pml.MeshSet.remove_zero_area_faces
    pml.MeshSet.meshing_remove_connected_component_by_diameter = pml.MeshSet.remove_isolated_pieces_wrt_diameter
    pml.MeshSet.meshing_remove_connected_component_by_face_number = pml.MeshSet.remove_isolated_pieces_wrt_face_num
    pml.MeshSet.meshing_repair_non_manifold_edges = pml.MeshSet.repair_non_manifold_edges_by_removing_faces
    pml.MeshSet.meshing_repair_non_manifold_vertices = pml.MeshSet.repair_non_manifold_vertices_by_splitting
    pml.PercentageValue = pml.Percentage
    pml.PureValue = float
elif PML_VER.startswith('2022.2'):
    # monkey patch for 2022.2
    pml.PercentageValue = pml.Percentage
    pml.PureValue = pml.AbsoluteValue


def decimate_mesh(
    verts, faces, target=5e4, backend="pymeshlab", remesh=False, optimalplacement=True, verbose=True
):
    """ perform mesh decimation.

    Args:
        verts (np.ndarray): mesh vertices, float [N, 3]
        faces (np.ndarray): mesh faces, int [M, 3]
        target (int): targeted number of faces
        backend (str, optional): algorithm backend, can be "pymeshlab" or "pyfqmr". Defaults to "pymeshlab".
        remesh (bool, optional): whether to remesh after decimation. Defaults to False.
        optimalplacement (bool, optional): For flat mesh, use False to prevent spikes. Defaults to True.
        verbose (bool, optional): whether to print the decimation process. Defaults to True.

    Returns:
        Tuple[np.ndarray]: vertices and faces after decimation.
    """

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        m.compact()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    if verbose:
        print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=0,
    min_d=0,
    repair=False,
    remesh=False,
    remesh_size=0.01,
    remesh_iters=3,
    verbose=True,
):
    """ perform mesh cleaning, including floater removal, non manifold repair, and remeshing.

    Args:
        verts (np.ndarray): mesh vertices, float [N, 3]
        faces (np.ndarray): mesh faces, int [M, 3]
        v_pct (int, optional): percentage threshold to merge close vertices. Defaults to 1.
        min_f (int, optional): maximal number of faces for isolated component to remove. Defaults to 0.
        min_d (int, optional): maximal diameter percentage of isolated component to remove. Defaults to 0.
        repair (bool, optional): whether to repair non-manifold faces (cannot gurantee). Defaults to True.
        remesh (bool, optional): whether to perform a remeshing after all cleaning. Defaults to True.
        remesh_size (float, optional): the targeted edge length for remeshing. Defaults to 0.01.
        remesh_iters (int, optional): the iterations of remeshing. Defaults to 3.
        verbose (bool, optional): whether to print the cleaning process. Defaults to True.

    Returns:
        Tuple[np.ndarray]: vertices and faces after decimation.
    """
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    # be careful: may lead to strangely missing triangles...
    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=remesh_iters, targetlen=pml.PureValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    m.compact()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    if verbose:
        print(f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def normalize_mesh(vertices, bound=0.95):
    """ normalize the mesh vertices to a unit cube.

    Args:
        vertices (np.ndarray): mesh vertices, float [N, 3]
        bound (float, optional): the bounding box size. Defaults to 0.95.
    
    Returns:
        np.ndarray: normalized vertices.
    """
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices

### mesh related losses

def laplacian_uniform(verts, faces):
    """ calculate laplacian uniform matrix

    Args:
        verts (torch.Tensor): mesh vertices, float [N, 3]
        faces (torch.Tensor): mesh faces, long [M, 3]

    Returns:
        torch.Tensor: sparse laplacian matrix.
    """

    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()


def laplacian_smooth_loss(verts, faces):
    """ calculate laplacian smooth loss.

    Args:
        verts (torch.Tensor): mesh vertices, float [N, 3]
        faces (torch.Tensor): mesh faces, int [M, 3]

    Returns:
        torch.Tensor: loss value.
    """
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss

@torch.no_grad()
def compute_edge_to_face_mapping(faces):
    """ compute edge to face mapping.

    Args:
        faces (torch.Tensor): mesh faces, int [M, 3]

    Returns:
        torch.Tensor: indices to faces for each edge, long, [N, 2]
    """
    # Get unique edges
    # Create all edges, packed by triangle
    all_edges = torch.cat((
        torch.stack((faces[:, 0], faces[:, 1]), dim=-1),
        torch.stack((faces[:, 1], faces[:, 2]), dim=-1),
        torch.stack((faces[:, 2], faces[:, 0]), dim=-1),
    ), dim=-1).view(-1, 2)

    # Swap edge order so min index is always first
    order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
    sorted_edges = torch.cat((
        torch.gather(all_edges, 1, order),
        torch.gather(all_edges, 1, 1 - order)
    ), dim=-1)

    # Elliminate duplicates and return inverse mapping
    unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

    tris = torch.arange(faces.shape[0]).repeat_interleave(3).cuda()

    tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

    # Compute edge to face table
    mask0 = order[:,0] == 0
    mask1 = order[:,0] == 1
    tris_per_edge[idx_map[mask0], 0] = tris[mask0]
    tris_per_edge[idx_map[mask1], 1] = tris[mask1]

    return tris_per_edge


def normal_consistency(verts, faces, face_normals=None):
    """ calculate normal consistency loss.

    Args:
        verts (torch.Tensor): mesh vertices, float [N, 3]
        faces (torch.Tensor): mesh faces, int [M, 3]
        face_normals (Optional[torch.Tensor]): the normal vector for each face, will be calculated if not provided, float [M, 3]

    Returns:
        torch.Tensor: loss value.
    """

    if face_normals is None:
        
        i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals)

    tris_per_edge = compute_edge_to_face_mapping(faces)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))
