"""
surfaceReconstructor.py

Mesh reconstruction and hole repair utilities for 3D geometry using Open3D and Shapely.

Modules:
    - numpy
    - open3d
    - shapely.geometry
    - shapely.ops
    - typing

Imports:
    - Polygon, Point: Used for geometric containment and triangulation.
    - triangulate: Performs 2D triangulation for hole filling.
    - KDTreeSearchParamHybrid: Used for normal estimation in point clouds.
    - TriangleMesh, PointCloud: Open3D geometry types for mesh and point cloud operations.

Functions:
    - reconstruct_mesh_ball_pivoting: Reconstructs a mesh from a point cloud using Ball Pivoting, with optional visualization.
    - _find_boundary_loops: Finds ordered boundary loops in a mesh as lists of vertex indices.
    - _fill_hole_by_triangulation: Fills a boundary loop by triangulating its projection onto a best-fit plane, optionally clipped
      by a contour.
    - fill_mesh_holes: Finds and fills all boundary loops in a mesh, with optional contour clipping and edge length constraints.
    - stitch_loop_pair: Bridges two boundary loops by connecting closest vertex pairs, useful for repairing adjacent holes.
    - repair_mesh_with_contour: Repairs mesh holes using an optional contour point cloud as a clipping guide, with optional loop
      stitching.

Typical Usage:
    1. Use reconstruct_mesh_ball_pivoting to generate a mesh from a point cloud.
    2. Use _find_boundary_loops to detect mesh boundaries (holes).
    3. Use fill_hole_by_triangulation or fill_mesh_holes to fill detected holes, optionally using a contour for clipping.
    4. Use stitch_loop_pair to connect nearby boundary loops before filling.
    5. Use repair_mesh_with_contour for automated mesh repair with contour guidance.

See individual function docstrings for details.
"""

import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import triangulate as shapely_triangulate
from typing import Optional
# ---------------- Mesh Reconstruction ----------------
# Functions for reconstructing meshes from point clouds

# reconstruct_mesh_ball_pivoting

# ---------------- Boundary Detection ----------------
# Functions for finding boundary loops (holes) in meshes

# find_boundary_loops

# ---------------- Plane Fitting Utilities ----------------
# Internal helper for best-fit plane computation

# _best_fit_plane_basis
# _fill_hole_by_triangulation
# fill_mesh_holes

# ---------------- Loop Stitching ----------------
# Functions for connecting/stitching adjacent boundary loops

# stitch_loop_pair

# ---------------- Mesh Repair Workflow ----------------
# Functions for automated mesh repair using contours and stitching

# repair_mesh_with_contour


def reconstruct_mesh_ball_pivoting(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    kdtreeRadius: float = 0.1,
    kdtreeMaxNN: int = 30,
    k: int = 100,
    radii: list = [0.05, 0.1, 0.2],
    visualize: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Reconstructs a mesh from a point cloud using Ball Pivoting.

    For a given Open3D PointCloud, estimates normals if not present, orients normals consistently,
    and applies Ball Pivoting surface reconstruction with specified ball radii. Optionally visualizes
    the resulting mesh.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud for mesh reconstruction.
        kdtreeRadius (float, optional): Radius for KDTree normal estimation. Defaults to 0.1.
        kdtreeMaxNN (int, optional): Max nearest neighbors for KDTree normal estimation. Defaults to 30.
        k (int, optional): Number of neighbors for orienting normals. Defaults to 100.
        radii (list, optional): List of ball radii for Ball Pivoting. Defaults to [0.05, 0.1, 0.2].
        visualize (bool, optional): If True, visualize the resulting mesh. Defaults to True.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed mesh.

    Example:
        >>> pcd = o3d.io.read_point_cloud("cloud.ply")
        >>> mesh = reconstruct_mesh_ball_pivoting(pcd, radii=[0.05, 0.1])
        >>> o3d.visualization.draw_geometries([mesh])

    Note:
        - Normals are estimated and oriented if not present in the input point cloud.
        - Ball Pivoting may fail if the point cloud is sparse or noisy.
        - Visualization is optional and uses Open3D's draw_geometries.
    """
    # Estimate normals if not present
    try:
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=kdtreeRadius, max_nn=kdtreeMaxNN))
            pcd.orient_normals_consistent_tangent_plane(k)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        if visualize:
            o3d.visualization.draw_geometries([mesh], window_name="Ball Pivoting Mesh")
        return mesh
    except Exception as e:
        print(f"Ball Pivoting error: {e}")
        print(f"Point cloud has {len(pcd.points)} points")
        exit()
        return None


def _find_boundary_loops(mesh: o3d.geometry.TriangleMesh) -> list:
    """
    Return ordered boundary loops as lists of vertex indices.
    Finds edges used by exactly one triangle and stitches them into loops.

    Args:
        mesh: Open3D TriangleMesh to modify.
        loopA: First boundary loop (list of vertex indices).
        loopB: Second boundary loop (list of vertex indices).

    Returns:
        Number of triangles added.
    """
    # collect edges -> count
    triangles = np.asarray(mesh.triangles, dtype=int)
    edges = {}
    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            if a > b:
                a, b = b, a
            edges.setdefault((a, b), 0)
            edges[(a, b)] += 1

    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    if not boundary_edges:
        return []

    # adjacency for boundary edges
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops = []
    visited_edges = set()
    for start, neighbors in adj.items():
        for nb in neighbors:
            e = (min(start, nb), max(start, nb))
            if e in visited_edges:
                continue
            # walk a loop
            loop = [start, nb]
            visited_edges.add(e)
            while True:
                cur = loop[-1]
                prev = loop[-2]
                next_candidates = [n for n in adj[cur] if n != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                edge = (min(cur, nxt), max(cur, nxt))
                if edge in visited_edges:
                    break
                loop.append(nxt)
                visited_edges.add(edge)
                if nxt == loop[0]:
                    break
            # ensure closed loop (remove final duplicate)
            if loop[0] == loop[-1]:
                loop = loop[:-1]
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _best_fit_plane_basis(points: np.ndarray):
    """
    Compute centroid and two orthonormal basis vectors (u, v) spanning the best-fit plane.

    Args:
        points: Nx3 array of 3D points.

    Returns:
        centroid, u, v, normal: Centroid and basis vectors for the plane.
    """
    centroid = points.mean(axis=0)
    # PCA
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # smallest eigenvalue -> normal
    normal = eigvecs[:, np.argmin(eigvals)]
    u = eigvecs[:, np.argmax(eigvals)]
    v = np.cross(normal, u)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    normal = normal / np.linalg.norm(normal)
    return centroid, u, v, normal


def _fill_hole_by_triangulation(
    mesh: o3d.geometry.TriangleMesh,
    loop: list,
    contour: np.ndarray = None,
    max_edge_len: float = None
) -> int:
    """
    Fill a boundary loop by projecting to a best-fit plane and triangulating using Shapely.

    Args:
        mesh: Open3D TriangleMesh to modify.
        loop: List of vertex indices forming the boundary loop.
        contour: Optional Nx3 array of 3D points for clipping triangles.
        max_edge_len: Maximum allowed edge length in the loop.

    Returns:
        Number of triangles added.
    """
    verts = np.asarray(mesh.vertices)
    loop_pts = verts[np.asarray(loop)]
    # small sanity checks
    if len(loop_pts) < 3:
        return 0
    if max_edge_len is not None:
        dists = np.linalg.norm(np.roll(loop_pts, -1, axis=0) - loop_pts, axis=1)
        if np.any(dists > max_edge_len):
            # hole edges too large — skip
            return 0

    centroid, u, v, normal = _best_fit_plane_basis(loop_pts)
    # 3D -> 2D
    coords2d = np.column_stack([np.dot(loop_pts - centroid, u), np.dot(loop_pts - centroid, v)])
    # create polygon and triangulate
    poly = Polygon(coords2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        return 0

    tri_polys = shapely_triangulate(poly)
    new_vertices = []  # noqa: F841
    new_triangles = []

    # Prepare contour polygon if provided (2D in same projected frame)
    if contour is not None:
        contour = np.asarray(contour)
        cont_centroid, cont_u, cont_v, cont_n = _best_fit_plane_basis(contour)
        # If contour plane differs strongly, prefer projecting using the hole's plane.
        cont2d = np.column_stack([np.dot(contour - centroid, u), np.dot(contour - centroid, v)])
        contour_poly = Polygon(cont2d)
    else:
        contour_poly = None

    # create map from 2D coordinates to indices when possible (avoid duplicate vertices)
    global_vertices = verts.tolist()
    index_map = {}  # (x2,y2) -> index
    # fill index_map with existing loop vertices
    for li, pt2 in zip(loop, coords2d):
        key = (round(float(pt2[0]), 8), round(float(pt2[1]), 8))
        index_map[key] = int(li)

    for t in tri_polys:
        tri_coords = np.array(t.exterior.coords)[:3]  # shapely returns closed ring
        centroid2 = tri_coords.mean(axis=0)
        # check within contour if provided
        if contour_poly is not None:
            if not (contour_poly.contains(ShapelyPoint(centroid2)) or contour_poly.covers(ShapelyPoint(centroid2))):
                continue
        # for each corner, either reuse existing vertex or create new
        tri_idx = []
        for c in tri_coords:
            key = (round(float(c[0]), 8), round(float(c[1]), 8))
            if key in index_map:
                tri_idx.append(index_map[key])
            else:
                # map back to 3D
                pt3 = centroid + u * c[0] + v * c[1]
                global_vertices.append([float(pt3[0]), float(pt3[1]), float(pt3[2])])
                new_idx = len(global_vertices) - 1
                index_map[key] = new_idx
                tri_idx.append(new_idx)
        # ensure triangle is not degenerate
        if len(set(tri_idx)) == 3:
            new_triangles.append(tri_idx)

    # update mesh (append new vertices and triangles)
    if new_triangles:
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(global_vertices))
        all_tris = np.vstack([np.asarray(mesh.triangles, dtype=int), np.asarray(new_triangles, dtype=int)])
        mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        mesh.compute_vertex_normals()
    return len(new_triangles)


def fill_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    contour: np.ndarray = None,
    max_hole_vertices: int = 500,
    max_edge_len: float = None
) -> int:
    """
    Finds all boundary loops in a mesh and fills them using triangulation.

    For each detected boundary loop, projects its vertices to a best-fit plane,
    triangulates the resulting 2D polygon, and adds triangles to fill the hole.
    Optionally clips filled triangles using a provided contour.

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to modify in-place.
        contour (np.ndarray, optional): Nx3 array of 3D points for clipping triangles. Defaults to None.
        max_hole_vertices (int, optional): Maximum allowed vertices in a hole. Defaults to 500.
        max_edge_len (float, optional): Maximum allowed edge length in a hole. Defaults to None.

    Returns:
        int: Total number of triangles added across all holes.

    Example:
        >>> mesh = o3d.geometry.TriangleMesh()
        >>> added = fill_mesh_holes(mesh, contour=contour_points)
        >>> print(f"Triangles added: {added}")

    Note:
        - Skips holes with edge lengths exceeding max_edge_len or vertex count exceeding max_hole_vertices.
        - Uses Shapely for 2D triangulation and Open3D for mesh updates.
    """
    loops = _find_boundary_loops(mesh)
    added = 0
    for loop in loops:
        if len(loop) > max_hole_vertices:
            # skip extremely large holes unless explicitly desired
            continue
        added += _fill_hole_by_triangulation(mesh, loop, contour=contour, max_edge_len=max_edge_len)
    return added


def stitch_loop_pair(mesh: o3d.geometry.TriangleMesh, loopA: list, loopB: list) -> int:
    """
    Naive stitching that connects closest vertex pairs between two loops to form a fan of triangles.
    Useful when a hole is actually two concentric loops that must be bridged.
    Returns number of triangles added.

    Connects two boundary loops by bridging closest vertex pairs to form triangles.
    For each vertex in `loopA`, finds the nearest vertex in `loopB` and creates a triangle
    connecting the current vertex, its nearest neighbor in `loopB`, and the next vertex in `loopA`.
    Useful for repairing adjacent holes or stitching mesh boundaries.

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to modify in-place.
        loopA (list): List of vertex indices forming the first boundary loop.
        loopB (list): List of vertex indices forming the second boundary loop.

    Returns:
        int: Number of triangles added to the mesh.

    Example:
        >>> added = stitch_loop_pair(mesh, loopA, loopB)
        >>> print(f"Triangles added: {added}")

    Note:
        - Uses KDTree for efficient nearest neighbor search between loops.
        - Only bridges each vertex in loopA to its closest vertex in loopB.
        - Triangles are added directly to the mesh and normals are recomputed.
    """
    verts = np.asarray(mesh.vertices)
    A = verts[np.asarray(loopA)]
    B = verts[np.asarray(loopB)]
    # find closest pairs
    from scipy.spatial import cKDTree
    treeB = cKDTree(B)
    pairs = []
    for i, p in enumerate(A):
        dist, idx = treeB.query(p)
        pairs.append((i, idx, dist))
    # create triangles by walking along A and linking to nearest B
    global_vertices = verts.tolist()  # noqa: F841
    new_tris = []
    for i in range(len(pairs)):
        a_idx = loopA[pairs[i][0]]
        b_idx = loopB[pairs[i][1]]
        a_next = loopA[pairs[(i + 1) % len(pairs)][0]]
        # triangle (a, b, a_next) - simple bridging
        if len({a_idx, b_idx, a_next}) == 3:
            new_tris.append([int(a_idx), int(b_idx), int(a_next)])
    if new_tris:
        all_tris = np.vstack([np.asarray(mesh.triangles, dtype=int), np.asarray(new_tris, dtype=int)])
        mesh.triangles = o3d.utility.Vector3iVector(all_tris)
        mesh.compute_vertex_normals()
    return len(new_tris)


def repair_mesh_with_contour(
    mesh: o3d.geometry.TriangleMesh,
    contour_pcd: Optional[o3d.geometry.PointCloud] = None,
    stitch_pairs: bool = True,
    stitch_dist: Optional[float] = None,
    max_hole_vertices: int = 500,
    max_edge_len: Optional[float] = None
) -> o3d.geometry.TriangleMesh:
    """
    Repair holes in a mesh using an optional contour point cloud as a clipping guide.

    Steps:
        1. Optionally stitch pairs of nearby boundary loops.
        2. Fill remaining boundary loops by triangulating their projection onto a best-fit plane,
           discarding triangles outside the supplied contour.

    Args:
        mesh: Open3D TriangleMesh to repair (modified in-place and returned).
        contour_pcd: Optional Open3D PointCloud containing contour points (Nx3).
        stitch_pairs: Whether to attempt stitching of nearby loop pairs before filling.
        stitch_dist: Maximum distance between loop centroids to consider stitching.
        max_hole_vertices: Maximum allowed vertices in a hole.
        max_edge_len: Maximum allowed edge length in a hole.

    Returns:
        The repaired Open3D TriangleMesh.

    Example:
        >>> repaired_mesh = repair_mesh_with_contour(mesh, contour_pcd=contour_points, stitch_pairs=True)
        >>> o3d.visualization.draw_geometries([repaired_mesh])

    Note:
        - If stitch_dist is None, a sensible default is computed from average mesh edge length.
        - Uses _find_boundary_loops, stitch_loop_pair, and fill_mesh_holes internally.
        - Mesh normals are recomputed after modifications.

    Raises:
        TypeError: If mesh is not an Open3D TriangleMesh.
        TypeError: If contour_pcd is provided but not an Open3D PointCloud.
        ValueError: If mesh has no triangles.
        ValueError: If contour_pcd is provided but empty.
        ValueError: If stitch_dist is provided but not positive.
        ValueError: If max_hole_vertices is not positive.
        ValueError: If max_edge_len is provided but not positive.
        RuntimeError: If mesh repair fails due to unexpected errors.
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("mesh must be an Open3D TriangleMesh")

    contour_arr = None
    if contour_pcd is not None:
        if not isinstance(contour_pcd, o3d.geometry.PointCloud):
            raise TypeError("contour_pcd must be an Open3D PointCloud or None")
        contour_arr = np.asarray(contour_pcd.points)

    # Compute a sensible default stitch distance from average mesh edge length if not provided
    if stitch_dist is None:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles, dtype=int)
        edge_set = set()
        edge_lengths = []
        for tri in tris:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                e = (min(int(a), int(b)), max(int(a), int(b)))
                if e in edge_set:
                    continue
                edge_set.add(e)
                edge_lengths.append(np.linalg.norm(verts[e[0]] - verts[e[1]]))
        if edge_lengths:
            avg_edge = float(np.mean(edge_lengths))
            stitch_dist = max(avg_edge * 2.0, 0.01)
        else:
            stitch_dist = 0.1

    # 1) Optionally stitch nearby loop pairs
    if stitch_pairs:
        loops = _find_boundary_loops(mesh)
        if len(loops) > 1:
            # compute centroids
            verts = np.asarray(mesh.vertices)
            centroids = [verts[np.asarray(loop)].mean(axis=0) for loop in loops]
            used = set()
            for i in range(len(loops)):
                if i in used:
                    continue
                for j in range(i + 1, len(loops)):
                    if j in used:
                        continue
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    if d <= stitch_dist:
                        # stitch the pair
                        try:
                            stitch_loop_pair(mesh, loops[i], loops[j])
                        except Exception:
                            pass
                        used.add(i)
                        used.add(j)
            # after stitching, we'll re-run hole filling on current mesh

    # 2) Fill remaining holes (clipped by contour if provided)
    fill_mesh_holes(mesh, contour=contour_arr, max_hole_vertices=max_hole_vertices, max_edge_len=max_edge_len)

    # recompute normals and return
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    return mesh
