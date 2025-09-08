import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import triangulate as shapely_triangulate
from typing import Optional


def reconstruct_mesh_ball_pivoting(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    kdtreeRadius: float = 0.1,
    kdtreeMaxNN: int = 30,
    k: int = 100,
    radii: list = [0.05, 0.1, 0.2],
    visualize: bool = True
):
    """
    Reconstructs a mesh from a point cloud using Ball Pivoting.
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        kdtreeRadius (float): Radius for KDTree search. Defaults to 0.1.
        kdtreeMaxNN (int): Maximum number of nearest neighbors for KDTree search. Defaults to 30.
        k (int): Number of nearest neighbors to use for constructing the Riemannian graph. Defaults to 100.
        radii (list): List of ball radii to use. Defaults to [0.05, 0.1, 0.2].
        visualize (bool): If True, visualize the mesh. Defaults to True.
    Returns:
        o3d.geometry.TriangleMesh: The reconstructed mesh.
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


def find_boundary_loops(mesh: o3d.geometry.TriangleMesh) -> list:
    """
    Return ordered boundary loops as lists of vertex indices.
    Finds edges used by exactly one triangle and stitches them into loops.
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
    Compute centroid and 2 orthonormal basis vectors (u, v) spanning the best-fit plane.
    Returns centroid, u, v, normal.
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


def fill_hole_by_triangulation(
    mesh: o3d.geometry.TriangleMesh,
    loop: list,
    contour: np.ndarray = None,
    max_edge_len: float = None
) -> int:
    """
    Fill one boundary loop by projecting to best-fit plane and using Shapely triangulate.
    If `contour` (Nx3) is provided, triangles whose centroids fall outside that contour are discarded.
    Returns number of triangles added.
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
    Find all boundary loops and fill them using triangulation.
    contour: optional Nx3 polygon to clip the fill (projected into the hole plane).
    Returns number of triangles added across all holes.
    """
    loops = find_boundary_loops(mesh)
    added = 0
    for loop in loops:
        if len(loop) > max_hole_vertices:
            # skip extremely large holes unless explicitly desired
            continue
        added += fill_hole_by_triangulation(mesh, loop, contour=contour, max_edge_len=max_edge_len)
    return added


def stitch_loop_pair(mesh: o3d.geometry.TriangleMesh, loopA: list, loopB: list) -> int:
    """
    Naive stitching that connects closest vertex pairs between two loops to form a fan of triangles.
    Useful when a hole is actually two concentric loops that must be bridged.
    Returns number of triangles added.
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
    Repair holes in `mesh` using optional contour point cloud as clipping guide.

    Steps:
    - Optionally stitch pairs of nearby boundary loops (useful when a hole is represented
      by two concentric or close loops).
    - Fill remaining boundary loops by projecting each loop to a best-fit plane,
      triangulating the loop polygon and discarding triangles outside the supplied contour.

    Args:
        mesh: Open3D TriangleMesh to repair (modified in-place and returned).
        contour_pcd: optional Open3D PointCloud containing contour points (Nx3). If provided,
            the filling will be clipped to this contour projected into each hole plane.
        stitch_pairs: whether to attempt stitching of nearby loop pairs before filling.
        stitch_dist: maximum distance between loop centroids to consider stitching. If None,
            a heuristic based on mesh edge lengths is used.
        max_hole_vertices: skip extremely large holes with more than this many boundary verts.
        max_edge_len: maximum allowed edge length within a hole; if exceeded the hole is skipped.

    Returns:
        The repaired Open3D TriangleMesh (same object instance, also returned for convenience).
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
        loops = find_boundary_loops(mesh)
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
