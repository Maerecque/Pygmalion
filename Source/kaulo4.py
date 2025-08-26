# Wharf Cellar: Robust Ridge & Skeleton Extraction (Memory-Safe, Single File)
# -------------------------------------------------------------------------
# What this script does
# - Segments planes (floor & walls) with Open3D RANSAC (optionally on a voxel-downsampled copy)
# - Extracts wall/floor ridges from plane intersections and local PCA
# - Estimates the roof barrel-vault centerline ("apex") from non-planar points
# - Builds a skeleton (polylines + simple node/edge graph)
# - Clips the skeleton to a user-supplied XZ contour via a **concave hull** (alpha shape)
# - Stays memory-safe: optional quantized downsampling for contour before alpha shape, no NxN matrices
#
# How to run
#   python wharfcellar_skeleton.py \
#       --points wharf_cellar.npy \
#       --contour wharf_cellar_contour.npy \
#       --voxel 0.05 \
#       --alpha 0.25 \
#       --max-contour-points 15000 \
#       --contour-grid 0.05
#
# Required packages
#   pip install open3d shapely scipy numpy
# (matplotlib only if you want to visualize in 2D debug plots)

from __future__ import annotations
import argparse
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# --- Third-party deps ---
try:
    import open3d as o3d
except Exception as e:
    o3d = None
    print("[WARN] open3d not available. Install with `pip install open3d`.", file=sys.stderr)

from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay


# =============================================================================
# Utility helpers
# =============================================================================

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def pca_directions(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(pts, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, float)
    _, V = pca_directions(pts)
    return pts.mean(axis=0), unit(V[:, 0])


def point_line_distance(points: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = unit(v)
    diff = points - p0
    cross = np.cross(diff, v)
    return np.linalg.norm(cross, axis=1)


# =============================================================================
# Open3D conversions
# =============================================================================

def to_o3d_pcd(points: np.ndarray) -> 'o3d.geometry.PointCloud':
    if o3d is None:
        raise RuntimeError("open3d is required for point cloud ops.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    return pcd


def from_o3d_pcd(pcd: 'o3d.geometry.PointCloud') -> np.ndarray:
    return np.asarray(pcd.points)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return np.asarray(points, float)
    pcd = to_o3d_pcd(points)
    pcd_ds = pcd.voxel_down_sample(voxel)
    return from_o3d_pcd(pcd_ds)


# =============================================================================
# Plane segmentation (RANSAC)
# =============================================================================
@dataclass
class Plane:
    n: np.ndarray  # normal
    d: float       # plane offset in ax+by+cz+d=0
    inliers: np.ndarray  # indices into the original array


def segment_multiple_planes(points: np.ndarray,
                            max_planes: int = 3,
                            distance_threshold: float = 0.02,
                            ransac_n: int = 3,
                            num_iterations: int = 2000,
                            min_points: int = 1000,
                            voxel_for_seg: float = 0.0) -> List[Plane]:
    """Iteratively segment planes on a (possibly) downsampled copy to stay fast.
       Returns planes with inlier indices mapped to the **original** points.
    """
    if o3d is None:
        raise RuntimeError("open3d required for RANSAC plane segmentation.")

    pts = np.asarray(points, float)
    if voxel_for_seg > 0:
        pts_ds = voxel_downsample(pts, voxel_for_seg)
        # Build a KDTree to map back to original indices
        pcd_full = to_o3d_pcd(pts)
        kdtree = o3d.geometry.KDTreeFlann(pcd_full)
        pts_for_seg = pts_ds
        map_idx = []
        for p in pts_ds:
            _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
            map_idx.append(idx[0])
        map_idx = np.asarray(map_idx, int)
    else:
        pts_for_seg = pts
        map_idx = np.arange(len(pts))

    remaining_local = np.arange(len(pts_for_seg))
    planes: List[Plane] = []

    for _ in range(max_planes):
        if len(remaining_local) < min_points:
            break
        pcd = to_o3d_pcd(pts_for_seg[remaining_local])
        model, inliers_local = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        a, b, c, d = model
        n = unit(np.array([a, b, c], float))
        inliers_local = np.asarray(inliers_local, int)
        if len(inliers_local) < min_points:
            break
        # Map inliers back to original indices
        inliers_global = map_idx[remaining_local[inliers_local]]
        planes.append(Plane(n=n, d=float(d), inliers=inliers_global))
        # Remove local inliers for next round
        keep_mask = np.ones(len(remaining_local), dtype=bool)
        keep_mask[inliers_local] = False
        remaining_local = remaining_local[keep_mask]

    planes.sort(key=lambda pl: len(pl.inliers), reverse=True)
    return planes


# =============================================================================
# Plane intersections -> ridge lines
# =============================================================================
@dataclass
class Line:
    p: np.ndarray  # point on line
    v: np.ndarray  # direction (unit)


def intersect_planes(p1: Plane, p2: Plane) -> Optional[Line]:
    n1, d1 = p1.n, p1.d
    n2, d2 = p2.n, p2.d
    v = np.cross(n1, n2)
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return None
    v = v / nv
    A = np.vstack([n1, n2, v])
    b = -np.array([d1, d2, 0.0])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return Line(p=p, v=v)


def extract_points_near_line(points: np.ndarray, line: Line, tol: float = 0.03) -> np.ndarray:
    d = point_line_distance(points, line.p, line.v)
    return points[d <= tol]


# =============================================================================
# Roof barrel-vault axis (middle ridge)
# =============================================================================
@dataclass
class RoofAxis:
    axis_point: np.ndarray
    axis_dir: np.ndarray
    radius: float


def estimate_roof_axis_from_points(roof_pts: np.ndarray) -> RoofAxis:
    if len(roof_pts) < 100:
        raise ValueError("Not enough roof points to estimate axis.")
    p0, v = fit_line_pca(roof_pts)
    r = np.median(point_line_distance(roof_pts, p0, v))
    return RoofAxis(axis_point=p0, axis_dir=unit(v), radius=float(r))


def sample_apex_polyline(axis: RoofAxis, bounds: Tuple[np.ndarray, np.ndarray], step: float = 0.1) -> np.ndarray:
    mn, mx = bounds
    # project bbox corners to parameter t along axis
    corners = np.array([[mn[0], mn[1], mn[2]],
                        [mx[0], mn[1], mn[2]],
                        [mn[0], mx[1], mn[2]],
                        [mn[0], mn[1], mx[2]],
                        [mx[0], mx[1], mn[2]],
                        [mx[0], mn[1], mx[2]],
                        [mn[0], mx[1], mx[2]],
                        [mx[0], mx[1], mx[2]]])
    ts = [np.dot(c - axis.axis_point, axis.axis_dir) for c in corners]
    tmin, tmax = float(np.min(ts)), float(np.max(ts))
    num = max(2, int(math.ceil((tmax - tmin) / max(step, 1e-3))))
    T = np.linspace(tmin, tmax, num)
    return axis.axis_point[None, :] + T[:, None] * axis.axis_dir[None, :]


# =============================================================================
# Concave hull (alpha shape) with memory-safe downsampling & jittered Delaunay
# =============================================================================

def quantized_unique(points2d: np.ndarray, grid: float) -> np.ndarray:
    """Return an index mask selecting one point per XZ grid cell of size `grid`.
    No large pairwise matrices; uses quantization + numpy unique."""
    if grid <= 0:
        return np.arange(len(points2d))
    qx = np.floor(points2d[:, 0] / grid).astype(np.int64)
    qz = np.floor(points2d[:, 1] / grid).astype(np.int64)
    keys = qx.astype(np.int64) << 32 | (qz.astype(np.int64) & 0xffffffff)
    _, idx = np.unique(keys, return_index=True)
    return np.sort(idx)


def safe_delaunay(points2d: np.ndarray) -> Tuple[Optional[Delaunay], np.ndarray]:
    """Try robust Delaunay with small jitters to avoid Qhull precision errors."""
    for jitter in [0.0, 1e-6, 1e-5, 1e-4]:
        try:
            pts = points2d if jitter == 0.0 else points2d + np.random.normal(scale=jitter, size=points2d.shape)
            tri = Delaunay(pts, qhull_options='QJ')  # QJ = joggle input
            return tri, pts
        except Exception:
            continue
    return None, points2d


def alpha_shape_polygon(points2d: np.ndarray, alpha: float,
                        max_points: int = 20000,
                        grid: float = 0.05) -> Polygon:
    """Compute a concave hull polygon (alpha shape) in 2D (XZ), memory-safe.
    - Optional quantized downsampling to `max_points`
    - Robust Delaunay with jitter fallback
    - Clean fallback to convex hull when needed
    """
    pts = np.asarray(points2d, float)
    if len(pts) < 4:
        return MultiPoint(pts).convex_hull

    # Quantized downsampling to cap points (no pairwise NxN distances!)
    if len(pts) > max_points:
        idx = quantized_unique(pts, grid)
        if len(idx) > max_points:
            # uniform subsample if still too many
            idx = np.random.choice(idx, size=max_points, replace=False)
        pts = pts[idx]

    tri, tri_pts = safe_delaunay(pts)
    if tri is None:
        # graceful fallback
        return MultiPoint(pts).convex_hull

    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = tri_pts[ia], tri_pts[ib], tri_pts[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area_val = s * (s - a) * (s - b) * (s - c)
        if area_val <= 0:
            circum_r = np.inf
        else:
            area = math.sqrt(area_val)
            circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / max(alpha, 1e-9):
            edges.add(tuple(sorted((ia, ib))))
            edges.add(tuple(sorted((ib, ic))))
            edges.add(tuple(sorted((ic, ia))))

    # Build boundary graph from kept edges
    if not edges:
        return MultiPoint(tri_pts).convex_hull

    segs = [LineString([tri_pts[i], tri_pts[j]]) for (i, j) in edges]
    merged = unary_union(segs)
    polys = list(polygonize(merged))
    if not polys:
        return MultiPoint(tri_pts).convex_hull
    return unary_union(polys)


# =============================================================================
# Clip 3D polylines to 2D polygon (XZ) with Y interpolation
# =============================================================================

def interpolate_y_along_polyline(orig3d: np.ndarray, points2d: np.ndarray) -> np.ndarray:
    """Given the original 3D polyline and a set of 2D (x,z) points on its projection,
    interpolate Y by arclength along the projected polyline.
    """
    P = np.asarray(orig3d, float)
    P2 = P[:, [0, 2]]
    seg = np.diff(P2, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    S = np.concatenate([[0.0], np.cumsum(seglen)])  # arclength at vertices
    total = S[-1] if S[-1] > 0 else 1.0

    LS = LineString(P2)
    Ys = []
    for q in points2d:
        s = LS.project(Point(q))
        # locate interval
        i = np.searchsorted(S, s, side='right') - 1
        i = max(0, min(i, len(P) - 2))
        t = 0.0 if (S[i+1] - S[i]) == 0 else (s - S[i]) / (S[i+1] - S[i])
        y = P[i, 1] * (1 - t) + P[i+1, 1] * t
        Ys.append(y)
    return np.asarray(Ys, float)


def clip_line_to_polygon_3d(line3d: np.ndarray, poly2d: Polygon) -> List[np.ndarray]:
    """Clip a 3D polyline to a 2D polygon (on XZ). Returns list of 3D polylines."""
    L2 = LineString(line3d[:, [0, 2]])
    inter = L2.intersection(poly2d)
    out: List[np.ndarray] = []
    def build3d(coords2d: List[Tuple[float, float]]):
        coords2d = np.asarray(coords2d, float)
        Ys = interpolate_y_along_polyline(line3d, coords2d)
        return np.c_[coords2d[:, 0], Ys, coords2d[:, 1]]

    if inter.is_empty:
        return out
    if inter.geom_type == 'LineString':
        out.append(build3d(list(inter.coords)))
    elif inter.geom_type == 'MultiLineString':
        for seg in inter.geoms:
            out.append(build3d(list(seg.coords)))
    # ignore points-only intersections
    return out


# =============================================================================
# Skeleton assembly (nodes/edges)
# =============================================================================

def build_skeleton_graph(polylines: List[np.ndarray], snap_tol: float = 0.05) -> Dict[str, Any]:
    nodes: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []

    def add_node(p: np.ndarray) -> int:
        for i, q in enumerate(nodes):
            if np.linalg.norm(p - q) <= snap_tol:
                nodes[i] = 0.5 * (nodes[i] + p)
                return i
        nodes.append(p.copy())
        return len(nodes) - 1

    for pl in polylines:
        if len(pl) < 2:
            continue
        i0 = add_node(pl[0])
        i1 = add_node(pl[-1])
        edges.append((i0, i1))

    return {'nodes': np.asarray(nodes, float), 'edges': np.asarray(edges, int)}


# =============================================================================
# Public pipeline
# =============================================================================

def extract_ridges_and_skeleton(points: Optional[np.ndarray] = None,
                                o3d_pcd: Optional['o3d.geometry.PointCloud'] = None,
                                contour_points: Optional[np.ndarray] = None,
                                plane_distance_threshold: float = 0.02,
                                plane_min_points: int = 1000,
                                line_point_tol: float = 0.03,
                                apex_step: float = 0.1,
                                max_planes: int = 3,
                                voxel_for_seg: float = 0.0,
                                alpha: float = 0.25,
                                max_contour_points: int = 20000,
                                contour_grid: float = 0.05) -> Dict[str, Any]:
    """End-to-end extraction with robust memory-safe steps."""
    if o3d_pcd is None and points is None:
        raise ValueError("Provide either `points` (Nx3) or `o3d_pcd`.")
    P = from_o3d_pcd(o3d_pcd) if o3d_pcd is not None else np.asarray(points, float)

    # 1) Segment planes (optionally on a downsampled copy) and map back
    planes: List[Plane] = segment_multiple_planes(
        P, max_planes=max_planes, distance_threshold=plane_distance_threshold,
        num_iterations=2000, min_points=plane_min_points, voxel_for_seg=voxel_for_seg)

    # Identify floor plane (max |n·z|) and walls (others)
    floor_plane = max(planes, key=lambda pl: abs(pl.n[2])) if planes else None

    # Split used vs roof points
    used = np.zeros(len(P), dtype=bool)
    for pl in planes:
        used[pl.inliers] = True
    roof_pts = P[~used]

    # 2) Plane intersections -> raw ridge polylines
    polylines: List[np.ndarray] = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            L = intersect_planes(planes[i], planes[j])
            if L is None:
                continue
            near = extract_points_near_line(P, L, tol=line_point_tol)
            if len(near) < 30:
                continue
            p0, v = fit_line_pca(near)
            # order by parameter along v
            t = (near - p0) @ v
            order = np.argsort(t)
            poly = p0[None, :] + t[order][:, None] * v[None, :]
            polylines.append(poly)

    # 3) Roof centerline (apex) from non-planar points
    apex_poly = None
    if len(roof_pts) >= 100:
        try:
            axis = estimate_roof_axis_from_points(roof_pts)
            bbox_min = P.min(axis=0)
            bbox_max = P.max(axis=0)
            apex_poly = sample_apex_polyline(axis, (bbox_min, bbox_max), step=apex_step)
            polylines.append(apex_poly)
        except Exception:
            pass

    # 4) Build contour concave hull & clip polylines (XZ)
    contour_info = None
    if contour_points is not None and len(contour_points) >= 3:
        pts2d = np.asarray(contour_points, float)[:, [0, 2]]
        hull_poly = alpha_shape_polygon(
            pts2d,
            alpha=alpha,
            max_points=max_contour_points,
            grid=contour_grid
        )

        # --- ensure we always end up with a Polygon ---
        if not isinstance(hull_poly, Polygon):
            hull_poly = MultiPoint(pts2d).convex_hull

        clipped: List[np.ndarray] = []
        for pl in polylines:
            parts = clip_line_to_polygon_3d(pl, hull_poly)
            for seg in parts:
                if len(seg) >= 2:
                    clipped.append(seg)
        polylines = clipped
        contour_info = {'hull_exterior': np.array(hull_poly.exterior.coords)}

    # 5) Build simple graph from polylines
    graph = build_skeleton_graph(polylines)

    return {
        'planes': [{'n': pl.n, 'd': pl.d, 'num_inliers': int(len(pl.inliers))} for pl in planes],
        'skeleton': {'lines': polylines, 'graph': graph},
        'roof_apex': apex_poly,
        'contour': contour_info
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_result(result: Dict[str, Any], points: np.ndarray):
    if o3d is None:
        raise RuntimeError("open3d is required for visualization.")
    geoms: List[Any] = []

    pcd = to_o3d_pcd(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd)

    # Contour hull (green)
    if result.get('contour'):
        ext = result['contour']['hull_exterior']
        ext3 = np.c_[ext[:, 0], np.full(len(ext), ext[:, 0].min()*0), ext[:, 1]]  # y=0 plane for display
        lines = np.column_stack([np.arange(len(ext)-1), np.arange(1, len(ext))])
        lines = np.vstack([lines, [len(ext)-1, 0]])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(ext3),
            lines=o3d.utility.Vector2iVector(lines.astype(int))
        )
        ls.paint_uniform_color([0, 1, 0])
        geoms.append(ls)

    # Skeleton (red)
    for pl in result['skeleton']['lines']:
        if len(pl) < 2:
            continue
        lines = np.column_stack([np.arange(len(pl)-1), np.arange(1, len(pl))]).astype(int)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pl),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.paint_uniform_color([1, 0, 0])
        geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Wharf cellar ridge & skeleton extraction (memory-safe)")
    ap.add_argument('--points', type=str, required=True, help='Path to Nx3 numpy .npy (full hull cloud)')
    ap.add_argument('--contour', type=str, default=None, help='Path to Nx3 numpy .npy (contour cloud)')
    ap.add_argument('--voxel', type=float, default=0.0, help='Voxel size used only for plane segmentation downsample (m)')
    ap.add_argument('--alpha', type=float, default=0.25, help='Alpha for concave hull (smaller=tighter)')
    ap.add_argument('--max-contour-points', type=int, default=20000, help='Cap contour points used in alpha shape')
    ap.add_argument('--contour-grid', type=float, default=0.05, help='Grid (m) for quantized downsampling of contour')
    ap.add_argument('--plane-dist', type=float, default=0.03, help='RANSAC plane distance threshold (m)')
    ap.add_argument('--plane-min', type=int, default=1500, help='Minimum inliers to accept a plane')
    ap.add_argument('--line-tol', type=float, default=0.05, help='Tolerance to collect points near plane-intersection lines (m)')
    ap.add_argument('--apex-step', type=float, default=0.2, help='Sampling step (m) for apex polyline along axis')
    ap.add_argument('--max-planes', type=int, default=3, help='Max planes to pull via RANSAC')
    ap.add_argument('--no-viz', action='store_true', help='Skip Open3D visualization')
    args = ap.parse_args()

    pts = np.load(args.points)
    contour = None if args.contour is None else np.load(args.contour)

    result = extract_ridges_and_skeleton(
        points=pts,
        contour_points=contour,
        plane_distance_threshold=args.plane_dist,
        plane_min_points=args.plane_min,
        line_point_tol=args.line_tol,
        apex_step=args.apex_step,
        max_planes=args.max_planes,
        voxel_for_seg=args.voxel,
        alpha=args.alpha,
        max_contour_points=args.max_contour_points,
        contour_grid=args.contour_grid,
    )

    # Print a tiny summary
    print("\n=== Summary ===")
    print(f"Planes: {len(result['planes'])} -> inliers: {[p['num_inliers'] for p in result['planes']]}")
    print(f"Skeleton lines: {len(result['skeleton']['lines'])}")
    if result.get('contour'):
        print(f"Contour hull vertices: {len(result['contour']['hull_exterior'])}")

    if not args.no_viz:
        visualize_result(result, points=pts)


if __name__ == '__main__':
    main()
