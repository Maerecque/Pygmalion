"""
Wharf Cellar Ridge & Skeleton Extraction (with Contour Constraint)
-----------------------------------------------------------------

This module takes a 3D point cloud (as either an Open3D PointCloud or an
Nx3 numpy array) representing the hull of a wharf cellar and:

1) Segments dominant planes (floor & walls) using iterative RANSAC.
2) Computes wall/floor ridges as intersection lines of detected planes.
3) Extracts the rounded roof points (non-planar remainder) and estimates a
   barrel-vault centerline ("middle ridge") via PCA-based cylinder proxy.
4) Produces a skeleton graph (nodes + edges) from ridges + roof centerline.
5) (NEW) If a contour point cloud is provided, the skeleton is clipped so
   that its XZ coordinates do not exceed the contour boundary.

Dependencies
------------
- open3d (>= 0.15 suggested)
- numpy
- scipy (for convex hull)
- matplotlib (for point-in-polygon)

"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union, polygonize
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    o3d = None
    print("Warning: open3d not installed. Install with `pip install open3d`. Visualization and some ops will be unavailable.")

# -------------------------------
# Utility math helpers
# -------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def pca_directions(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(pts, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.dot(Xc.T, Xc) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def point_line_distance(points: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = unit(v)
    diff = points - p0
    cross = np.cross(diff, v)
    return np.linalg.norm(cross, axis=1)


def fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    _, V = pca_directions(pts)
    dir_vec = unit(V[:, 0])
    return pts.mean(axis=0), dir_vec


# -------------------------------
# Open3D conversion helpers
# -------------------------------

def to_o3d_pcd(points: np.ndarray) -> 'o3d.geometry.PointCloud':
    if o3d is None:
        raise RuntimeError("open3d is required for point cloud ops.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    return pcd


def from_o3d_pcd(pcd: 'o3d.geometry.PointCloud') -> np.ndarray:
    return np.asarray(pcd.points)


# -------------------------------
# Segmentation: planes (floor, walls)
# -------------------------------
@dataclass
class Plane:
    n: np.ndarray
    d: float
    inliers: np.ndarray


def segment_multiple_planes(points: np.ndarray,
                            max_planes: int = 3,
                            distance_threshold: float = 0.02,
                            ransac_n: int = 3,
                            num_iterations: int = 2000,
                            min_points: int = 1000) -> List[Plane]:
    if o3d is None:
        raise RuntimeError("open3d required for RANSAC plane segmentation.")

    pts = np.asarray(points, float)
    remaining_idx = np.arange(len(pts))
    planes: List[Plane] = []

    for _ in range(max_planes):
        if len(remaining_idx) < min_points:
            break
        pcd = to_o3d_pcd(pts[remaining_idx])
        model, inliers_local = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        a, b, c, d = model
        n = unit(np.array([a, b, c], float))
        inliers = remaining_idx[np.array(inliers_local, int)]
        if len(inliers) < min_points:
            break
        planes.append(Plane(n=n, d=float(d), inliers=inliers))
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[np.array(inliers_local, int)] = False
        remaining_idx = remaining_idx[mask]

    planes.sort(key=lambda pl: len(pl.inliers), reverse=True)
    return planes


# -------------------------------
# Ridges: intersection lines between planes
# -------------------------------
@dataclass
class Line:
    p: np.ndarray
    v: np.ndarray


def intersect_planes(p1: Plane, p2: Plane) -> Optional[Line]:
    n1, d1 = p1.n, p1.d
    n2, d2 = p2.n, p2.d
    v = np.cross(n1, n2)
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return None
    v = v / nv
    A = np.vstack([n1, n2])
    b = -np.array([d1, d2])
    A_ext = np.vstack([A, v])
    b_ext = np.hstack([b, 0.0])
    p, *_ = np.linalg.lstsq(A_ext, b_ext, rcond=None)
    return Line(p=p, v=v)


def extract_points_near_line(points: np.ndarray, line: Line, tol: float = 0.03) -> np.ndarray:
    d = point_line_distance(points, line.p, line.v)
    return points[d <= tol]


# -------------------------------
# Roof: barrel-vault axis via PCA
# -------------------------------
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
    corners = np.array([[mn[0], mn[1], mn[2]],
                        [mx[0], mn[1], mn[2]],
                        [mn[0], mx[1], mn[2]],
                        [mn[0], mn[1], mx[2]],
                        [mx[0], mx[1], mn[2]],
                        [mx[0], mn[1], mx[2]],
                        [mn[0], mx[1], mx[2]],
                        [mx[0], mx[1], mx[2]]])
    def proj_t(x):
        return np.dot(x - axis.axis_point, axis.axis_dir)
    ts = [proj_t(c) for c in corners]
    tmin, tmax = min(ts), max(ts)
    num = int(max(2, np.ceil((tmax - tmin) / max(step, 1e-3))))
    T = np.linspace(tmin, tmax, num)
    pts = axis.axis_point[None, :] + T[:, None] * axis.axis_dir[None, :]
    return pts


# -------------------------------
# Skeleton assembly
# -------------------------------

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

    return {'nodes': np.array(nodes, float), 'edges': np.array(edges, int)}


# -------------------------------
# Contour helpers
# -------------------------------
def alpha_shape(points: np.ndarray, alpha: float) -> Polygon:
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    points: Nx2 array
    alpha: alpha parameter (smaller = tighter fit, larger = looser fit)
    Returns a shapely Polygon
    """
    from shapely.geometry import Point
    import shapely

    if len(points) < 4:
        # not enough points for a polygon
        return MultiPoint(points).convex_hull

    from scipy.spatial import Delaunay
    tri = Delaunay(points)

    edges = set()
    edge_points = []
    # Loop over triangles
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area == 0.0:
            circum_r = np.inf
        else:
            area = np.sqrt(area)
            circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    for i, j in edges:
        edge_points.append(points[[i, j]])

    m = MultiPoint(points)
    triangles = list(polygonize(edge_points))
    return unary_union(triangles)


def build_contour_hull(contour_points: np.ndarray, alpha: float = 0.1):
    pts2d = contour_points[:, [0, 2]]
    poly = alpha_shape(pts2d, alpha=alpha)
    if poly.geom_type == "Polygon":
        return np.array(poly.exterior.coords)
    else:
        return np.array(poly.convex_hull.exterior.coords)


def point_in_polygon_2d(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path
    path = Path(polygon)
    return path.contains_points(points)


def clip_polyline_to_contour(poly: np.ndarray, contour_hull: np.ndarray) -> np.ndarray:
    if len(poly) == 0:
        return poly
    pts2d = poly[:, [0, 2]]
    inside = point_in_polygon_2d(pts2d, contour_hull)
    return poly[inside]


# -------------------------------
# Public pipeline
# -------------------------------

def extract_ridges_and_skeleton(points: Optional[np.ndarray] = None,
                                o3d_pcd: Optional['o3d.geometry.PointCloud'] = None,
                                roof_points: Optional[np.ndarray] = None,
                                wall_points: Optional[np.ndarray] = None,
                                floor_points: Optional[np.ndarray] = None,
                                contour_points: Optional[np.ndarray] = None,
                                plane_distance_threshold: float = 0.02,
                                plane_min_points: int = 1000,
                                line_point_tol: float = 0.03,
                                apex_step: float = 0.1,
                                max_planes: int = 3) -> Dict[str, Any]:

    if o3d_pcd is None and points is None:
        raise ValueError("Provide either `points` (Nx3) or `o3d_pcd`.")
    pts_all = from_o3d_pcd(o3d_pcd) if o3d_pcd is not None else np.asarray(points, float)
    bbox_min = pts_all.min(axis=0)
    bbox_max = pts_all.max(axis=0)

    planes: List[Plane] = segment_multiple_planes(pts_all,
                                                  max_planes=max_planes,
                                                  distance_threshold=plane_distance_threshold,
                                                  min_points=plane_min_points)
    def planar_score_floor(pl: Plane):
        return abs(pl.n[2])
    floor_plane = max(planes, key=planar_score_floor) if planes else None
    floor_inliers = floor_plane.inliers if floor_plane is not None else np.array([], int)
    wall_planes = [pl for pl in planes if pl is not floor_plane]
    wall_inliers = np.unique(np.concatenate([pl.inliers for pl in wall_planes])) if wall_planes else np.array([], int)

    if floor_points is None:
        floor_points = pts_all[floor_inliers]
    if wall_points is None:
        wall_points = pts_all[wall_inliers]

    if roof_points is None:
        used = np.zeros(len(pts_all), dtype=bool)
        for pl in planes:
            used[pl.inliers] = True
        roof_points = pts_all[~used]

    plane_ridges = []
    for i in range(len(planes)):
        for j in range(i+1, len(planes)):
            L = intersect_planes(planes[i], planes[j])
            if L is None:
                continue
            near_pts = extract_points_near_line(pts_all, L, tol=line_point_tol)
            if len(near_pts) < 50:
                continue
            p_fit, v_fit = fit_line_pca(near_pts)
            plane_ridges.append({'p': p_fit, 'v': v_fit, 'pts': near_pts})

    roof_axis = None
    apex_poly = None
    if roof_points is not None and len(roof_points) >= 100:
        roof_axis = estimate_roof_axis_from_points(roof_points)
        apex_poly = sample_apex_polyline(roof_axis, (bbox_min, bbox_max), step=apex_step)

    polylines: List[np.ndarray] = []
    for ridge in plane_ridges:
        p0, v = ridge['p'], unit(ridge['v'])
        P = ridge['pts']
        t = np.dot(P - p0, v)
        order = np.argsort(t)
        poly = p0[None, :] + t[order][:, None] * v[None, :]
        polylines.append(poly)

    if apex_poly is not None:
        polylines.append(apex_poly)

    if contour_points is not None and len(contour_points) >= 3:
        contour_hull = build_contour_hull(contour_points)
        clipped_polys = []
        for poly in polylines:
            clipped = clip_polyline_to_contour(poly, contour_hull)
            if len(clipped) >= 2:
                clipped_polys.append(clipped)
        polylines = clipped_polys
        contour_info = {'points': contour_points, 'hull': contour_hull}
    else:
        contour_info = None

    graph = build_skeleton_graph(polylines)

    return {
        'planes': [{'n': pl.n, 'd': pl.d, 'inliers': pl.inliers} for pl in planes],
        'plane_ridges': plane_ridges,
        'roof': None if roof_axis is None else {
            'axis_point': roof_axis.axis_point,
            'axis_dir': roof_axis.axis_dir,
            'radius': roof_axis.radius,
            'apex_polyline': apex_poly
        },
        'skeleton': {
            'lines': polylines,
            'graph': graph
        },
        'contour': contour_info
    }


# -------------------------------
# Visualization
# -------------------------------

def visualize_result(result: Dict[str, Any], points: Optional[np.ndarray] = None, o3d_pcd: Optional['o3d.geometry.PointCloud'] = None):
    if o3d is None:
        raise RuntimeError("open3d is required for visualization.")
    if o3d_pcd is None:
        if points is None:
            raise ValueError("Provide either `points` or `o3d_pcd` for visualization.")
        cloud = to_o3d_pcd(points)
    else:
        cloud = o3d_pcd

    geometries = [cloud]

    for ridge in result.get('plane_ridges', []):
        pts = ridge['pts']
        if len(pts) < 2:
            continue
        p0, v = ridge['p'], unit(ridge['v'])
        t = np.dot(pts - p0, v)
        tmin, tmax = t.min(), t.max()
        seg = np.vstack([p0 + tmin * v, p0 + tmax * v])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(seg),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1]], int))
        )
        geometries.append(ls)

    roof = result.get('roof')
    if roof and roof.get('apex_polyline') is not None and len(roof['apex_polyline']) >= 2:
        poly = roof['apex_polyline']
        lines = np.column_stack([np.arange(len(poly)-1), np.arange(1, len(poly))])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(poly),
            lines=o3d.utility.Vector2iVector(lines.astype(int))
        )
        geometries.append(ls)

    o3d.visualization.draw_geometries(geometries)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example:
    # pts = np.load('wharf_cellar.npy')
    # contour = np.load('wharf_cellar_contour.npy')
    # result = extract_ridges_and_skeleton(points=pts, contour_points=contour,
    #                                      plane_distance_threshold=0.03,
    #                                      plane_min_points=1500,
    #                                      line_point_tol=0.05,
    #                                      apex
    #                                      step=0.2, max_planes=3)
    pass
