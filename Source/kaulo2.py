import numpy as np
import open3d as o3d
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import alphashape
from typing import List, Dict


# ---------- Geometry helpers ----------

def estimate_alpha(points: np.ndarray) -> float:
    """Estimate a reasonable alpha parameter for alphashape based on average nearest-neighbor distance."""
    from sklearn.neighbors import NearestNeighbors

    if len(points) < 10:
        return 1.0

    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_dist = distances[:, 1].mean()
    return 1.0 / (mean_dist * 2.0)


def build_contour_hull(contour_points: np.ndarray, alpha: float = None) -> Polygon:
    """Build a concave hull polygon from contour points projected on XZ-plane."""
    pts2d = contour_points[:, [0, 2]]

    if alpha is None:
        alpha = estimate_alpha(pts2d)

    poly = alphashape.alphashape(pts2d, alpha)
    if poly.geom_type == "Polygon":
        return poly
    else:
        return poly.convex_hull


def clip_line_to_polygon(line: np.ndarray, polygon: Polygon) -> List[np.ndarray]:
    """Clip a 3D polyline (projected to XZ-plane) to a 2D polygon, return list of 3D polylines."""
    line2d = LineString(line[:, [0, 2]])
    clipped = line2d.intersection(polygon)

    clipped_lines = []
    if clipped.is_empty:
        return clipped_lines

    if clipped.geom_type == "LineString":
        coords = np.array(clipped.coords)
        y_mean = line[:, 1].mean()
        clipped_lines.append(np.c_[coords[:, 0], np.full(len(coords), y_mean), coords[:, 1]])
    elif clipped.geom_type == "MultiLineString":
        for seg in clipped.geoms:
            coords = np.array(seg.coords)
            y_mean = line[:, 1].mean()
            clipped_lines.append(np.c_[coords[:, 0], np.full(len(coords), y_mean), coords[:, 1]])
    return clipped_lines


# ---------- Main processing ----------

def extract_ridges_and_skeleton(
    points: np.ndarray,
    contour_points: np.ndarray,
    plane_distance_threshold: float = 0.02,
    plane_min_points: int = 1000,
    line_point_tol: float = 0.05,
    apex_step: float = 0.1,
    max_planes: int = 3,
    alpha: float = None
) -> Dict[str, List[np.ndarray]]:
    """
    Detect planes (walls, floor), find ridges and roof apex, build skeleton,
    and clip skeleton inside concave contour hull.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # Plane segmentation (walls, floor, roof planes)
    planes = []
    remaining = pcd
    for _ in range(max_planes):
        if len(remaining.points) < plane_min_points:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        inlier_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        planes.append((plane_model, inlier_cloud))

    # Approximate ridges = PCA major axes of planes
    ridges = []
    for model, cloud in planes:
        pts = np.asarray(cloud.points)
        if len(pts) < 2:
            continue
        pts_mean = pts.mean(axis=0)
        u, s, vh = np.linalg.svd(pts - pts_mean)
        direction = vh[0]
        line = np.stack([pts_mean - direction, pts_mean + direction])
        ridges.append(line)

    # Roof apex (approx: highest Y sample every apex_step along X)
    roof_points = np.asarray(pcd.points)
    xs = roof_points[:, 0]
    bins = np.arange(xs.min(), xs.max(), apex_step)
    apex_points = []
    for b in bins:
        mask = (xs >= b) & (xs < b + apex_step)
        if np.any(mask):
            seg = roof_points[mask]
            top = seg[seg[:, 1].argmax()]
            apex_points.append(top)
    if len(apex_points) > 1:
        ridges.append(np.array(apex_points))

    # Build concave hull from contour
    hull_polygon = build_contour_hull(contour_points, alpha=alpha)

    # Clip ridges to hull
    skeleton = []
    for ridge in ridges:
        clipped_parts = clip_line_to_polygon(ridge, hull_polygon)
        skeleton.extend(clipped_parts)

    return {
        "planes": planes,
        "ridges": ridges,
        "skeleton": skeleton,
        "contour": np.array(hull_polygon.exterior.coords)
    }


# ---------- Visualization ----------

def visualize_result(result: Dict[str, List[np.ndarray]], points: np.ndarray):
    geoms = []

    # Original cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd)

    # Contour hull
    contour = result["contour"]
    contour_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.c_[contour[:, 0], np.zeros(len(contour)), contour[:, 1]]),
        lines=o3d.utility.Vector2iVector([[i, (i + 1) % len(contour)] for i in range(len(contour))])
    )
    contour_line.paint_uniform_color([0, 1, 0])
    geoms.append(contour_line)

    # Skeleton
    for line in result["skeleton"]:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(line) - 1)])
        )
        ls.paint_uniform_color([1, 0, 0])
        geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)


# ---------- Example usage ----------
if __name__ == "__main__":
    pts = np.load("wharf_cellar.npy")               # full hull cloud
    contour = np.load("wharf_cellar_contour.npy")   # contour cloud

    result = extract_ridges_and_skeleton(
        points=pts,
        contour_points=contour,
        plane_distance_threshold=0.03,
        plane_min_points=1500,
        line_point_tol=0.05,
        apex_step=0.2,
        max_planes=3,
        alpha=None   # None = auto-estimate
    )

    visualize_result(result, points=pts)
