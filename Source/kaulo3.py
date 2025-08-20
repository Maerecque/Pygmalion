import numpy as np
import open3d as o3d
from shapely.geometry import MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

# ----------------------------
# Alpha Shape (Concave Hull)
# ----------------------------
def alpha_shape(points: np.ndarray, alpha: float) -> Polygon:
    """
    Compute the alpha shape (concave hull) of a set of 2D points using Delaunay triangulation.
    This avoids NxN distance matrices to prevent memory errors.
    """
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges = set()
    edge_segments = []

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)

        if area == 0.0:
            continue  # degenerate triangle

        area = np.sqrt(area)
        circum_r = a * b * c / (4.0 * area)

        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    for i, j in edges:
        edge_segments.append(LineString([points[i], points[j]]))

    m = unary_union(edge_segments)
    polygons = list(polygonize(m))

    if not polygons:
        return MultiPoint(points).convex_hull

    return unary_union(polygons)

# ----------------------------
# Contour Hull with Downsampling
# ----------------------------
def build_contour_hull(contour_points: np.ndarray, alpha: float = 0.1, voxel_size: float = 0.1):
    """
    Build a concave hull from contour points.
    Downsamples the point cloud first to prevent excessive memory usage.
    """
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contour_points)

    # Downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pts2d = np.asarray(pcd.points)[:, [0, 2]]
    poly = alpha_shape(pts2d, alpha=alpha)

    if poly.geom_type == "Polygon":
        return np.array(poly.exterior.coords)
    else:
        return np.array(poly.convex_hull.exterior.coords)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Load your contour point cloud (Nx3 numpy array)
    contour = np.load("wharf_cellar_contour.npy")

    hull = build_contour_hull(contour, alpha=0.1, voxel_size=0.05)
    print("Hull coordinates:", hull.shape)

    # Optional visualization of contour and hull
    import matplotlib.pyplot as plt
    plt.scatter(contour[:, 0], contour[:, 2], s=1, label="contour")
    plt.plot(hull[:, 0], hull[:, 1], "r-", label="hull")
    plt.legend()
    plt.show()
