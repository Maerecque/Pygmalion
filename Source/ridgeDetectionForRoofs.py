import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skimage.filters import median
from skimage.morphology import disk
from shapely.geometry import LineString, Point
from rasterio import features as rfeatures
import rasterio
import math

import os
import sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import get_file_path, readout_LAS_file
from Source.pointCloudEditor import open_point_cloud_editor as opce


def pointcloud_to_voxel_grid(pcd, voxel_size=0.2):
    print("[INFO] Starting voxelization and DEM creation...")
    """
    Convert an Open3D point cloud to a 3D binary voxel grid.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud of a single building.
        voxel_size (float): Size of one voxel in meters (resolution).

    Returns:
        hull (np.ndarray): 3D ndarray (Z, Y, X) with boolean occupancy (True=building voxel).
        (dz, dy, dx): voxel size in meters (tuple).
    """
    pts = np.asarray(pcd.points)
    min_bounds = pts.min(axis=0)
    max_bounds = pts.max(axis=0)

    # Compute voxel indices for each point
    indices = np.floor((pts - min_bounds) / voxel_size).astype(int)
    xs, ys, zs = indices[:, 0], indices[:, 1], indices[:, 2]

    # Only create a dense 2D DEM (max Z per XY)
    dem_shape = (
        ys.max() + 1,
        xs.max() + 1
    )
    dem = np.full(dem_shape, -np.inf, dtype=float)
    for x, y, z, orig_z in zip(xs, ys, zs, pts[:, 2]):
        # Store the highest Z value for each (x, y)
        if orig_z > dem[y, x]:
            dem[y, x] = orig_z

    print(f"[INFO] DEM created with shape {dem.shape} and voxel size {voxel_size}")

    dz = dy = dx = voxel_size
    # Instead of returning a 3D hull, return the DEM and min_bounds
    return dem, (dz, dy, dx), min_bounds


def detect_ridges_single_building(
    hull, dz=0.2, dy=0.2, dx=0.2,
    min_length_m=0.01,
    min_sinuosity=0.0,
    angle_merge_deg=5.0,
    endpoint_tol_m=2.0,
    major_height_quantile=0.80,
    major_aspect_index_thresh=0.20,  # Lowered to allow less sharp ridges
    major_slope_thresh=0.40  # Increased to allow flatter ridges
):
    print("[INFO] Starting ridge detection on DEM...")
    """
    Detect roof ridges from a single building voxel hull.

    Steps:
    1. Collapse voxel hull into a roof DEM (max Z for each XY).
    2. Compute slope, aspect, and aspect index.
    3. Detect local maxima as ridge seeds.
    4. Skeletonize ridges and vectorize to LineStrings.
    5. Filter, merge, and classify ridges into major/minor.

    Args:
        hull (np.ndarray): 3D occupancy grid (Z,Y,X).
        dz,dy,dx (float): voxel resolution in meters.
        min_length_m (float): Minimum ridge length to keep.
        min_sinuosity (float): Minimum straightness (chord/length).
        angle_merge_deg (float): Angle tolerance (degrees) for merging collinear lines.
        endpoint_tol_m (float): Distance tolerance for merging endpoints (meters).
        major_height_quantile (float): Height percentile threshold for major ridge classification.
        major_aspect_index_thresh (float): Aspect index threshold for major ridges.
        major_slope_thresh (float): Maximum slope allowed for major ridges.

    Returns:
        merged (list[LineString]): All detected ridges.
        major (list[LineString]): Ridges classified as major.
        minor (list[LineString]): Ridges classified as minor.
    """
    def show_slope_heatmap(slope_arr):
        plt.figure(figsize=(8, 6))
        plt.imshow(slope_arr, cmap='viridis', origin='lower')
        plt.colorbar(label='Slope (rise/run)')
        plt.title('Roof Slope Heatmap')
        plt.xlabel('X voxel index')
        plt.ylabel('Y voxel index')
        plt.tight_layout()
        plt.show()

    # --- 1) Roof DEM: already provided as input (for memory efficiency) ---
    top_dem = hull  # hull is now the DEM (2D array)
    # Smooth DEM to enhance broad ridges
    print("[INFO] Smoothing DEM to enhance broad ridges...")
    from scipy.ndimage import gaussian_filter
    top_dem = gaussian_filter(top_dem, sigma=1)

    # --- 2) Compute slope and aspect ---
    print("[INFO] Calculating slope and aspect...")
    dzdx = (np.roll(top_dem, -1, axis=1) - np.roll(top_dem, 1, axis=1)) / (2 * dx)
    dzdy = (np.roll(top_dem, -1, axis=0) - np.roll(top_dem, 1, axis=0)) / (2 * dy)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
    aspect = (np.arctan2(dzdy, -dzdx) + 2 * np.pi) % (2 * np.pi)
    # Show slope heatmap for visualization
    print("[INFO] Displaying slope heatmap...")
    show_slope_heatmap(slope)

    # --- 3) Aspect index (local variation) ---
    print("[INFO] Calculating aspect index...")
    aspect_med = median(aspect, disk(1))
    aspect_index = np.abs(aspect - aspect_med) / np.pi

    # --- 4) Ridge seeds = local maxima in DEM ---
    print("[INFO] Detecting ridge seeds (local maxima, larger window)...")
    ridge_seed = top_dem == ndi.maximum_filter(top_dem, size=7)

    # --- 5) Skeletonize ---
    print("[INFO] Skeletonizing ridge seeds...")
    ridge_skel = skeletonize(ridge_seed)

    # --- 6) Vectorize ridge skeleton ---
    print("[INFO] Vectorizing skeleton to lines...")
    shapes = rfeatures.shapes(
        ridge_skel.astype(np.uint8),
        mask=ridge_skel,
        transform=rasterio.Affine(dx, 0, 0, 0, -dy, 0)
    )
    raw_lines = [LineString(geom['coordinates'][0]) for geom, val in shapes if val == 1]
    print(f"[INFO] Found {len(raw_lines)} raw ridge lines.")

    # --- 7) Filter short or curvy ridges ---
    def sinuosity(line):
        if len(line.coords) < 2:
            return 0
        chord = Point(line.coords[0]).distance(Point(line.coords[-1]))
        return chord / line.length if line.length > 0 else 0

    print("[INFO] Filtering short or curvy ridges...")
    # Print length and sinuosity for the first 10 raw lines for inspection
    print("[DEBUG] First 10 raw ridge lines (length, sinuosity):")
    for i, line in enumerate(raw_lines[:10]):
        print(f"  Line {i+1}: length={line.length:.3f}, sinuosity={sinuosity(line):.3f}")
    filtered = [
        line for line in raw_lines
        if line.length >= min_length_m and sinuosity(line) >= min_sinuosity
    ]
    print(f"[INFO] {len(filtered)} ridges remain after filtering. (min_length_m={min_length_m}, min_sinuosity={min_sinuosity})")

    # --- 8) Merge nearby collinear segments ---
    def orientation(line):
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180

    print("[INFO] Merging nearby collinear ridge segments...")
    merged = []
    used = [False] * len(filtered)
    for i, li in enumerate(filtered):
        if used[i]:
            continue
        group = [li]
        used[i] = True
        for j, lj in enumerate(filtered):
            if used[j]:
                continue
            angdiff = abs(orientation(li) - orientation(lj))
            if min(angdiff, 180 - angdiff) <= angle_merge_deg:
                if Point(li.coords[-1]).distance(Point(lj.coords[0])) <= endpoint_tol_m:
                    group.append(lj)
                    used[j] = True
        # connect farthest endpoints
        endpoints = [ln.coords[0] for ln in group] + [ln.coords[-1] for ln in group]
        p0, p1 = max(
            ((a, b) for a in endpoints for b in endpoints),
            key=lambda ab: Point(ab[0]).distance(Point(ab[1]))
        )
        merged.append(LineString([p0, p1]))
    print(f"[INFO] {len(merged)} ridges after merging.")

    # --- 9) Classify major vs minor ridges ---
    print("[INFO] Classifying major and minor ridges...")
    q80 = np.nanpercentile(top_dem, major_height_quantile * 100)
    major, minor = [], []
    for ln in merged:
        xs, ys = zip(*ln.coords)
        rows = np.clip((np.array(ys) / dy).astype(int), 0, top_dem.shape[0] - 1)
        cols = np.clip((np.array(xs) / dx).astype(int), 0, top_dem.shape[1] - 1)
        hs = top_dem[rows, cols]
        ais = aspect_index[rows, cols]
        sls = slope[rows, cols]

        if (
            hs.mean() > q80
            and ais.mean() > major_aspect_index_thresh
            and sls.mean() < major_slope_thresh
        ):
            major.append(ln)
        else:
            minor.append(ln)
    print(f"[INFO] {len(major)} major ridges, {len(minor)} minor ridges classified.")
    return merged, major, minor


def ridge_detection_from_pointcloud(pcd, voxel_size=0.2):
    print("[INFO] Running full ridge detection pipeline...")
    """
    Convenience wrapper: Full ridge detection pipeline starting from Open3D point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Single building point cloud.
        voxel_size (float): Voxelization resolution in meters.

    Returns:
        merged (list[LineString]): All ridge lines.
        major (list[LineString]): Major ridge lines.
        minor (list[LineString]): Minor ridge lines.
    """
    hull, (dz, dy, dx), min_bounds = pointcloud_to_voxel_grid(pcd, voxel_size=voxel_size)
    merged, major, minor = detect_ridges_single_building(hull, dz=dz, dy=dy, dx=dx)
    return merged, major, minor, (dz, dy, dx), min_bounds

# Improved ridge plotting function (real-world coordinates, clearer output)
def plot_ridges_on_dem(dem, merged, major, minor, dz, dy, dx, min_bounds):
    plt.figure(figsize=(10, 8))
    plt.imshow(dem, cmap='gray', origin='lower',
               extent=[min_bounds[0], min_bounds[0] + dem.shape[1] * dx,
                       min_bounds[1], min_bounds[1] + dem.shape[0] * dy])
    def plot_lines(lines, color, label, lw, alpha):
        for i, line in enumerate(lines):
            x, y = zip(*line.coords)
            # Heuristic: if all x/y are within DEM shape, treat as voxel indices, else as world coords
            if (all(0 <= xx < dem.shape[1] for xx in x) and all(0 <= yy < dem.shape[0] for yy in y)):
                xw = [min_bounds[0] + xx * dx for xx in x]
                yw = [min_bounds[1] + yy * dy for yy in y]
            else:
                xw = x
                yw = y
            if i == 0:
                plt.plot(xw, yw, color=color, linewidth=lw, alpha=alpha, label=label)
            else:
                plt.plot(xw, yw, color=color, linewidth=lw, alpha=alpha)
    plot_lines(minor, 'orange', 'Minor Ridge', 1.5, 0.7)
    plot_lines(major, 'red', 'Major Ridge', 2.5, 0.9)
    plt.title('Detected Ridges on DEM (World Coordinates)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    pcd = o3d.io.read_point_cloud(get_file_path("Select a point cloud file to process", "PLY files (*.ply)"))

    opce(pcd)  # Open the point cloud editor for visualization

    merged, major, minor, (dz, dy, dx), min_bounds = ridge_detection_from_pointcloud(pcd, voxel_size=0.05)
    print(f"Detected {len(merged)} total ridges, {len(major)} major, {len(minor)} minor.")
    dem, _, _ = pointcloud_to_voxel_grid(pcd, voxel_size=0.05)
    plot_ridges_on_dem(dem, merged, major, minor, dz, dy, dx, min_bounds)


