"""
linesetTools.py

Utility functions for processing 3D geometry, including LineSet and mesh operations for building modeling.

Modules:
    - numpy
    - open3d
    - scipy.spatial
    - shapely.geometry
    - tkinter
    - json

Imports:
    - Delaunay: Performs 2D triangulation for mesh generation.
    - Polygon, Point, LineString: Used for geometric containment and filtering.
    - filedialog: Provides file save dialog for exporting CityJSON.

Functions:
    - contour_to_lineset: Creates a closed LineSet from ordered contour points.
    - filter_lines_within_contour: Filters lines in a LineSet, keeping only those completely within a given 2D contour.
    - merge_lineset: Merges multiple LineSets into a single LineSet.
    - lineset_to_trianglemesh: Converts a LineSet to a TriangleMesh, keeping only triangles within a supplied 2D contour.
    - generate_city_json_from_building: Exports a 3D building (floor, walls, roof) to CityJSON using a save dialog, repairing
      invalid polygons.

Typical Usage:
    1. Use contour_to_lineset to create a closed loop from contour points.
    2. Use filter_lines_within_contour to retain only lines inside a boundary.
    3. Use merge_lineset to combine multiple LineSets for unified processing.
    4. Use lineset_to_trianglemesh to generate a mesh from a LineSet within a contour.
    5. Use generate_city_json_from_building to export building geometry to CityJSON format.

See individual function docstrings for details.
"""

import numpy as np
import open3d as o3d
from typing import Optional
import json
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

from scipy.spatial import Delaunay
from shapely.geometry import Point as ShapelyPoint, Polygon, LineString


def contour_to_lineset(points):
    """
    Creates a closed LineSet from ordered 3D contour points.

    Connects each point in `points` to the next, forming a closed loop.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) containing ordered 3D coordinates of the contour.

    Returns:
        o3d.geometry.LineSet: Open3D LineSet object representing the closed contour.

    Example:
        >>> contour = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> lineset = contour_to_lineset(contour)
        >>> print(len(lineset.lines))  # Should print 4

    Note:
        - The contour is closed by connecting the last point back to the first.
        - All input points are included in the output LineSet.
    """
    n = len(points)
    lines = [[i, (i + 1) % n] for i in range(n)]  # closed loop
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def filter_lines_within_contour(contour_points: np.ndarray, lineset: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    Filters lines in a LineSet, keeping only those that are completely within a given 2D contour.

    For each line in `lineset`, checks if the line segment (projected to XY) is fully contained within or on the boundary of the
    polygon defined by `contour_points`. Only such lines are retained in the output.

    Parameters:
        contour_points (np.ndarray): Array of shape (N, 3) representing ordered 3D coordinates of the 2D contour (XY plane).
        lineset (o3d.geometry.LineSet): Open3D LineSet containing lines to be filtered.

    Returns:
        o3d.geometry.LineSet: Filtered LineSet containing only lines within the contour.

    Example:
        >>> contour = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
        >>> lineset = contour_to_lineset(np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]]))
        >>> filtered = filter_lines_within_contour(contour, lineset)
        >>> print(len(filtered.lines))  # Should print 4

    Note:
        - The contour is treated as a closed polygon in the XY plane.
        - Lines are retained only if fully inside or on the boundary of the polygon.
        - All original points are preserved in the output LineSet.
    """
    # Ensure contour is closed (first point == last point)
    if not np.allclose(contour_points[0], contour_points[-1]):
        contour_points = np.vstack([contour_points, contour_points[0]])

    polygon = Polygon(contour_points[:, :2])

    # Copy points and lines from lineset
    line_points = np.asarray(lineset.points)
    line_indices = np.asarray(lineset.lines)

    new_lines = []
    for (p1_idx, p2_idx) in line_indices:
        p1 = line_points[p1_idx]
        p2 = line_points[p2_idx]

        line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])

        # Check if line is completely within (or on) the polygon
        if polygon.contains(line) or polygon.covers(line):
            new_lines.append([p1_idx, p2_idx])

    # Create new LineSet
    filtered = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(new_lines)
    )

    return filtered


def merge_lineset(*linesets: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    Merges multiple LineSets into a single LineSet.

    Connects pairs of 3D points from two sets (floor and wall) if their XY coordinates are aligned within a specified tolerance.
    For each point in `floor_points`, finds the closest point in `wall_points` (in XY plane) within `xy_tol` distance,
    and creates a line connecting them. Useful for visualizing vertical connections between lower and upper surfaces.

    Parameters:
        *linesets: Variable number of o3d.geometry.LineSet objects to merge.

    Returns:
        o3d.geometry.LineSet: Combined LineSet containing all points and lines.

    Example:
        >>> floor = np.array([[0, 0, 0], [1, 1, 0]])
        >>> wall = np.array([[0, 0, 3], [1, 1, 3]])
        >>> lineset = connect_vertically_aligned_points(floor, wall, xy_tol=0.05)
        >>> print(len(lineset.lines))  # Should print 2

    Note:
        - Uses KDTree for efficient nearest neighbor search in the XY plane.
        - Only connects each floor point to its closest wall point if within tolerance.
        - All points (floor and wall) are included in the output LineSet.
    """
    all_points = []
    all_lines = []
    all_colors = []
    offset = 0

    for ls in linesets:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        all_points.append(pts)
        all_lines.append(lines + offset)
        if ls.has_colors():
            all_colors.append(np.asarray(ls.colors))
        offset += len(pts)

    merged = o3d.geometry.LineSet()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    if all_colors and all(len(c) == len(p) for c, p in zip(all_colors, all_points)):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    return merged


def lineset_to_trianglemesh(lineset, contour_points):
    """
    Converts a LineSet to a TriangleMesh, keeping only triangles within a supplied 2D contour.

    Parameters:
        lineset (o3d.geometry.LineSet): The input LineSet containing 3D points and lines.
        contour_points (np.ndarray): Array of shape (N, 3) representing the 2D contour in the XY plane.

    Returns:
        o3d.geometry.TriangleMesh: Open3D TriangleMesh object containing triangles whose centroids are inside the contour.

    Example:
        >>> contour = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
        >>> lineset = contour_to_lineset(np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]]))
        >>> mesh = lineset_to_trianglemesh(lineset, contour)
        >>> print(len(mesh.triangles))  # Should print number of triangles inside contour

    Note:
        - Projects all points to the XY plane for triangulation.
        - Only triangles whose centroid is inside or on the boundary of the contour are kept.
        - Raises ValueError if there are fewer than 3 points to form a mesh.
    """
    points = np.asarray(lineset.points)
    if len(points) < 3:
        raise ValueError("Need at least 3 points to form a mesh.")

    # Project to 2D for triangulation
    points_2d = points[:, :2]
    tri = Delaunay(points_2d)
    triangles = tri.simplices

    # Ensure contour is closed
    if not np.allclose(contour_points[0], contour_points[-1]):
        contour_points = np.vstack([contour_points, contour_points[0]])
    polygon = Polygon(contour_points[:, :2])

    # Filter triangles: keep only those whose centroid is inside the polygon
    filtered_triangles = []
    for simplex in tqdm(triangles, desc="Filtering triangles"):
        tri_pts = points_2d[simplex]
        centroid = np.mean(tri_pts, axis=0)
        if polygon.contains(ShapelyPoint(centroid)) or polygon.covers(ShapelyPoint(centroid)):
            filtered_triangles.append(simplex)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(filtered_triangles))
    mesh.compute_vertex_normals()
    return mesh


# This function is not even in use anymore, but I want to keep it for now
def generate_city_json_from_building(
    floor_lineset: o3d.geometry.LineSet,
    wall_lineset: o3d.geometry.LineSet,
    roof_lineset: o3d.geometry.LineSet,
    cityjson_properties: Optional[dict] = None
):
    """
    Export a 3D building (floor, walls, roof) to CityJSON using a save dialog.

    Repairs invalid polygons by removing duplicates, closing rings, and ensuring at least 3 unique vertices.

    Parameters:
        floor_lineset (o3d.geometry.LineSet): Floor boundary as LineSet.
        wall_lineset (o3d.geometry.LineSet): Wall boundary as LineSet.
        roof_lineset (o3d.geometry.LineSet): Roof boundary as LineSet.
        cityjson_properties (dict, optional): Additional CityJSON properties.

    Returns:
        None

    Raises:
        ValueError: If all surfaces are invalid or empty.
    """
    def lineset_to_closed_ring(lineset):
        """
        Convert a LineSet to a closed ring of vertex indices.

        Parameters:
            lineset (o3d.geometry.LineSet): Input LineSet.

        Returns:
            tuple: (points, idx_order) where points is the array of points and idx_order is the ordered list of indices.
        """
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)
        if len(points) == 0 or len(lines) == 0:
            raise ValueError("LineSet is empty.")
        idx_order = [int(lines[0][0]), int(lines[0][1])]
        used = set(idx_order)
        for _ in range(len(lines) - 1):
            last = idx_order[-1]
            found = False
            for line in lines:
                a, b = int(line[0]), int(line[1])
                if a == last and b not in used:
                    idx_order.append(b)
                    used.add(b)
                    found = True
                    break
                elif b == last and a not in used:
                    idx_order.append(a)
                    used.add(a)
                    found = True
                    break
            if not found:
                break
        return points, idx_order

    def repair_polygon(idx_ring, pts):
        """
        Repair a ring by removing consecutive duplicates, closing the ring, and ensuring at least 3 unique vertices.

        Parameters:
            idx_ring (list): List of vertex indices forming the ring.
            pts (np.ndarray): Array of points.

        Returns:
            list: Repaired ring of vertex indices.
        """
        # Remove consecutive duplicates
        idx_ring = [idx_ring[0]] + [idx for i, idx in enumerate(idx_ring[1:]) if idx != idx_ring[i]]
        # Close the ring if not closed
        if idx_ring[0] != idx_ring[-1]:
            idx_ring.append(idx_ring[0])
        # Ensure at least 3 unique vertices
        unique = []
        for idx in idx_ring:
            if idx not in unique:
                unique.append(idx)
        if len(unique) < 3:
            # Try to add more unique points from the lineset
            all_indices = set(range(len(pts)))
            missing = list(all_indices - set(unique))
            while len(unique) < 3 and missing:
                unique.append(missing.pop())
            # If still not enough, duplicate last unique
            while len(unique) < 3:
                unique.append(unique[-1])
            # Rebuild closed ring
            idx_ring = unique + [unique[0]]
        return idx_ring

    # 1. Extract closed rings for floor, wall, and roof
    surface_names = ['FloorSurface', 'WallSurface', 'RoofSurface']
    linesets = [floor_lineset, wall_lineset, roof_lineset]
    all_points = []
    rings = []
    for ls in linesets:
        try:
            pts, ring = lineset_to_closed_ring(ls)
            all_points.append(pts)
            rings.append(ring)
        except Exception as e:
            all_points.append(np.zeros((0, 3)))
            rings.append([])
            print(f"Warning: Could not extract ring: {e}")

    # 2. Stack all points and deduplicate globally
    all_points_concat = np.vstack([p for p in all_points if len(p) > 0])
    unique_points, inverse = np.unique(
        np.round(all_points_concat, 8), axis=0, return_inverse=True
    )

    # 3. Remap and repair rings to global indices
    offsets = np.cumsum([0] + [len(p) for p in all_points[:-1]])
    boundaries = []
    semantics = []
    for i, (ring, pts, name) in zip(rings, all_points, surface_names):
        if len(ring) == 0 or len(pts) == 0:
            print(f"Warning: {name} is empty and will be skipped.")
            continue
        offset = offsets[i]
        idx_ring = [int(inverse[offset + idx]) for idx in ring]
        idx_ring = repair_polygon(idx_ring, unique_points)
        boundaries.append([idx_ring])
        semantics.append({'type': name})

    if not boundaries:
        raise ValueError("No valid surfaces to export.")

    # 4. Build CityJSON object
    cityjson = {
        'type': 'CityJSON',
        'version': '1.1',
        'CityObjects': {
            'building_1': {
                'type': 'Building',
                'geometry': [{
                    'type': 'MultiSurface',
                    'lod': 2,
                    'boundaries': boundaries,
                    'semantics': {
                        'surfaces': semantics,
                        'values': [[i] for i in range(len(boundaries))]
                    }
                }],
                'attributes': {
                    'name': 'Example Building'
                }
            }
        },
        'vertices': [list(map(float, pt)) for pt in unique_points.tolist()],
        'metadata': {
            'referenceSystem': 'urn:ogc:def:crs:EPSG::28992'
        }
    }
    if cityjson_properties:
        cityjson.update(cityjson_properties)

    # 5. Tkinter save file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension='.json',
        filetypes=[('CityJSON files', '*.json')],
        title='Save CityJSON file'
    )
    if not file_path:
        print("Save cancelled.")
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cityjson, f, indent=2)
    print(f"CityJSON file saved to {file_path}")
