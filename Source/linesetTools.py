import numpy as np
import open3d as o3d
from typing import Optional
import json
import tkinter as tk
from tkinter import filedialog

from scipy.spatial import Delaunay
from shapely.geometry import Point as ShapelyPoint, Polygon, LineString


def contour_to_lineset(points):
    """
    Create a closed LineSet from ordered contour points.
    Args:
        points (np.ndarray): Nx3 array of ordered 3D points.
    Returns:
        o3d.geometry.LineSet
    """
    n = len(points)
    lines = [[i, (i + 1) % n] for i in range(n)]  # closed loop
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def filter_lines_within_contour(contour_points: np.ndarray, lineset: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    contour_points: Nx3 numpy array that represents the 2D contour in the XY plane.
    lineset: o3d.geometry.LineSet with lines.

    Return: filtered o3d.geometry.LineSet without lines that fall outside the contour.
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
    Merge multiple LineSets into one.
    Args:
        *linesets: Variable number of o3d.geometry.LineSet objects.
    Returns:
        o3d.geometry.LineSet: Combined LineSet.
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
    Convert a LineSet to a TriangleMesh, only keeping triangles within a supplied 2D contour.
    Args:
        lineset (o3d.geometry.LineSet): The input LineSet (assumed to be a closed loop).
        contour_points (np.ndarray): Nx3 array representing the 2D contour in the XY plane.
    Returns:
        o3d.geometry.TriangleMesh: Mesh with triangles inside the contour.
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
    for simplex in triangles:
        tri_pts = points_2d[simplex]
        centroid = np.mean(tri_pts, axis=0)
        if polygon.contains(ShapelyPoint(centroid)) or polygon.covers(ShapelyPoint(centroid)):
            filtered_triangles.append(simplex)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(filtered_triangles))
    mesh.compute_vertex_normals()
    return mesh


def export_3d_building_to_cityjson_with_dialog(
    floor_lineset: o3d.geometry.LineSet,
    wall_lineset: o3d.geometry.LineSet,
    roof_lineset: o3d.geometry.LineSet,
    cityjson_properties: Optional[dict] = None
):
    """Export a 3D building (floor, walls, roof) to CityJSON via a save dialog.

    Repairs invalid polygons (removes duplicates, closes ring, ensures 3 unique vertices).

    Args:
        floor_lineset (o3d.geometry.LineSet): Floor boundary as LineSet.
        wall_lineset (o3d.geometry.LineSet): Wall boundary as LineSet.
        roof_lineset (o3d.geometry.LineSet): Roof boundary as LineSet.
        cityjson_properties (dict, optional): Extra CityJSON properties.

    Returns:
        None

    Raises:
        ValueError: If all surfaces are invalid or empty.
    """
    def lineset_to_closed_ring(lineset):
        """Convert a LineSet to a closed ring of vertex indices."""
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
        """Repair a ring: remove consecutive duplicates, close, ensure 3 unique vertices."""
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
    for i, (ring, pts, name) in enumerate(zip(rings, all_points, surface_names)):
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
