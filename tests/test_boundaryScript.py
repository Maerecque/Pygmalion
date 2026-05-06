"""Tests for Source/boundaryScript.py — expand_boundary"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.boundaryScript import expand_boundary, ZeroExpansionError, NonFlatMeshError


def _flat_square_mesh(z: float = 0.0) -> o3d.geometry.TriangleMesh:
    """Unit square as a flat TriangleMesh (two triangles) at given z."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]
    ]))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]]))
    return mesh


def _flat_square_lineset(z: float = 0.0) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]
    ]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    return ls


def _flat_square_pcd(z: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]
    ]))
    return pcd


# ── ZeroExpansionError / NonFlatMeshError ─────────────────────────────────────

class TestCustomExceptions:
    def test_zero_expansion_error_is_exception(self):
        assert issubclass(ZeroExpansionError, Exception)

    def test_non_flat_mesh_error_is_exception(self):
        assert issubclass(NonFlatMeshError, Exception)


# ── expand_boundary ────────────────────────────────────────────────────────────

class TestExpandBoundary:
    # ── Pure logic ───────────────────────────────────────────────────────────

    def test_zero_expansion_raises(self):
        mesh = _flat_square_mesh()
        with pytest.raises(ZeroExpansionError):
            expand_boundary(mesh, expansion_size=0.0)

    def test_negative_expansion_raises(self):
        mesh = _flat_square_mesh()
        with pytest.raises(ZeroExpansionError):
            expand_boundary(mesh, expansion_size=-5.0)

    def test_non_flat_mesh_raises(self):
        """A mesh with vertices at different z values raises NonFlatMeshError."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]  # different z
        ]))
        mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
        with pytest.raises(NonFlatMeshError):
            expand_boundary(mesh, expansion_size=10.0)

    def test_mesh_input_returns_triangle_mesh(self):
        mesh = _flat_square_mesh()
        result = expand_boundary(mesh, expansion_size=10.0)
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_lineset_input_returns_lineset(self):
        ls = _flat_square_lineset()
        result = expand_boundary(ls, expansion_size=10.0)
        assert isinstance(result, o3d.geometry.LineSet)

    def test_pcd_input_returns_point_cloud(self):
        pcd = _flat_square_pcd()
        result = expand_boundary(pcd, expansion_size=10.0)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_expanded_pcd_has_points(self):
        """The result PointCloud is non-empty after expansion."""
        pcd = _flat_square_pcd()
        result = expand_boundary(pcd, expansion_size=10.0)
        assert len(result.points) > 0

    def test_z_value_preserved_in_result(self):
        """The z-coordinate of the original mesh is preserved in output points."""
        z = 5.0
        mesh = _flat_square_mesh(z=z)
        result = expand_boundary(mesh, expansion_size=10.0)
        z_vals = np.asarray(result.vertices)[:, 2]
        assert np.allclose(z_vals, z)

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_shapely_buffer_called(self):
        """Shapely Polygon.buffer is called to expand the 2D boundary."""
        mesh = _flat_square_mesh()
        with patch("Source.boundaryScript.Polygon") as mock_polygon_cls:
            mock_polygon = MagicMock()
            mock_polygon_cls.return_value = mock_polygon
            mock_expanded = MagicMock()
            mock_expanded.exterior.coords = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
            mock_polygon.buffer.return_value = mock_expanded
            expand_boundary(mesh, expansion_size=10.0)
            mock_polygon.buffer.assert_called_once()

    def test_visualization_not_called_when_disabled(self):
        """opce is NOT called when point_visualization=False."""
        mesh = _flat_square_mesh()
        with patch("Source.boundaryScript.opce") as mock_opce:
            expand_boundary(mesh, expansion_size=10.0, point_visualization=False)
            mock_opce.assert_not_called()

    def test_contour_to_lineset_called_for_mesh_input(self):
        """contour_to_lineset is called to build an intermediate lineset."""
        mesh = _flat_square_mesh()
        with patch("Source.boundaryScript.contour_to_lineset",
                   wraps=__import__("Source.linesetTools",
                                    fromlist=["contour_to_lineset"]).contour_to_lineset) as mock_c2l:
            expand_boundary(mesh, expansion_size=10.0)
            mock_c2l.assert_called()

    def test_np_asarray_called_to_read_vertices(self):
        """np.asarray is used to extract vertex data from the mesh."""
        mesh = _flat_square_mesh()
        with patch("numpy.asarray", wraps=np.asarray) as mock_asarray:
            expand_boundary(mesh, expansion_size=10.0)
            mock_asarray.assert_called()
