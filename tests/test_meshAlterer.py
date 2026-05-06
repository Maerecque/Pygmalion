"""Tests for Source/meshAlterer.py (filter_vertices_and_faces, o3d_to_cityjson)"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.meshAlterer import filter_vertices_and_faces, o3d_to_cityjson
import open3d as o3d


def _make_simple_mesh() -> o3d.geometry.TriangleMesh:
    """Tetrahedron mesh: 4 vertices, 4 triangles."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]])
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0]] * 4)
    )
    return mesh


class TestFilterVerticesAndFaces:
    def test_keeps_all_vertices_below_threshold(self):
        mesh = _make_simple_mesh()
        distances = np.array([0.01, 0.02, 0.03, 0.04])
        verts, colors, faces = filter_vertices_and_faces(mesh, distances, distance_threshold=0.1)
        assert len(verts) == 4

    def test_removes_vertices_above_threshold(self):
        mesh = _make_simple_mesh()
        distances = np.array([0.01, 0.01, 0.01, 10.0])  # last vertex too far
        verts, colors, faces = filter_vertices_and_faces(mesh, distances, distance_threshold=0.1)
        assert len(verts) == 3

    def test_faces_referencing_removed_vertex_are_dropped(self):
        mesh = _make_simple_mesh()
        # Remove vertex 3 (index 3) – all faces using it should be dropped
        distances = np.array([0.01, 0.01, 0.01, 10.0])
        verts, colors, faces = filter_vertices_and_faces(mesh, distances, distance_threshold=0.1)
        # Only triangles [0,1,2] survives; the three that use vertex 3 are gone
        assert len(faces) == 1

    def test_zero_threshold_removes_all(self):
        mesh = _make_simple_mesh()
        distances = np.array([0.1, 0.1, 0.1, 0.1])
        verts, colors, faces = filter_vertices_and_faces(mesh, distances, distance_threshold=0.0)
        assert len(verts) == 0
        assert len(faces) == 0

    def test_output_colors_match_kept_vertices(self):
        mesh = _make_simple_mesh()
        distances = np.array([0.01, 10.0, 0.01, 0.01])
        verts, colors, faces = filter_vertices_and_faces(mesh, distances, distance_threshold=0.1)
        assert len(verts) == len(colors)


class TestO3dToCityJson:
    def test_output_type_is_dict(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh)
        assert isinstance(result, dict)

    def test_cityjson_type_field(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh)
        assert result["type"] == "CityJSON"

    def test_cityjson_version_field(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh)
        assert result["version"] == "1.1"

    def test_custom_cityobject_id(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="building_42")
        assert "building_42" in result["CityObjects"]

    def test_custom_obj_type(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t", obj_type="TINRelief")
        assert result["CityObjects"]["t"]["type"] == "TINRelief"

    def test_custom_lod(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t", lod="2.0")
        geometry = result["CityObjects"]["t"]["geometry"]
        assert geometry[0]["lod"] == "2.0"

    def test_vertices_match_mesh(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t")
        expected = np.asarray(mesh.vertices).tolist()
        assert result["vertices"] == expected

    def test_boundaries_are_triangles(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t")
        geometry = result["CityObjects"]["t"]["geometry"]
        solid_boundaries = geometry[0]["boundaries"][0]
        for face_wrapper in solid_boundaries:
            assert len(face_wrapper[0]) == 3  # each face is a triangle

    def test_geometry_type_is_solid(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t")
        assert result["CityObjects"]["t"]["geometry"][0]["type"] == "Solid"

    def test_boundary_count_matches_triangle_count(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, cityobject_id="t")
        solid_boundaries = result["CityObjects"]["t"]["geometry"][0]["boundaries"][0]
        assert len(solid_boundaries) == len(mesh.triangles)


class TestFilterVerticesAndFacesMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_np_where_called_once(self):
        """np.where is used exactly once to find the kept vertex indices."""
        mesh = _make_simple_mesh()
        distances = np.array([0.01, 0.01, 0.01, 0.01])
        with patch("numpy.where", wraps=np.where) as mock_where:
            filter_vertices_and_faces(mesh, distances, 0.1)
            mock_where.assert_called_once()

    def test_tqdm_wraps_triangle_iteration(self):
        """tqdm is called to wrap the face-filtering loop."""
        mesh = _make_simple_mesh()
        distances = np.array([0.01, 0.01, 0.01, 0.01])
        with patch("Source.meshAlterer.tqdm", wraps=__import__("tqdm").tqdm) as mock_tqdm:
            filter_vertices_and_faces(mesh, distances, 0.1)
            mock_tqdm.assert_called_once()


class TestO3dToCityJsonMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_np_asarray_called_for_vertices_and_triangles(self):
        """np.asarray is called at least twice: once for vertices, once for triangles."""
        mesh = _make_simple_mesh()
        with patch("numpy.asarray", wraps=np.asarray) as mock_asarray:
            o3d_to_cityjson(mesh, cityobject_id="t")
            assert mock_asarray.call_count >= 2

    def test_mock_mesh_vertices_and_triangles_accessed(self):
        """The function accesses .vertices and .triangles on the supplied mesh object."""
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        mock_mesh.triangles = np.array([[0, 1, 2]])
        result = o3d_to_cityjson(mock_mesh, cityobject_id="mock_id")
        assert "mock_id" in result["CityObjects"]

    def test_default_parameters_produce_valid_cityjson(self):
        """Default call with only mesh produces a structurally valid CityJSON dict."""
        mesh = _make_simple_mesh()
        with patch("Source.meshAlterer.np.asarray", wraps=np.asarray):
            result = o3d_to_cityjson(mesh)
        assert result["type"] == "CityJSON"
        assert "obj1" in result["CityObjects"]
