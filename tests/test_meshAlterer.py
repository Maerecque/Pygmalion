"""Tests for Source/meshAlterer.py"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.meshAlterer import (
    filter_vertices_and_faces,
    o3d_to_cityjson,
    compute_distances_to_point_cloud,
    mesh_simple_downsample,
    repair_mesh,
    combine_meshes,
    transform_pcd_to_mesh,
)
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

    def test_no_crs_produces_no_metadata(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh)
        assert "metadata" not in result

    def test_epsg_shorthand_is_converted_to_ogc_uri(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, crs="EPSG:28992")
        assert result["metadata"]["referenceSystem"] == \
            "https://www.opengis.net/def/crs/EPSG/0/28992"

    def test_full_ogc_uri_is_passed_through_unchanged(self):
        mesh = _make_simple_mesh()
        uri = "https://www.opengis.net/def/crs/EPSG/0/4326"
        result = o3d_to_cityjson(mesh, crs=uri)
        assert result["metadata"]["referenceSystem"] == uri

    def test_crs_lowercase_epsg_is_accepted(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, crs="epsg:28992")
        assert result["metadata"]["referenceSystem"] == \
            "https://www.opengis.net/def/crs/EPSG/0/28992"

    def test_metadata_key_absent_when_crs_is_none(self):
        mesh = _make_simple_mesh()
        result = o3d_to_cityjson(mesh, crs=None)
        assert "metadata" not in result

    def test_invalid_crs_non_numeric_epsg_raises(self):
        mesh = _make_simple_mesh()
        with pytest.raises(ValueError, match="EPSG code must be numeric"):
            o3d_to_cityjson(mesh, crs="EPSG:abc")

    def test_invalid_crs_random_string_raises(self):
        mesh = _make_simple_mesh()
        with pytest.raises(ValueError, match="Invalid CRS format"):
            o3d_to_cityjson(mesh, crs="RD_New")

    def test_invalid_crs_epsg_without_code_raises(self):
        mesh = _make_simple_mesh()
        with pytest.raises(ValueError, match="EPSG code must be numeric"):
            o3d_to_cityjson(mesh, crs="EPSG:")


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


# ── compute_distances_to_point_cloud ──────────────────────────────────────────

def _make_simple_pcd() -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]])
    )
    return pcd


class TestComputeDistancesToPointCloud:
    def test_returns_ndarray(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        result = compute_distances_to_point_cloud(mesh, pcd)
        assert isinstance(result, np.ndarray)

    def test_length_matches_vertex_count(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        result = compute_distances_to_point_cloud(mesh, pcd)
        assert len(result) == len(mesh.vertices)

    def test_distances_are_non_negative(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        result = compute_distances_to_point_cloud(mesh, pcd)
        assert np.all(result >= 0)

    def test_identical_mesh_and_pcd_have_zero_distances(self):
        """When mesh vertices == pcd points, all distances should be ~0."""
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        result = compute_distances_to_point_cloud(mesh, pcd)
        assert np.allclose(result, 0.0, atol=1e-6)

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_tqdm_wraps_vertex_iteration(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        with patch("Source.meshAlterer.tqdm", side_effect=lambda it, **kw: it) as mock_tqdm:
            compute_distances_to_point_cloud(mesh, pcd)
            mock_tqdm.assert_called_once()

    def test_kdtreeflann_constructed_once(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        with patch("Source.meshAlterer.o3d.geometry.KDTreeFlann",
                   wraps=o3d.geometry.KDTreeFlann) as mock_kd:
            compute_distances_to_point_cloud(mesh, pcd)
            mock_kd.assert_called_once_with(pcd)


# ── mesh_simple_downsample ────────────────────────────────────────────────────

class TestMeshSimpleDownsample:
    def test_raises_on_empty_mesh(self):
        """Falsy-checked empty mesh raises ValueError."""
        with pytest.raises((ValueError, RuntimeError)):
            mesh_simple_downsample(None, _make_simple_pcd())  # type: ignore

    def test_raises_on_empty_pcd(self):
        """Empty point cloud causes an error during KD-tree construction."""
        with pytest.raises((ValueError, RuntimeError)):
            mesh_simple_downsample(_make_simple_mesh(), o3d.geometry.PointCloud())

    def test_returns_triangle_mesh(self):
        result = mesh_simple_downsample(_make_simple_mesh(), _make_simple_pcd(), distance_threshold=10.0)
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_high_threshold_keeps_all_vertices(self):
        """Very high threshold keeps everything."""
        mesh = _make_simple_mesh()
        result = mesh_simple_downsample(mesh, _make_simple_pcd(), distance_threshold=1000.0)
        assert len(result.vertices) > 0

    def test_zero_threshold_removes_all_triangles(self):
        """Threshold of 0 removes all vertices that are not exactly on the cloud."""
        mesh = _make_simple_mesh()
        # Point cloud is offset so all mesh vertex distances > 0
        offset_pcd = o3d.geometry.PointCloud()
        offset_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) + np.array([100.0, 0.0, 0.0])
        )
        result = mesh_simple_downsample(mesh, offset_pcd, distance_threshold=0.0)
        assert len(result.triangles) == 0

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_compute_distances_called(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        with patch("Source.meshAlterer.compute_distances_to_point_cloud",
                   wraps=compute_distances_to_point_cloud) as mock_dist:
            mesh_simple_downsample(mesh, pcd, distance_threshold=10.0)
            mock_dist.assert_called_once()

    def test_visualize_calls_draw_geometries(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        with patch("Source.meshAlterer.o3d.visualization.draw_geometries") as mock_draw:
            mesh_simple_downsample(mesh, pcd, distance_threshold=10.0, visualize_mesh=True)
            mock_draw.assert_called_once()

    def test_no_visualize_does_not_call_draw(self):
        mesh = _make_simple_mesh()
        pcd = _make_simple_pcd()
        with patch("Source.meshAlterer.o3d.visualization.draw_geometries") as mock_draw:
            mesh_simple_downsample(mesh, pcd, distance_threshold=10.0, visualize_mesh=False)
            mock_draw.assert_not_called()


# ── repair_mesh ───────────────────────────────────────────────────────────────

class TestRepairMesh:
    def test_single_mesh_returns_triangle_mesh(self):
        result = repair_mesh(_make_simple_mesh())
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_list_of_meshes_returns_triangle_mesh(self):
        result = repair_mesh([_make_simple_mesh(), _make_simple_mesh()])
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_result_has_vertices(self):
        result = repair_mesh(_make_simple_mesh())
        assert len(result.vertices) > 0

    def test_result_has_triangles(self):
        result = repair_mesh(_make_simple_mesh())
        assert len(result.triangles) > 0

    def test_two_meshes_combined_have_more_vertices(self):
        single = repair_mesh(_make_simple_mesh())
        combined = repair_mesh([_make_simple_mesh(), _make_simple_mesh()])
        assert len(combined.vertices) >= len(single.vertices)

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_fill_holes_called_if_not_watertight(self):
        """fill_holes is called when the mesh has holes."""
        import trimesh as tm
        mesh = _make_simple_mesh()
        with patch.object(tm.Trimesh, "fill_holes") as mock_fill:
            with patch.object(tm.Trimesh, "is_watertight", new_callable=lambda: property(lambda self: False)):
                repair_mesh(mesh)
                mock_fill.assert_called_once()


# ── combine_meshes ────────────────────────────────────────────────────────────

class TestCombineMeshes:
    def test_returns_triangle_mesh(self):
        result = combine_meshes([_make_simple_mesh()])
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_single_mesh_vertex_count_preserved(self):
        mesh = _make_simple_mesh()
        result = combine_meshes([mesh])
        assert len(result.vertices) == len(mesh.vertices)

    def test_two_meshes_vertex_count_sum(self):
        mesh = _make_simple_mesh()
        result = combine_meshes([mesh, mesh])
        assert len(result.vertices) == 2 * len(mesh.vertices)

    def test_two_meshes_triangle_count_sum(self):
        mesh = _make_simple_mesh()
        result = combine_meshes([mesh, mesh])
        assert len(result.triangles) == 2 * len(mesh.triangles)

    def test_empty_list_returns_empty_mesh(self):
        result = combine_meshes([])
        assert len(result.vertices) == 0

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_compute_vertex_normals_called(self):
        """Normals are recomputed — result has vertex normals."""
        result = combine_meshes([_make_simple_mesh()])
        assert result.has_vertex_normals()


# ── transform_pcd_to_mesh ─────────────────────────────────────────────────────

class TestTransformPcdToMesh:
    def _make_pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.random.rand(20, 3).astype(float)
        )
        return pcd

    def test_returns_unstructured_grid(self):
        import pyvista as pv
        result = transform_pcd_to_mesh(self._make_pcd())
        assert isinstance(result, pv.UnstructuredGrid)

    def test_calls_delaunay_3d_by_default(self):
        """bool_3d_mesh=True (default) calls delaunay_3d."""
        import pyvista as pv
        mock_polydata = MagicMock(spec=pv.PolyData)
        mock_polydata.delaunay_3d.return_value = MagicMock(spec=pv.UnstructuredGrid)
        with patch("Source.meshAlterer.pv.PolyData", return_value=mock_polydata):
            transform_pcd_to_mesh(self._make_pcd(), bool_3d_mesh=True)
            mock_polydata.delaunay_3d.assert_called_once()

    def test_calls_delaunay_2d_when_not_3d(self):
        """bool_3d_mesh=False calls delaunay_2d."""
        import pyvista as pv
        mock_polydata = MagicMock(spec=pv.PolyData)
        mock_polydata.delaunay_2d.return_value = MagicMock(spec=pv.UnstructuredGrid)
        with patch("Source.meshAlterer.pv.PolyData", return_value=mock_polydata):
            transform_pcd_to_mesh(self._make_pcd(), bool_3d_mesh=False)
            mock_polydata.delaunay_2d.assert_called_once()

    def test_visualize_calls_shell_plot(self):
        """visualize_bool=True calls .plot() on the extracted geometry shell."""
        import pyvista as pv
        mock_volume = MagicMock(spec=pv.UnstructuredGrid)
        mock_shell = MagicMock()
        mock_volume.extract_geometry.return_value = mock_shell
        mock_polydata = MagicMock(spec=pv.PolyData)
        mock_polydata.delaunay_3d.return_value = mock_volume
        with patch("Source.meshAlterer.pv.PolyData", return_value=mock_polydata):
            transform_pcd_to_mesh(self._make_pcd(), visualize_bool=True)
            mock_shell.plot.assert_called_once()

    def test_no_visualize_does_not_call_plot(self):
        import pyvista as pv
        mock_volume = MagicMock(spec=pv.UnstructuredGrid)
        mock_polydata = MagicMock(spec=pv.PolyData)
        mock_polydata.delaunay_3d.return_value = mock_volume
        with patch("Source.meshAlterer.pv.PolyData", return_value=mock_polydata):
            transform_pcd_to_mesh(self._make_pcd(), visualize_bool=False)
            mock_volume.extract_geometry.assert_not_called()
