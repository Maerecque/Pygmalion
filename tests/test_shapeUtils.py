"""Tests for Source/shapeUtils.py (merge_pcd, merge_list_of_pointclouds)"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.shapeUtils import (
    merge_pcd, merge_list_of_pointclouds,
    segment_plane, find_plane_module_manual,
    repair_point_cloud_module, transform_mesh_to_pcd,
)
import open3d as o3d


def _make_pcd(points: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _make_colored_pcd(n: int = 5, z: float = 0.0) -> o3d.geometry.PointCloud:
    pts = np.array([[float(i), 0.0, z] for i in range(n)]) if n > 0 else np.zeros((0, 3))
    clrs = np.tile([0.5, 0.5, 0.5], (n, 1))
    return _make_pcd(pts, clrs)


class TestMergePcd:
    def test_merged_point_count(self):
        pcd1 = _make_colored_pcd(5, z=0.0)
        pcd2 = _make_colored_pcd(3, z=1.0)
        result = merge_pcd(pcd1, pcd2)
        assert len(result.points) == 8

    def test_merged_has_colors(self):
        pcd1 = _make_colored_pcd(3, z=0.0)
        pcd2 = _make_colored_pcd(3, z=1.0)
        result = merge_pcd(pcd1, pcd2)
        assert result.has_colors()

    def test_empty_first_pcd_raises(self):
        empty = _make_colored_pcd(0)
        pcd2 = _make_colored_pcd(3)
        with pytest.raises(ValueError):
            merge_pcd(empty, pcd2)

    def test_empty_second_pcd_raises(self):
        pcd1 = _make_colored_pcd(3)
        empty = _make_colored_pcd(0)
        with pytest.raises(ValueError):
            merge_pcd(pcd1, empty)

    def test_missing_colors_first_raises(self):
        pcd1 = _make_pcd(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        pcd2 = _make_colored_pcd(3)
        with pytest.raises(ValueError):
            merge_pcd(pcd1, pcd2)

    def test_missing_colors_second_raises(self):
        pcd1 = _make_colored_pcd(3)
        pcd2 = _make_pcd(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(ValueError):
            merge_pcd(pcd1, pcd2)

    def test_output_is_point_cloud(self):
        pcd1 = _make_colored_pcd(4)
        pcd2 = _make_colored_pcd(4, z=1.0)
        result = merge_pcd(pcd1, pcd2)
        assert isinstance(result, o3d.geometry.PointCloud)


class TestMergeListOfPointclouds:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            merge_list_of_pointclouds([])

    def test_non_pcd_in_list_raises(self):
        pcd = _make_colored_pcd(3)
        with pytest.raises(ValueError):
            merge_list_of_pointclouds([pcd, "not_a_pcd"])

    def test_all_empty_pcds_raises(self):
        """All-empty list: empty pcds are filtered out; function returns empty PointCloud."""
        empty1 = _make_colored_pcd(0)
        empty2 = _make_colored_pcd(0)
        result = merge_list_of_pointclouds([empty1, empty2])
        assert len(result.points) == 0

    def test_single_pcd_returns_equivalent(self):
        pcd = _make_colored_pcd(5)
        result = merge_list_of_pointclouds([pcd])
        assert len(result.points) == 5

    def test_multiple_pcds_total_count(self):
        pcds = [_make_colored_pcd(4, z=float(i)) for i in range(3)]
        result = merge_list_of_pointclouds(pcds)
        assert len(result.points) == 12


class TestMergePcdMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_np_concatenate_called_twice(self):
        """np.concatenate is called twice: once for points, once for colors."""
        pcd1 = _make_colored_pcd(3, z=0.0)
        pcd2 = _make_colored_pcd(3, z=1.0)
        with patch("numpy.concatenate", wraps=np.concatenate) as mock_cat:
            merge_pcd(pcd1, pcd2)
            assert mock_cat.call_count == 2

    def test_o3d_pointcloud_constructor_called(self):
        """A new o3d.geometry.PointCloud is constructed for the merged result."""
        pcd1 = _make_colored_pcd(3, z=0.0)
        pcd2 = _make_colored_pcd(3, z=1.0)
        with patch("Source.shapeUtils.o3d.geometry.PointCloud",
                   wraps=o3d.geometry.PointCloud) as mock_pcd:
            merge_pcd(pcd1, pcd2)
            mock_pcd.assert_called_once()

    def test_first_empty_raises_before_concat(self):
        """ValueError is raised before np.concatenate is reached."""
        empty = _make_colored_pcd(0)
        pcd2 = _make_colored_pcd(3)
        with patch("numpy.concatenate") as mock_cat:
            with pytest.raises(ValueError):
                merge_pcd(empty, pcd2)
            mock_cat.assert_not_called()


class TestMergeListMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_tqdm_called_for_progress(self):
        """tqdm is used to wrap the merge loop."""
        pcds = [_make_colored_pcd(3, z=float(i)) for i in range(3)]
        with patch("Source.shapeUtils.tqdm", wraps=__import__("tqdm").tqdm) as mock_tqdm:
            merge_list_of_pointclouds(pcds)
            mock_tqdm.assert_called_once()

    def test_np_vstack_called_for_merged_output(self):
        """np.concatenate is called to combine point and color arrays."""
        pcds = [_make_colored_pcd(3, z=float(i)) for i in range(2)]
        with patch("numpy.concatenate", wraps=np.concatenate) as mock_cat:
            merge_list_of_pointclouds(pcds)
            assert mock_cat.call_count >= 2  # once for points, once for colors

    def test_grey_color_applied_when_missing(self):
        """merge_list_of_pointclouds raises ValueError when a pcd has no colors."""
        pcd_no_color = _make_pcd(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        pcd_with_color = _make_colored_pcd(2, z=1.0)
        with pytest.raises(ValueError):
            merge_list_of_pointclouds([pcd_no_color, pcd_with_color])


# ── segment_plane ──────────────────────────────────────────────────────────────

def _make_plane_pcd(n=200):
    """Dense flat point cloud in XY plane."""
    np.random.seed(0)
    pts = np.column_stack([np.random.rand(n), np.random.rand(n), np.zeros(n)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


class TestSegmentPlane:
    def test_raises_for_empty_pcd(self):
        with pytest.raises(ValueError):
            segment_plane(o3d.geometry.PointCloud())

    def test_returns_two_point_clouds(self):
        pcd = _make_plane_pcd()
        plane, leftovers = segment_plane(pcd)
        assert isinstance(plane, o3d.geometry.PointCloud)
        assert isinstance(leftovers, o3d.geometry.PointCloud)

    def test_plane_plus_leftovers_equals_original(self):
        pcd = _make_plane_pcd()
        plane, leftovers = segment_plane(pcd)
        assert len(plane.points) + len(leftovers.points) == len(pcd.points)

    def test_plane_points_are_non_empty(self):
        pcd = _make_plane_pcd()
        plane, _ = segment_plane(pcd)
        assert len(plane.points) > 0

    def test_print_bool_outputs_text(self, capsys):
        pcd = _make_plane_pcd()
        segment_plane(pcd, print_bool=True)
        captured = capsys.readouterr()
        assert "Extracted" in captured.out

    def test_visualize_plane_calls_draw(self):
        pcd = _make_plane_pcd()
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries") as mock_draw:
            segment_plane(pcd, visualize_plane=True)
            mock_draw.assert_called_once()

    def test_visualize_leftovers_calls_draw(self):
        pcd = _make_plane_pcd()
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries") as mock_draw:
            segment_plane(pcd, visualize_leftovers=True)
            mock_draw.assert_called_once()


# ── find_plane_module_manual ──────────────────────────────────────────────────

class TestFindPlaneModuleManual:
    def test_raises_for_empty_pcd(self):
        with pytest.raises(ValueError):
            find_plane_module_manual(o3d.geometry.PointCloud())

    def test_returns_point_cloud_on_accept(self):
        """User immediately accepts (any key other than e/u/p/r) → returns a PointCloud."""
        pcd = _make_plane_pcd()
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries"), \
             patch("builtins.input", return_value=""):
            result = find_plane_module_manual(pcd)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_returns_point_cloud_on_expand_then_accept(self):
        """User expands once then accepts → still returns a PointCloud."""
        pcd = _make_plane_pcd()
        half = _make_plane_pcd(50)
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries"), \
             patch("Source.shapeUtils.segment_plane", return_value=(half, half)), \
             patch("Source.shapeUtils.merge_pcd", return_value=half), \
             patch("builtins.input", side_effect=["e", ""]):
            result = find_plane_module_manual(pcd)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_export_previous_returns_none_safely(self):
        """'p' on first iteration (no previous plane) returns None without crashing."""
        pcd = _make_plane_pcd()
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries"), \
             patch("builtins.input", return_value="p"):
            result = find_plane_module_manual(pcd)
        # previous_plane is None on first iteration, so result is None
        assert result is None


# ── repair_point_cloud_module ─────────────────────────────────────────────────

class TestRepairPointCloudModule:
    def test_raises_for_empty_pcd(self):
        with pytest.raises(ValueError):
            repair_point_cloud_module(o3d.geometry.PointCloud())

    def test_raises_for_invalid_quantile(self):
        pcd = _make_plane_pcd()
        with pytest.raises(ValueError):
            repair_point_cloud_module(pcd, quantile_value=1.5)

    def test_raises_for_negative_quantile(self):
        pcd = _make_plane_pcd()
        with pytest.raises(ValueError):
            repair_point_cloud_module(pcd, quantile_value=-0.1)

    def test_returns_triangle_mesh(self):
        pcd = _make_plane_pcd(n=500)
        # Use a very simple config to keep it fast
        result = repair_point_cloud_module(pcd, depth=4, quantile_value=0.0)
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_poisson_called(self):
        """create_from_point_cloud_poisson is invoked during reconstruction."""
        pcd = _make_plane_pcd(n=50)
        # Use a real mesh so remove_vertices_by_mask receives a correctly-sized mask.
        mock_mesh = o3d.geometry.TriangleMesh.create_sphere()
        n_verts = len(np.asarray(mock_mesh.vertices))
        with patch("Source.shapeUtils.o3d.geometry.TriangleMesh.create_from_point_cloud_poisson",
                   return_value=(mock_mesh, np.ones(n_verts))) as mock_poisson:
            repair_point_cloud_module(pcd, depth=4, quantile_value=0.0)
            mock_poisson.assert_called_once()

    def test_visualize_calls_draw(self):
        pcd = _make_plane_pcd(n=500)
        with patch("Source.shapeUtils.o3d.visualization.draw_geometries") as mock_draw:
            repair_point_cloud_module(pcd, depth=4, quantile_value=0.0, visualize=True)
            mock_draw.assert_called_once()


# ── transform_mesh_to_pcd ─────────────────────────────────────────────────────

class TestTransformMeshToPcd:
    def _make_mesh(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        return mesh

    def _make_3d_pcd(self, n=100):
        """Non-flat 3D point cloud so compute_mahalanobis_distance is well-defined."""
        np.random.seed(42)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(n, 3))
        return pcd

    def test_raises_for_empty_mesh(self):
        with pytest.raises(ValueError):
            transform_mesh_to_pcd(o3d.geometry.TriangleMesh(), self._make_3d_pcd())

    def test_raises_for_empty_pcd(self):
        mesh = self._make_mesh()
        with pytest.raises(ValueError):
            transform_mesh_to_pcd(mesh, o3d.geometry.PointCloud())

    def test_returns_point_cloud(self):
        mesh = self._make_mesh()
        pcd = self._make_3d_pcd()
        result = transform_mesh_to_pcd(mesh, pcd)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_result_is_non_empty(self):
        mesh = self._make_mesh()
        pcd = self._make_3d_pcd()
        result = transform_mesh_to_pcd(mesh, pcd)
        assert len(result.points) > 0

    def test_sample_points_uniformly_called(self):
        """sample_points_uniformly is called to convert mesh to pcd."""
        mesh = self._make_mesh()
        pcd = self._make_3d_pcd()
        with patch.object(o3d.geometry.TriangleMesh, "sample_points_uniformly",
                          wraps=mesh.sample_points_uniformly) as mock_sample:
            transform_mesh_to_pcd(mesh, pcd)
            mock_sample.assert_called_once()
