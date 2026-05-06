"""Tests for Source/shapeUtils.py (merge_pcd, merge_list_of_pointclouds)"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.shapeUtils import merge_pcd, merge_list_of_pointclouds
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
