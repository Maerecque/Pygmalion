"""Tests for Source/pointCloudAltering.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.pointCloudAltering import (
    grid_subsampling,
    alter_point_density,
    remove_noise_statistical,
    merge_point_clouds,
    get_difference_point_cloud,
    combine_point_cloud,
)


def _make_pcd(n: int = 10, spread: float = 1.0) -> o3d.geometry.PointCloud:
    pts = np.random.rand(n, 3) * spread
    clrs = np.tile([0.5, 0.5, 0.5], (n, 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(clrs)
    return pcd


# ── grid_subsampling ───────────────────────────────────────────────────────────

class TestGridSubsampling:
    def test_returns_point_cloud(self):
        pcd = _make_pcd(100, spread=5.0)
        result = grid_subsampling(pcd, voxel_size=0.5, print_result=False)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_reduces_point_count(self):
        pcd = _make_pcd(500, spread=0.1)  # dense → large voxel removes many
        result = grid_subsampling(pcd, voxel_size=0.5, print_result=False)
        assert len(result.points) <= len(pcd.points)

    def test_voxel_down_sample_called(self):
        """voxel_down_sample is delegated to Open3D internals."""
        mock_pcd = MagicMock()
        mock_result = _make_pcd(5)
        mock_pcd.voxel_down_sample.return_value = mock_result
        result = grid_subsampling(mock_pcd, voxel_size=0.1, print_result=False)
        mock_pcd.voxel_down_sample.assert_called_once_with(0.1)
        assert result is mock_result

    def test_print_suppressed(self, capsys):
        pcd = _make_pcd(10)
        grid_subsampling(pcd, voxel_size=1.0, print_result=False)
        assert capsys.readouterr().out == ""

    def test_print_enabled(self, capsys):
        pcd = _make_pcd(50)
        grid_subsampling(pcd, voxel_size=0.01, print_result=True)
        assert "point cloud" in capsys.readouterr().out.lower()


# ── alter_point_density ────────────────────────────────────────────────────────

class TestAlterPointDensity:
    def test_empty_pcd_raises(self):
        empty = o3d.geometry.PointCloud()
        with pytest.raises(ValueError, match="empty"):
            alter_point_density(empty)

    def test_non_positive_density_raises(self):
        pcd = _make_pcd(20)
        with pytest.raises(ValueError, match="positive"):
            alter_point_density(pcd, points_per_cm=0)

    def test_negative_density_raises(self):
        pcd = _make_pcd(20)
        with pytest.raises(ValueError):
            alter_point_density(pcd, points_per_cm=-1.0)

    def test_returns_point_cloud(self):
        pcd = _make_pcd(50, spread=5.0)
        result = alter_point_density(pcd, points_per_cm=1.0, print_result=False)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_voxel_down_sample_called_with_correct_size(self):
        """The voxel size passed is (1/points_per_cm) * 0.01."""
        expected_voxel = (1 / 2.0) * 0.01
        mock_pcd = MagicMock()
        mock_pcd.has_points.return_value = True
        mock_result = MagicMock()
        mock_pcd.voxel_down_sample.return_value = mock_result
        mock_bbox = MagicMock()
        mock_bbox.volume.return_value = 1.0
        mock_pcd.get_axis_aligned_bounding_box.return_value = mock_bbox
        alter_point_density(mock_pcd, points_per_cm=2.0, print_result=False)
        mock_pcd.voxel_down_sample.assert_called_once_with(
            voxel_size=pytest.approx(expected_voxel)
        )

    def test_bounding_box_volume_queried(self):
        """get_axis_aligned_bounding_box is called to compute volume."""
        mock_pcd = MagicMock()
        mock_pcd.has_points.return_value = True
        mock_bbox = MagicMock()
        mock_bbox.volume.return_value = 1.0
        mock_pcd.get_axis_aligned_bounding_box.return_value = mock_bbox
        mock_result = MagicMock()
        mock_pcd.voxel_down_sample.return_value = mock_result
        alter_point_density(mock_pcd, points_per_cm=1.0, print_result=False)
        mock_pcd.get_axis_aligned_bounding_box.assert_called_once()
        mock_bbox.volume.assert_called_once()


# ── remove_noise_statistical ───────────────────────────────────────────────────

class TestRemoveNoiseStatistical:
    def test_returns_point_cloud(self):
        pcd = _make_pcd(50, spread=1.0)
        result = remove_noise_statistical(pcd, print_removal_amount=False)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_remove_statistical_outlier_called(self):
        """remove_statistical_outlier is called with the supplied nb_neighbors and std_ratio."""
        mock_pcd = MagicMock()
        mock_cl = _make_pcd(25)
        mock_ind = list(range(25))
        mock_pcd.remove_statistical_outlier.return_value = (mock_cl, mock_ind)
        remove_noise_statistical(mock_pcd, nb_neighbors=15, std_ratio=1.5,
                                 print_removal_amount=False)
        mock_pcd.remove_statistical_outlier.assert_called_once_with(15, 1.5)

    def test_print_removal_amount_false_suppresses_output(self, capsys):
        pcd = _make_pcd(30)
        remove_noise_statistical(pcd, print_removal_amount=False)
        assert capsys.readouterr().out == ""

    def test_print_removal_amount_true_prints_message(self, capsys):
        pcd = _make_pcd(100, spread=10.0)
        remove_noise_statistical(pcd, nb_neighbors=5, std_ratio=0.1,
                                 print_removal_amount=True)
        out = capsys.readouterr().out
        # Some message about points removed (or none removed) should appear
        assert len(out) > 0


# ── merge_point_clouds ─────────────────────────────────────────────────────────

class TestMergePointClouds:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            merge_point_clouds([])

    def test_list_of_all_empty_pcds_raises(self):
        empty = o3d.geometry.PointCloud()
        with pytest.raises(ValueError):
            merge_point_clouds([empty, empty])

    def test_ndarray_converted_to_pcd(self):
        arr = np.random.rand(10, 3)
        result = merge_point_clouds([arr])
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_colorless_pcd_gets_grey(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(5, 3))
        result = merge_point_clouds([pcd])
        colors = np.asarray(result.colors)
        assert np.allclose(colors[0], [0.5, 0.5, 0.5])

    def test_total_point_count(self):
        pcd1 = _make_pcd(7)
        pcd2 = _make_pcd(8)
        result = merge_point_clouds([pcd1, pcd2])
        assert len(result.points) == 15

    def test_np_vstack_called_for_merge(self):
        """np.vstack is used to concatenate points and colors."""
        pcd1 = _make_pcd(5)
        pcd2 = _make_pcd(5)
        with patch("numpy.vstack", wraps=np.vstack) as mock_vstack:
            merge_point_clouds([pcd1, pcd2])
            assert mock_vstack.call_count >= 2


# ── get_difference_point_cloud / combine_point_cloud ──────────────────────────

class TestGetDifferencePointCloud:
    def test_returns_point_cloud(self):
        pcd1 = _make_pcd(20, spread=1.0)
        # pcd2 is a strict subset – select first 5 points
        pts2 = np.asarray(pcd1.points)[:5]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        result = get_difference_point_cloud(pcd1, pcd2)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_compute_point_cloud_distance_called(self):
        """compute_point_cloud_distance is called on input_pcd."""
        mock_pcd1 = MagicMock()
        pcd2 = _make_pcd(5)
        fake_distances = np.ones(10)
        mock_pcd1.compute_point_cloud_distance.return_value = fake_distances
        get_difference_point_cloud(mock_pcd1, pcd2)
        mock_pcd1.compute_point_cloud_distance.assert_called_once_with(pcd2)

    def test_result_excludes_overlapping_points(self):
        """Points with zero distance (present in both clouds) are excluded."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts)
        # pcd2 contains only the first point
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts[:1])
        result = get_difference_point_cloud(pcd1, pcd2)
        # The first point (distance 0) should be excluded
        result_pts = np.asarray(result.points)
        assert not any(np.allclose(p, [0.0, 0.0, 0.0]) for p in result_pts)


class TestCombinePointCloud:
    def test_returns_point_cloud(self):
        pcd1 = _make_pcd(10)
        pcd2 = _make_pcd(5)
        result = combine_point_cloud(pcd1, pcd2)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_compute_distance_called(self):
        mock_pcd1 = MagicMock()
        pcd2 = _make_pcd(5)
        fake_distances = np.zeros(10)
        mock_pcd1.compute_point_cloud_distance.return_value = fake_distances
        combine_point_cloud(mock_pcd1, pcd2)
        mock_pcd1.compute_point_cloud_distance.assert_called_once_with(pcd2)
