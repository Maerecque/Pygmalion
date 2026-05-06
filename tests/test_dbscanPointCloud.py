"""Tests for Source/dbscanPointCloud.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.dbscanPointCloud import pointcloud_dbscan


def _make_pcd(n: int = 60) -> o3d.geometry.PointCloud:
    """Two compact clusters well separated in XY."""
    cluster_a = np.random.rand(n // 2, 3) * 0.1           # near origin
    cluster_b = np.random.rand(n // 2, 3) * 0.1 + 10.0    # far from origin
    pts = np.vstack([cluster_a, cluster_b])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pts), 1)))
    return pcd


class TestPointCloudDbscan:
    # ── Pure-logic tests ───────────────────────────────────────────────────────

    def test_returns_point_cloud(self):
        pcd = _make_pcd()
        result = pointcloud_dbscan(
            pcd, eps=0.5, min_samples=2,
            visualize_all=False, keep_only_labels=True,
            visualize_only_labels=False, keep_no_labels=False,
            visualize_no_labels=False,
        )
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_labeled_points_kept_with_keep_only_labels(self):
        """keep_only_labels=True should remove the noise cluster (label -1 → dark blue)."""
        pcd = _make_pcd(60)
        result = pointcloud_dbscan(
            pcd, eps=0.5, min_samples=2,
            keep_only_labels=True,
            visualize_all=False,
            visualize_only_labels=False,
            keep_no_labels=False,
            visualize_no_labels=False,
        )
        assert len(result.points) > 0

    # ── Mock tests ─────────────────────────────────────────────────────────────

    def test_dbscan_fit_predict_called(self):
        """sklearn DBSCAN.fit_predict is called with the XYZ point array."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_dbscan_cls:
            mock_instance = MagicMock()
            mock_instance.fit_predict.return_value = np.zeros(20, dtype=int)
            mock_dbscan_cls.return_value = mock_instance

            pointcloud_dbscan(
                pcd, eps=0.5, min_samples=2,
                visualize_all=True,          # short-circuit before color filtering
                keep_only_labels=False,
                keep_no_labels=False,
                visualize_only_labels=False,
                visualize_no_labels=False,
            )
            mock_instance.fit_predict.assert_called_once()
            called_array = mock_instance.fit_predict.call_args[0][0]
            assert called_array.shape == (20, 3)

    def test_dbscan_constructed_with_correct_params(self):
        """DBSCAN constructor receives the eps, min_samples, metric, etc. supplied."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_dbscan_cls:
            mock_instance = MagicMock()
            mock_instance.fit_predict.return_value = np.zeros(20, dtype=int)
            mock_dbscan_cls.return_value = mock_instance

            pointcloud_dbscan(
                pcd, eps=0.25, min_samples=5, metric="cityblock",
                algorithm="ball_tree", leaf_size=20,
                visualize_all=True,
                keep_only_labels=False, keep_no_labels=False,
                visualize_only_labels=False, visualize_no_labels=False,
            )
            mock_dbscan_cls.assert_called_once_with(
                eps=0.25,
                min_samples=5,
                metric="cityblock",
                algorithm="ball_tree",
                leaf_size=20,
            )

    def test_visualize_all_returns_colored_pcd(self):
        """visualize_all=True returns pcd after coloring, before filtering."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_cls, \
             patch("Source.dbscanPointCloud.open_point_cloud_editor"):
            mock_inst = MagicMock()
            mock_inst.fit_predict.return_value = np.zeros(20, dtype=int)
            mock_cls.return_value = mock_inst

            result = pointcloud_dbscan(
                pcd, visualize_all=True,
                keep_only_labels=False, keep_no_labels=False,
                visualize_only_labels=False, visualize_no_labels=False,
            )
            assert result is pcd

    def test_visualize_all_opens_editor(self):
        """When visualize_all=True, open_point_cloud_editor is called once."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_cls, \
             patch("Source.dbscanPointCloud.open_point_cloud_editor") as mock_editor:
            mock_inst = MagicMock()
            mock_inst.fit_predict.return_value = np.zeros(20, dtype=int)
            mock_cls.return_value = mock_inst

            pointcloud_dbscan(
                pcd, visualize_all=True,
                keep_only_labels=False, keep_no_labels=False,
                visualize_only_labels=False, visualize_no_labels=False,
            )
            mock_editor.assert_called_once()

    def test_open_editor_not_called_when_visualize_false(self):
        """When all visualize flags are False, open_point_cloud_editor is not called."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_cls, \
             patch("Source.dbscanPointCloud.open_point_cloud_editor") as mock_editor:
            mock_inst = MagicMock()
            mock_inst.fit_predict.return_value = np.zeros(20, dtype=int)
            mock_cls.return_value = mock_inst

            pointcloud_dbscan(
                pcd, visualize_all=False,
                keep_only_labels=True,
                visualize_only_labels=False,
                keep_no_labels=False,
                visualize_no_labels=False,
            )
            mock_editor.assert_not_called()

    def test_label_count_printed(self, capsys):
        """The number of DBSCAN labels is printed to stdout."""
        pcd = _make_pcd(20)
        with patch("Source.dbscanPointCloud.DBSCAN") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.fit_predict.return_value = np.array([0] * 10 + [1] * 10)
            mock_cls.return_value = mock_inst

            pointcloud_dbscan(
                pcd, visualize_all=True,
                keep_only_labels=False, keep_no_labels=False,
                visualize_only_labels=False, visualize_no_labels=False,
            )
        out = capsys.readouterr().out
        assert "label" in out.lower()
