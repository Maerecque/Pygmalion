"""Tests for Source/pointCloudEditor.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.pointCloudEditor import open_point_cloud_editor, open_mesh_and_lineset_viewer


def _make_pcd(n: int = 5) -> o3d.geometry.PointCloud:
    pts = np.random.rand(n, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _make_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    )
    return mesh


def _make_lineset() -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )
    ls.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    )
    return ls


# ── open_point_cloud_editor ────────────────────────────────────────────────────

class TestOpenPointCloudEditor:
    # ── Pure logic ───────────────────────────────────────────────────────────

    def test_empty_pcd_raises_value_error(self):
        empty = o3d.geometry.PointCloud()
        with pytest.raises(ValueError, match="empty"):
            open_point_cloud_editor(empty)

    def test_non_empty_pcd_does_not_raise_on_validation(self):
        """Validation passes for a non-empty pcd (visualization is mocked)."""
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing"):
            open_point_cloud_editor(pcd, show_help=False)  # should not raise

    def test_list_of_pcds_accepted(self):
        """A list input bypasses the empty-check and is forwarded to Open3D."""
        pcds = [_make_pcd(3), _make_pcd(3)]
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing") as mock_draw:
            open_point_cloud_editor(pcds, show_help=False)
            mock_draw.assert_called_once()

    def test_returns_none(self):
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing"):
            result = open_point_cloud_editor(pcd, show_help=False)
        assert result is None

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_draw_geometries_with_editing_called_once(self):
        """The visualization function is called exactly once."""
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing") as mock_draw:
            open_point_cloud_editor(pcd, show_help=False)
            mock_draw.assert_called_once()

    def test_draw_called_with_list_wrapping_pcd(self):
        """For a single pcd, it is wrapped in a list before passing to draw."""
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing") as mock_draw:
            open_point_cloud_editor(pcd, show_help=False)
            call_args = mock_draw.call_args[0][0]
            assert isinstance(call_args, list)
            assert pcd in call_args

    def test_show_help_true_prints_controls(self, capsys):
        """With show_help=True, control instructions are printed."""
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing"):
            open_point_cloud_editor(pcd, show_help=True)
        out = capsys.readouterr().out
        assert "Press" in out

    def test_show_help_false_suppresses_output(self, capsys):
        """With show_help=False, nothing is printed."""
        pcd = _make_pcd(5)
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing"):
            open_point_cloud_editor(pcd, show_help=False)
        assert capsys.readouterr().out == ""

    def test_draw_called_with_list_directly_when_list_input(self):
        """A list input is passed directly to draw (not re-wrapped)."""
        pcds = [_make_pcd(3), _make_pcd(3)]
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries_with_editing") as mock_draw:
            open_point_cloud_editor(pcds, show_help=False)
            call_args = mock_draw.call_args[0][0]
            assert call_args is pcds


# ── open_mesh_and_lineset_viewer ───────────────────────────────────────────────

class TestOpenMeshAndLinesetViewer:
    # ── Pure logic ───────────────────────────────────────────────────────────

    def test_empty_mesh_raises_value_error(self):
        empty_mesh = o3d.geometry.TriangleMesh()
        with pytest.raises(ValueError, match="empty"):
            open_mesh_and_lineset_viewer(empty_mesh)

    def test_empty_lineset_raises_value_error(self):
        empty_ls = o3d.geometry.LineSet()
        with pytest.raises(ValueError, match="empty"):
            open_mesh_and_lineset_viewer(empty_ls)

    def test_non_empty_mesh_does_not_raise(self):
        mesh = _make_mesh()
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries"):
            open_mesh_and_lineset_viewer(mesh)  # should not raise

    def test_non_empty_lineset_does_not_raise(self):
        ls = _make_lineset()
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries"):
            open_mesh_and_lineset_viewer(ls)  # should not raise

    def test_returns_none(self):
        mesh = _make_mesh()
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries"):
            result = open_mesh_and_lineset_viewer(mesh)
        assert result is None

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_draw_geometries_called_once_for_mesh(self):
        mesh = _make_mesh()
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries") as mock_draw:
            open_mesh_and_lineset_viewer(mesh)
            mock_draw.assert_called_once()

    def test_draw_geometries_called_once_for_lineset(self):
        ls = _make_lineset()
        with patch("Source.pointCloudEditor.o3d.visualization.draw_geometries") as mock_draw:
            open_mesh_and_lineset_viewer(ls)
            mock_draw.assert_called_once()
