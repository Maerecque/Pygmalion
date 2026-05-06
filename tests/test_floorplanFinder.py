"""Tests for Source/floorplanFinder.py"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.floorplanFinder import alpha_shape, sort_points_in_hull, find_corners, find_boundary_from_floor
import open3d as o3d


class TestAlphaShape:
    def test_fewer_than_4_points_returns_convex_hull(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        result = alpha_shape(points, alpha=1.0)
        assert result is not None
        assert hasattr(result, "geom_type")

    def test_square_points_returns_polygon(self):
        # Dense square grid - alpha shape should produce a polygon
        pts = []
        for i in np.linspace(0, 4, 20):
            for j in np.linspace(0, 4, 20):
                pts.append([i, j])
        points = np.array(pts)
        result = alpha_shape(points, alpha=0.5)
        assert result.geom_type in ("Polygon", "MultiPolygon")

    def test_returns_shapely_geometry(self):
        from shapely.geometry.base import BaseGeometry
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
                           [1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
        result = alpha_shape(points, alpha=1.0)
        assert isinstance(result, BaseGeometry)


class TestSortPointsInHull:
    def test_invalid_1d_input_raises_value_error(self):
        with pytest.raises(ValueError):
            sort_points_in_hull(np.array([1.0, 2.0, 3.0]))

    def test_invalid_single_column_raises_value_error(self):
        with pytest.raises(ValueError):
            sort_points_in_hull(np.array([[1.0], [2.0]]))

    def test_fewer_than_2_points_returns_empty(self):
        result = sort_points_in_hull(np.array([[1.0, 2.0, 3.0]]))
        assert len(result) == 0

    def test_nearest_method_returns_all_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        result = sort_points_in_hull(pts, method="nearest")
        # Sorted points should contain all original points (possibly + 1 if loop closed)
        assert len(result) >= len(pts)

    def test_angle_method_returns_correct_count(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        result = sort_points_in_hull(pts, method="angle")
        # angle method closes the loop: N+1 points
        assert len(result) == len(pts) + 1

    def test_2d_input_works(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        result = sort_points_in_hull(pts, method="nearest")
        assert result.ndim == 2


class TestFindCorners:
    def test_right_angle_detected(self):
        # Horizontal then vertical – a clear 90° corner at index 3
        pts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],  # corner here
            [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0],
        ])
        corners = find_corners(pts, angle_threshold_deg=45, window=1, merge_radius=1)
        # Should detect the corner and include first and last point
        assert len(corners) >= 3
        assert any(np.allclose(c, [3.0, 0.0, 0.0]) for c in corners)

    def test_always_includes_first_and_last(self):
        pts = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        corners = find_corners(pts, angle_threshold_deg=45, window=1, merge_radius=1)
        assert np.allclose(corners[0], pts[0])
        assert np.allclose(corners[-1], pts[-1])

    def test_straight_line_has_no_interior_corners(self):
        pts = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        corners = find_corners(pts, angle_threshold_deg=45, window=1, merge_radius=1)
        # Only first and last (no interior corners on a straight line)
        assert len(corners) == 2

    def test_output_is_subset_of_input(self):
        pts = np.random.rand(20, 3)
        corners = find_corners(pts, angle_threshold_deg=30, window=2, merge_radius=2)
        for c in corners:
            assert any(np.allclose(c, p) for p in pts)

    def test_merge_radius_reduces_duplicates(self):
        # Two nearly-adjacent corners that should be merged into one
        pts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [2.0, 0.5, 0.0], [2.0, 1.0, 0.0],   # corner cluster
            [2.0, 2.0, 0.0], [2.0, 3.0, 0.0],
        ])
        corners_large_merge = find_corners(pts, angle_threshold_deg=30, window=1, merge_radius=3)
        corners_no_merge = find_corners(pts, angle_threshold_deg=30, window=1, merge_radius=0)
        assert len(corners_large_merge) <= len(corners_no_merge)


class TestFindBoundaryFromFloor:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_type_error_returns_none(self):
        """Non-PointCloud input returns None (TypeError caught internally)."""
        result = find_boundary_from_floor("not a pcd", alpha=10)
        assert result is None

    def test_empty_pcd_returns_none(self):
        """Empty PointCloud returns None (ValueError caught internally)."""
        pcd = o3d.geometry.PointCloud()
        result = find_boundary_from_floor(pcd, alpha=10)
        assert result is None

    def test_alpha_shape_called_with_correct_alpha(self):
        """alpha_shape is invoked with the alpha value passed to find_boundary_from_floor."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        with patch("Source.floorplanFinder.alpha_shape", wraps=alpha_shape) as mock_alpha:
            find_boundary_from_floor(pcd, alpha=7.5)
            args, kwargs = mock_alpha.call_args
            assert args[1] == 7.5

    def test_error_in_alpha_shape_returns_none(self):
        """If alpha_shape raises, find_boundary_from_floor returns None."""
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0], [0.5, 0.5, 1.0]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        with patch("Source.floorplanFinder.alpha_shape", side_effect=RuntimeError("boom")):
            result = find_boundary_from_floor(pcd, alpha=10)
            assert result is None

    def test_ckdtree_queried_for_boundary(self):
        """cKDTree.query is called to map boundary coords back to input points."""
        pts = np.array([[float(i), float(j), 0.0]
                        for i in range(5) for j in range(5)])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        with patch("Source.floorplanFinder.cKDTree", wraps=__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree) as mock_tree_cls:
            find_boundary_from_floor(pcd, alpha=0.5)
            mock_tree_cls.assert_called_once()
