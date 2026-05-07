"""Tests for Source/wallTools.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.wallTools import (
    extract_wall_points,
    define_min_height_roof,
    connect_vertically_aligned_points,
    connect_vertically_aligned_points2,
    divide_wall_into_layers,
)


def _make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _make_colored_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = _make_pcd(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(points), 1)))
    return pcd


# ── extract_wall_points ────────────────────────────────────────────────────────

class TestExtractWallPoints:
    def _floor_pcd(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        return _make_pcd(pts)

    def _tbp_pcd(self):
        # Points at same XY as floor corners but at varying heights
        pts = np.array([
            [0.0, 0.0, 0.5], [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5], [1.0, 0.0, 1.5],
        ])
        return _make_pcd(pts)

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_wall_points("not a pcd", self._floor_pcd())

    def test_empty_tbp_raises_value_error(self):
        empty = o3d.geometry.PointCloud()
        with pytest.raises(ValueError):
            extract_wall_points(empty, self._floor_pcd())

    def test_empty_floor_raises_value_error(self):
        empty = o3d.geometry.PointCloud()
        with pytest.raises(ValueError):
            extract_wall_points(self._tbp_pcd(), empty)

    def test_negative_search_radius_raises(self):
        with pytest.raises(ValueError):
            extract_wall_points(self._tbp_pcd(), self._floor_pcd(), search_radius=-0.1)

    def test_returns_point_cloud(self):
        result = extract_wall_points(self._tbp_pcd(), self._floor_pcd(),
                                     search_radius=0.1)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_result_has_points(self):
        result = extract_wall_points(self._tbp_pcd(), self._floor_pcd(),
                                     search_radius=0.1)
        assert len(result.points) > 0

    def test_no_matching_points_raises(self):
        """If no tbp points are above the contour within radius, ValueError is raised."""
        tbp = _make_pcd(np.array([[99.0, 99.0, 0.5]]))
        floor = _make_pcd(np.array([[0.0, 0.0, 1.0]]))  # floor z=1 > tbp z=0.5
        with pytest.raises(ValueError, match="No TBP points"):
            extract_wall_points(tbp, floor, search_radius=0.01)

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_ckdtree_built_on_tbp_xy(self):
        """cKDTree is constructed using the XY coordinates of the tbp point cloud."""
        tbp = self._tbp_pcd()
        floor = self._floor_pcd()
        with patch("Source.wallTools.cKDTree",
                   wraps=__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree
                   ) as mock_tree:
            extract_wall_points(tbp, floor, search_radius=0.1)
            mock_tree.assert_called_once()

    def test_create_point_cloud_called_for_result(self):
        """create_point_cloud is called to build the returned PointCloud."""
        tbp = self._tbp_pcd()
        floor = self._floor_pcd()
        with patch("Source.wallTools.create_point_cloud",
                   wraps=__import__("Source.heightMapModule",
                                    fromlist=["create_point_cloud"]).create_point_cloud
                   ) as mock_cp:
            extract_wall_points(tbp, floor, search_radius=0.1)
            mock_cp.assert_called_once()


# ── define_min_height_roof ─────────────────────────────────────────────────────

class TestDefineMinHeightRoof:
    def _wall_pcd(self):
        pts = np.array([
            [0.0, 0.0, 0.2], [0.0, 0.0, 0.8],
            [0.0, 0.0, 1.5], [0.0, 0.0, 2.5],
        ])
        return _make_pcd(pts)

    def _floor_pcd(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        return _make_pcd(pts)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            define_min_height_roof("not a pcd", self._floor_pcd())

    def test_empty_wall_raises(self):
        with pytest.raises(ValueError):
            define_min_height_roof(o3d.geometry.PointCloud(), self._floor_pcd())

    def test_all_below_threshold_raises(self):
        """ValueError if all wall points are below floor + height."""
        wall = _make_pcd(np.array([[0.0, 0.0, 0.1]]))
        floor = _make_pcd(np.array([[0.0, 0.0, 0.0]]))
        with pytest.raises(ValueError, match="No wall points"):
            define_min_height_roof(wall, floor, height=5.0)

    def test_returns_two_point_clouds(self):
        result = define_min_height_roof(self._wall_pcd(), self._floor_pcd(), height=1.0)
        assert len(result) == 2
        assert all(isinstance(r, o3d.geometry.PointCloud) for r in result)

    def test_kept_points_are_above_threshold(self):
        """The first returned pcd contains only points above floor_height + height."""
        wall = _make_pcd(np.array([
            [0.0, 0.0, 0.5],   # below
            [0.0, 0.0, 2.0],   # above
            [0.0, 0.0, 3.0],   # above
        ]))
        floor = _make_pcd(np.array([[0.0, 0.0, 0.0]]))
        kept, removed = define_min_height_roof(wall, floor, height=1.0)
        kept_z = np.asarray(kept.points)[:, 2]
        assert np.all(kept_z > 1.0)

    def test_removed_points_are_below_threshold(self):
        wall = _make_pcd(np.array([
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 2.0],
        ]))
        floor = _make_pcd(np.array([[0.0, 0.0, 0.0]]))
        kept, removed = define_min_height_roof(wall, floor, height=1.0)
        removed_z = np.asarray(removed.points)[:, 2]
        assert np.all(removed_z <= 1.0)

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_np_min_called_to_find_floor_height(self):
        """np.min is used to find the minimum Z of the floor point cloud."""
        wall = self._wall_pcd()
        floor = self._floor_pcd()
        with patch("numpy.min", wraps=np.min) as mock_min:
            define_min_height_roof(wall, floor, height=1.0)
            mock_min.assert_called()

    def test_blue_color_applied_to_kept_points(self):
        """Kept wall points are colored blue ([0, 0, 1])."""
        wall = _make_pcd(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]))
        floor = _make_pcd(np.array([[0.0, 0.0, 0.0]]))
        kept, _ = define_min_height_roof(wall, floor, height=1.0)
        assert kept.has_colors()
        colors = np.asarray(kept.colors)
        assert np.allclose(colors[0], [0, 0, 1])


# ── connect_vertically_aligned_points ─────────────────────────────────────────

class TestConnectVerticallyAlignedPoints:
    def test_returns_lineset(self):
        floor = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        wall = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0]])
        result = connect_vertically_aligned_points(floor, wall, xy_tol=0.05)
        assert isinstance(result, o3d.geometry.LineSet)

    def test_matching_pairs_create_lines(self):
        floor = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        wall = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0]])
        ls = connect_vertically_aligned_points(floor, wall, xy_tol=0.05)
        assert len(ls.lines) == 2

    def test_non_matching_pairs_create_no_lines(self):
        floor = np.array([[0.0, 0.0, 0.0]])
        wall = np.array([[99.0, 99.0, 3.0]])  # far away in XY
        ls = connect_vertically_aligned_points(floor, wall, xy_tol=0.01)
        assert len(ls.lines) == 0

    def test_all_points_in_result(self):
        """All floor and wall points appear in the returned LineSet."""
        floor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        wall = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        ls = connect_vertically_aligned_points(floor, wall, xy_tol=0.05)
        assert len(ls.points) == len(floor) + len(wall)

    def test_tight_tolerance_rejects_slightly_offset(self):
        floor = np.array([[0.0, 0.0, 0.0]])
        wall = np.array([[0.05, 0.05, 3.0]])  # outside tol=0.01
        ls = connect_vertically_aligned_points(floor, wall, xy_tol=0.01)
        assert len(ls.lines) == 0

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_ckdtree_built_on_wall_xy(self):
        """cKDTree is constructed on the XY slice of wall_points."""
        floor = np.array([[0.0, 0.0, 0.0]])
        wall = np.array([[0.0, 0.0, 3.0]])
        with patch("Source.wallTools.cKDTree",
                   wraps=__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree
                   ) as mock_tree:
            connect_vertically_aligned_points(floor, wall, xy_tol=0.1)
            mock_tree.assert_called_once()

    def test_lineset_constructed_once(self):
        """Exactly one o3d.geometry.LineSet is returned."""
        floor = np.array([[0.0, 0.0, 0.0]])
        wall = np.array([[0.0, 0.0, 3.0]])
        with patch("Source.wallTools.o3d.geometry.LineSet",
                   wraps=o3d.geometry.LineSet) as mock_ls:
            connect_vertically_aligned_points(floor, wall, xy_tol=0.1)
            mock_ls.assert_called_once()


# ── connect_vertically_aligned_points2 ──────────────────────────────────────────────

class TestConnectVerticallyAlignedPoints2:
    def test_returns_lineset(self):
        base = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        upper = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0]])
        result = connect_vertically_aligned_points2(base, upper, xy_tol=0.05)
        assert isinstance(result, o3d.geometry.LineSet)

    def test_single_upper_array_creates_lines(self):
        base = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        upper = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 3.0]])
        ls = connect_vertically_aligned_points2(base, upper, xy_tol=0.05)
        assert len(ls.lines) == 2

    def test_list_of_upper_levels(self):
        """Accepts a list of upper-level arrays and connects each base point to the first match."""
        base = np.array([[0.0, 0.0, 0.0]])
        upper1 = np.array([[0.0, 0.0, 1.0]])
        upper2 = np.array([[0.0, 0.0, 2.0]])
        ls = connect_vertically_aligned_points2(base, [upper1, upper2], xy_tol=0.05)
        # Should connect to upper1 (first match) only
        assert len(ls.lines) == 1

    def test_non_matching_creates_no_lines(self):
        base = np.array([[0.0, 0.0, 0.0]])
        upper = np.array([[99.0, 99.0, 3.0]])
        ls = connect_vertically_aligned_points2(base, upper, xy_tol=0.01)
        assert len(ls.lines) == 0

    def test_all_points_in_result(self):
        """All base and upper points appear in the returned LineSet."""
        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        upper = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        ls = connect_vertically_aligned_points2(base, upper, xy_tol=0.05)
        assert len(ls.points) == len(base) + len(upper)

    def test_each_base_point_matched_at_most_once(self):
        """Each base point stops at the first matching level."""
        base = np.array([[0.0, 0.0, 0.0]])
        upper1 = np.array([[0.0, 0.0, 1.0]])
        upper2 = np.array([[0.0, 0.0, 2.0]])
        ls = connect_vertically_aligned_points2(base, [upper1, upper2], xy_tol=0.1)
        assert len(ls.lines) == 1  # stops at upper1, not both

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_ckdtree_called_per_level(self):
        """cKDTree is built once per upper level that is actually checked."""
        base = np.array([[0.0, 0.0, 0.0]])
        upper1 = np.array([[99.0, 99.0, 1.0]])  # far away → no match, moves to upper2
        upper2 = np.array([[0.0, 0.0, 2.0]])    # matches
        with patch("Source.wallTools.cKDTree",
                   wraps=__import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree
                   ) as mock_tree:
            connect_vertically_aligned_points2(base, [upper1, upper2], xy_tol=0.1)
            assert mock_tree.call_count == 2  # one per level

    def test_lineset_constructed_once(self):
        base = np.array([[0.0, 0.0, 0.0]])
        upper = np.array([[0.0, 0.0, 3.0]])
        with patch("Source.wallTools.o3d.geometry.LineSet",
                   wraps=o3d.geometry.LineSet) as mock_ls:
            connect_vertically_aligned_points2(base, upper, xy_tol=0.1)
            mock_ls.assert_called_once()


# ── divide_wall_into_layers ────────────────────────────────────────────────────────

class TestDivideWallIntoLayers:
    def _make_wall_pcd(self):
        pts = np.array([[0.0, 0.0, float(z)] for z in range(20)], dtype=float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            divide_wall_into_layers("not a pcd")

    def test_empty_pcd_raises_value_error(self):
        with pytest.raises(ValueError):
            divide_wall_into_layers(o3d.geometry.PointCloud())

    def test_non_positive_layer_amount_raises(self):
        with pytest.raises(ValueError):
            divide_wall_into_layers(self._make_wall_pcd(), layer_amount=0)

    def test_returns_list(self):
        result = divide_wall_into_layers(self._make_wall_pcd(), layer_amount=5)
        assert isinstance(result, list)

    def test_each_element_is_ndarray(self):
        result = divide_wall_into_layers(self._make_wall_pcd(), layer_amount=5)
        for item in result:
            assert isinstance(item, np.ndarray)

    def test_layer_count_at_most_layer_amount(self):
        """Cannot produce more layers than requested (empty layers are skipped)."""
        result = divide_wall_into_layers(self._make_wall_pcd(), layer_amount=5)
        assert len(result) <= 5

    def test_single_layer_contains_all_points(self):
        """With a flat wall (all points at same z), layer_amount=1 produces 1 layer."""
        # All 5 points at z=0 -> single slice at z=0, all 5 pass the +-0.01 mask
        flat_pts = np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(flat_pts)
        result = divide_wall_into_layers(pcd, layer_amount=1)
        assert len(result) == 1

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_tqdm_called_for_slice_iteration(self):
        """tqdm wraps the slice-height iteration."""
        with patch("Source.wallTools.tqdm", side_effect=lambda it, **kw: it) as mock_tqdm:
            divide_wall_into_layers(self._make_wall_pcd(), layer_amount=3)
            mock_tqdm.assert_called_once()
