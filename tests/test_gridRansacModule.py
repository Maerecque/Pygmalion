"""Tests for Source/gridRansacModule.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.gridRansacModule import divide_pointcloud_into_grid, get_points_from_grid, ransac_plane_finder, walk_through_grid


def _make_pcd(n: int = 27) -> o3d.geometry.PointCloud:
    """Regular 3×3×3 grid of points with grey colors."""
    pts = np.array([[float(x), float(y), float(z)]
                    for x in range(3) for y in range(3) for z in range(3)])[:n]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (n, 1)))
    return pcd


# ── divide_pointcloud_into_grid ────────────────────────────────────────────────

class TestDividePointcloudIntoGrid:
    def test_returns_dict(self):
        pcd = _make_pcd()
        result = divide_pointcloud_into_grid(pcd, grid_size=1.5, overlap=0)
        assert isinstance(result, dict)

    def test_returns_dict_and_num_cells_when_requested(self):
        pcd = _make_pcd()
        result, num_cells = divide_pointcloud_into_grid(
            pcd, grid_size=1.5, overlap=0, give_grid_size=True
        )
        assert isinstance(result, dict)
        assert isinstance(num_cells, np.ndarray)
        assert len(num_cells) == 3

    def test_all_points_assigned_to_at_least_one_cell(self):
        pcd = _make_pcd(27)
        grid = divide_pointcloud_into_grid(pcd, grid_size=3.0, overlap=0)
        all_indices = {idx for indices in grid.values() for idx in indices}
        assert all_indices == set(range(len(pcd.points)))

    def test_overlap_increases_cell_counts(self):
        """With overlap=1 every point appears in more cells than with overlap=0."""
        pcd = _make_pcd()
        grid_no_overlap = divide_pointcloud_into_grid(pcd, grid_size=1.5, overlap=0)
        grid_with_overlap = divide_pointcloud_into_grid(pcd, grid_size=1.5, overlap=1)
        total_no = sum(len(v) for v in grid_no_overlap.values())
        total_with = sum(len(v) for v in grid_with_overlap.values())
        assert total_with >= total_no

    def test_single_cell_with_large_grid_size(self):
        """If grid_size > extent, all points land in a single cell (no overlap)."""
        pcd = _make_pcd(27)
        grid = divide_pointcloud_into_grid(pcd, grid_size=100.0, overlap=0)
        assert len(grid) == 1

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_tqdm_wraps_point_iteration(self):
        """tqdm is used to report progress over individual points."""
        pcd = _make_pcd(8)
        with patch("Source.gridRansacModule.tqdm",
                   wraps=__import__("tqdm").tqdm) as mock_tqdm:
            divide_pointcloud_into_grid(pcd, grid_size=2.0, overlap=0)
            mock_tqdm.assert_called_once()

    def test_get_min_bound_called(self):
        """get_min_bound() is called on the input point cloud."""
        pts = np.array([[float(x), float(y), float(z)]
                        for x in range(3) for y in range(3) for z in range(3)])
        mock_pcd = MagicMock()
        mock_pcd.points = pts
        mock_pcd.get_min_bound.return_value = np.array([0.0, 0.0, 0.0])
        mock_pcd.get_max_bound.return_value = np.array([2.0, 2.0, 2.0])
        divide_pointcloud_into_grid(mock_pcd, grid_size=1.5, overlap=0)
        mock_pcd.get_min_bound.assert_called_once()

    def test_get_max_bound_called(self):
        """get_max_bound() is called on the input point cloud."""
        pts = np.array([[float(x), float(y), float(z)]
                        for x in range(3) for y in range(3) for z in range(3)])
        mock_pcd = MagicMock()
        mock_pcd.points = pts
        mock_pcd.get_min_bound.return_value = np.array([0.0, 0.0, 0.0])
        mock_pcd.get_max_bound.return_value = np.array([2.0, 2.0, 2.0])
        divide_pointcloud_into_grid(mock_pcd, grid_size=1.5, overlap=0)
        mock_pcd.get_max_bound.assert_called_once()

    def test_print_called_with_grid_summary(self, capsys):
        """Grid dimensions and cell count are printed after division."""
        pcd = _make_pcd()
        divide_pointcloud_into_grid(pcd, grid_size=1.5, overlap=0)
        out = capsys.readouterr().out
        assert "grid" in out.lower()


# ── get_points_from_grid ───────────────────────────────────────────────────────

class TestGetPointsFromGrid:
    def _setup(self):
        pcd = _make_pcd(27)
        grid = divide_pointcloud_into_grid(pcd, grid_size=3.0, overlap=0)
        cell_key = next(iter(grid))         # first available cell key
        cell_index = np.array(             # parse "[x y z]" string back to array
            list(map(int, cell_key.strip("[]").split()))
        )
        return pcd, grid, cell_index

    def test_returns_point_cloud(self):
        pcd, grid, cell_index = self._setup()
        result = get_points_from_grid(pcd, grid, cell_index)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_result_contains_correct_points(self):
        """Points in the extracted cell match those indexed from the full cloud."""
        pcd, grid, cell_index = self._setup()
        result = get_points_from_grid(pcd, grid, cell_index)
        key = str(cell_index)
        expected_pts = np.asarray(pcd.points)[grid[key]]
        result_pts = np.asarray(result.points)
        assert result_pts.shape == expected_pts.shape
        np.testing.assert_array_almost_equal(
            np.sort(result_pts, axis=0), np.sort(expected_pts, axis=0)
        )

    def test_unknown_key_raises_key_error(self):
        pcd = _make_pcd()
        grid = divide_pointcloud_into_grid(pcd, grid_size=1.5, overlap=0)
        bad_index = np.array([999, 999, 999])
        with pytest.raises(KeyError):
            get_points_from_grid(pcd, grid, bad_index)

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_np_asarray_called_to_extract_points(self):
        """np.asarray is used to read the point cloud data."""
        pcd, grid, cell_index = self._setup()
        with patch("numpy.asarray", wraps=np.asarray) as mock_asarray:
            get_points_from_grid(pcd, grid, cell_index)
            mock_asarray.assert_called()

    def test_o3d_point_cloud_constructed_for_result(self):
        """o3d.geometry.PointCloud is instantiated to hold extracted points."""
        pcd, grid, cell_index = self._setup()
        with patch("Source.gridRansacModule.o3d.geometry.PointCloud",
                   wraps=o3d.geometry.PointCloud) as mock_pcd:
            get_points_from_grid(pcd, grid, cell_index)
            mock_pcd.assert_called_once()


# ── ransac_plane_finder ────────────────────────────────────────────────────────

class TestRansacPlaneFinder:
    def _make_plane_pcd(self, n=600):
        """Flat point cloud lying in the XY plane (z=0) — easy for RANSAC."""
        np.random.seed(42)
        pts = np.column_stack([
            np.random.rand(n),
            np.random.rand(n),
            np.zeros(n)
        ])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def test_returns_point_cloud_for_valid_input(self):
        pcd = self._make_plane_pcd()
        result = ransac_plane_finder(pcd, min_plane_points=10)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_returns_none_when_too_few_points(self):
        """A pcd with < 3 points returns None (loop never segments)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
        result = ransac_plane_finder(pcd)
        assert result is None

    def test_returns_none_when_no_large_plane(self):
        """With min_plane_points much larger than pcd size, no plane qualifies."""
        pcd = self._make_plane_pcd(n=50)
        result = ransac_plane_finder(pcd, min_plane_points=10000)
        assert result is None

    def test_print_found_planes_outputs_text(self, capsys):
        pcd = self._make_plane_pcd()
        ransac_plane_finder(pcd, min_plane_points=10, print_found_planes=True)
        captured = capsys.readouterr()
        assert "plane" in captured.out.lower()

    def test_segment_plane_called_via_shapeutils(self):
        """su.segment_plane is called at least once."""
        pcd = self._make_plane_pcd()
        with patch("Source.gridRansacModule.su.segment_plane",
                   wraps=__import__("Source.shapeUtils", fromlist=["segment_plane"]).segment_plane
                   ) as mock_seg:
            ransac_plane_finder(pcd, min_plane_points=10)
            mock_seg.assert_called()


# ── walk_through_grid ──────────────────────────────────────────────────────────

class TestWalkThroughGrid:
    def _make_pcd_and_grid(self, n_per_cell=600):
        """Two dense cells placed 10 units apart so they land in separate grid cells."""
        np.random.seed(7)
        pts_a = np.column_stack([np.random.rand(n_per_cell),
                                 np.random.rand(n_per_cell),
                                 np.random.rand(n_per_cell)])  # non-zero z so grid has z-cells
        pts_b = pts_a.copy()
        pts_b[:, 0] += 10.0

        all_pts = np.vstack([pts_a, pts_b])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(all_pts), 3)))
        from Source.gridRansacModule import divide_pointcloud_into_grid
        grid = divide_pointcloud_into_grid(pcd, grid_size=2.0, overlap=0)
        return pcd, grid

    def test_returns_point_cloud(self):
        pcd, grid = self._make_pcd_and_grid()
        mock_plane = o3d.geometry.PointCloud()
        mock_plane.points = o3d.utility.Vector3dVector(np.zeros((3, 3)))
        empty_pcd = o3d.geometry.PointCloud()
        with patch("Source.gridRansacModule.ransac_plane_finder", return_value=mock_plane), \
             patch("Source.gridRansacModule.su.merge_list_of_pointclouds", return_value=empty_pcd):
            result = walk_through_grid(pcd, grid, min_cell_size=10, min_plane_points=10)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_small_cells_skipped(self):
        """Cells below min_cell_size are not passed to ransac_plane_finder."""
        pcd, grid = self._make_pcd_and_grid()
        empty_pcd = o3d.geometry.PointCloud()
        with patch("Source.gridRansacModule.ransac_plane_finder") as mock_rf, \
             patch("Source.gridRansacModule.su.merge_list_of_pointclouds", return_value=empty_pcd):
            walk_through_grid(pcd, grid, min_cell_size=10**9, min_plane_points=10)
            mock_rf.assert_not_called()

    def test_ransac_called_for_large_cells(self):
        """ransac_plane_finder is invoked for cells meeting the min_cell_size threshold."""
        pcd, grid = self._make_pcd_and_grid()
        mock_plane = o3d.geometry.PointCloud()
        mock_plane.points = o3d.utility.Vector3dVector(np.zeros((3, 3)))
        empty_pcd = o3d.geometry.PointCloud()
        with patch("Source.gridRansacModule.ransac_plane_finder",
                   return_value=mock_plane) as mock_rf, \
             patch("Source.gridRansacModule.su.merge_list_of_pointclouds", return_value=empty_pcd):
            walk_through_grid(pcd, grid, min_cell_size=10, min_plane_points=10)
            mock_rf.assert_called()
