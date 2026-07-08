"""Tests for Source/roofTools.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import open3d as o3d
from Source.roofTools import slice_roof_up, keep_highest_point_above_corner, smooth_roof


def _make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def _flat_roof_pcd(n: int = 40) -> o3d.geometry.PointCloud:
    """Simple flat roof: points on a grid at different Z levels."""
    pts = []
    for z in np.linspace(0.0, 3.0, 4):
        for x in np.linspace(0, 5, 10):
            pts.append([x, 0.0, z])
    return _make_pcd(np.array(pts[:n]))


# ── slice_roof_up ──────────────────────────────────────────────────────────────

class TestSliceRoofUp:
    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            slice_roof_up("not a pcd")

    def test_zero_slices_raises_value_error(self):
        pcd = _flat_roof_pcd()
        with pytest.raises(ValueError, match="at least 1"):
            slice_roof_up(pcd, slices_amount=0)

    def test_negative_slices_raises_value_error(self):
        pcd = _flat_roof_pcd()
        with pytest.raises(ValueError):
            slice_roof_up(pcd, slices_amount=-1)

    def test_returns_list(self):
        pcd = _flat_roof_pcd()
        result = slice_roof_up(pcd, slices_amount=2, slab_fatness=2.0, visualize=False)
        assert isinstance(result, list)

    def test_correct_number_of_slices_returned(self):
        """With generous slab_fatness every slice should be non-empty."""
        pts = np.array([[float(i), 0.0, float(z)]
                        for z in range(4) for i in range(5)])
        pcd = _make_pcd(pts)
        result = slice_roof_up(pcd, slices_amount=4, slab_fatness=0.6, visualize=False)
        assert len(result) == 4

    def test_each_element_is_ndarray(self):
        pcd = _flat_roof_pcd()
        result = slice_roof_up(pcd, slices_amount=2, slab_fatness=2.0, visualize=False)
        for item in result:
            assert isinstance(item, np.ndarray)

    def test_flattened_z_matches_slice_center(self):
        """All points in each slice share the same Z value (the slice center)."""
        pts = np.array([[float(i), 0.0, float(z)]
                        for z in [0.0, 1.0, 2.0] for i in range(5)])
        pcd = _make_pcd(pts)
        result = slice_roof_up(pcd, slices_amount=3, slab_fatness=0.4, visualize=False)
        for arr in result:
            z_vals = arr[:, 2]
            assert np.allclose(z_vals, z_vals[0]), "All Z values in a slice must be equal"

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_grid_subsampling_called_per_slice(self):
        """grid_subsampling is called once per non-empty slice."""
        pcd = _flat_roof_pcd(40)
        mock_result = _make_pcd(np.random.rand(5, 3))
        mock_result.points = o3d.utility.Vector3dVector(np.random.rand(5, 3))

        with patch("Source.roofTools.grid_subsampling",
                   return_value=mock_result) as mock_gs, \
             patch("Source.roofTools.find_corners",
                   return_value=np.random.rand(4, 3)):
            slice_roof_up(pcd, slices_amount=3, slab_fatness=2.0,
                          visualize=False)
            assert mock_gs.call_count == 3

    def test_find_corners_called_per_slice(self):
        """find_corners is called once per non-empty slice after subsampling."""
        pts = np.array([[float(i), 0.0, float(z)]
                        for z in [0.0, 1.0] for i in range(5)])
        pcd = _make_pcd(pts)
        mock_subsampled = _make_pcd(np.random.rand(5, 3))

        with patch("Source.roofTools.grid_subsampling",
                   return_value=mock_subsampled), \
             patch("Source.roofTools.find_corners",
                   return_value=np.random.rand(3, 3)) as mock_fc:
            slice_roof_up(pcd, slices_amount=2, slab_fatness=0.6, visualize=False)
            assert mock_fc.call_count == 2

    def test_visualize_false_does_not_call_editor(self):
        """When visualize=False, open_point_cloud_editor is never called."""
        pcd = _flat_roof_pcd()
        with patch("Source.roofTools.opce") as mock_opce, \
             patch("Source.roofTools.grid_subsampling",
                   return_value=_make_pcd(np.random.rand(5, 3))), \
             patch("Source.roofTools.find_corners",
                   return_value=np.random.rand(4, 3)):
            slice_roof_up(pcd, slices_amount=2, slab_fatness=2.0, visualize=False)
            mock_opce.assert_not_called()

    def test_tqdm_wraps_slice_iteration(self):
        """tqdm is used to display progress over slices."""
        pcd = _flat_roof_pcd()
        with patch("Source.roofTools.tqdm",
                   wraps=__import__("tqdm").tqdm) as mock_tqdm, \
             patch("Source.roofTools.grid_subsampling",
                   return_value=_make_pcd(np.random.rand(5, 3))), \
             patch("Source.roofTools.find_corners",
                   return_value=np.random.rand(4, 3)):
            slice_roof_up(pcd, slices_amount=2, slab_fatness=2.0, visualize=False)
            mock_tqdm.assert_called_once()


# ── keep_highest_point_above_corner ───────────────────────────────────────────

class TestKeepHighestPointAboveCorner:
    def _corner_pcd(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        return _make_pcd(pts)

    def _full_pcd(self):
        pts = np.array([
            [0.0, 0.0, 1.0], [0.0, 0.0, 2.0],  # above corner (0,0)
            [1.0, 1.0, 1.5], [1.0, 1.0, 0.5],   # above corner (1,1)
        ])
        return _make_pcd(pts)

    def test_returns_point_cloud(self):
        result = keep_highest_point_above_corner(
            self._corner_pcd(), self._full_pcd(), search_radius=0.05
        )
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_one_result_per_corner(self):
        """One highest point is selected per corner."""
        result = keep_highest_point_above_corner(
            self._corner_pcd(), self._full_pcd(), search_radius=0.05
        )
        assert len(result.points) == 2

    def test_selects_highest_z_above_each_corner(self):
        """The highest Z point above each corner is kept."""
        result = keep_highest_point_above_corner(
            self._corner_pcd(), self._full_pcd(), search_radius=0.05
        )
        z_vals = sorted(np.asarray(result.points)[:, 2].tolist())
        assert z_vals == pytest.approx(sorted([2.0, 1.5]))

    # ── Mock tests ──────────────────────────────────────────────────────────────

    def test_np_asarray_called_for_both_pcds(self):
        """np.asarray is called to extract points from both input pcds."""
        corners = self._corner_pcd()
        full = self._full_pcd()
        with patch("numpy.asarray", wraps=np.asarray) as mock_asarray:
            keep_highest_point_above_corner(corners, full, search_radius=0.05)
            assert mock_asarray.call_count >= 2

    def test_o3d_point_cloud_created_for_result(self):
        """The result is a new o3d.geometry.PointCloud instance."""
        corners = self._corner_pcd()
        full = self._full_pcd()
        result = keep_highest_point_above_corner(corners, full, search_radius=0.05)
        assert isinstance(result, o3d.geometry.PointCloud)


# ── smooth_roof ───────────────────────────────────────────────────────────────

class TestSmoothRoof:
    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            smooth_roof("not a pcd")

    def test_empty_cloud_raises_value_error(self):
        with pytest.raises(ValueError, match="must contain points"):
            smooth_roof(_make_pcd(np.empty((0, 3))))

    def test_non_positive_voxel_size_raises_value_error(self):
        pcd = _make_pcd(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
        with pytest.raises(ValueError, match="positive"):
            smooth_roof(pcd, voxel_size=0.0)

    def test_invalid_upsample_factor_raises_value_error(self):
        pcd = _make_pcd(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
        with pytest.raises(ValueError, match="upsample_factor"):
            smooth_roof(pcd, voxel_size=1.0, upsample_factor=0)

    def test_reduces_spike_error_against_linear_trend(self):
        x = np.arange(0, 21, dtype=float)
        trend = 0.2 * x + 5.0
        z = trend.copy()

        # Inject two strong spikes that should be suppressed.
        z[8] += 4.0
        z[14] -= 3.0

        pts = np.column_stack([x, np.zeros_like(x), z])
        pcd = _make_pcd(pts)

        with patch("Source.roofTools.grid_subsampling", side_effect=lambda pc, **kwargs: pc):
            smoothed = smooth_roof(pcd, voxel_size=1.0, visualize=False)

        smoothed_z = np.asarray(smoothed.points)[:, 2]
        original_mae = np.mean(np.abs(z - trend))
        smoothed_mae = np.mean(np.abs(smoothed_z - trend))

        assert smoothed_mae < original_mae

    def test_visualize_false_does_not_call_editor(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [2.0, 0.0, 0.4]])
        pcd = _make_pcd(pts)

        with patch("Source.roofTools.grid_subsampling", side_effect=lambda pc, **kwargs: pc), \
             patch("Source.roofTools.opce") as mock_opce:
            smooth_roof(pcd, voxel_size=1.0, visualize=False)

        mock_opce.assert_not_called()

    def test_upsample_factor_increases_point_count(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 0.4],
            [3.0, 0.0, 0.6],
        ])
        pcd = _make_pcd(pts)

        with patch("Source.roofTools.grid_subsampling", side_effect=lambda pc, **kwargs: pc):
            smoothed = smooth_roof(pcd, voxel_size=1.0, upsample_factor=3, visualize=False)

        assert len(smoothed.points) == len(pts) * 3
