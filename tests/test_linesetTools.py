"""Tests for Source/linesetTools.py (contour_to_lineset and merge_lineset)"""
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.linesetTools import contour_to_lineset, merge_lineset
import open3d as o3d


def _make_square_lineset(z: float = 0.0) -> o3d.geometry.LineSet:
    pts = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]])
    return contour_to_lineset(pts)


class TestContourToLineset:
    def test_closed_loop_line_count(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        ls = contour_to_lineset(pts)
        assert len(ls.lines) == 4  # one line per point (closed loop)

    def test_triangle_has_3_lines(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        ls = contour_to_lineset(pts)
        assert len(ls.lines) == 3

    def test_all_input_points_preserved(self):
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        ls = contour_to_lineset(pts)
        assert len(ls.points) == 4

    def test_last_line_connects_back_to_first(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        ls = contour_to_lineset(pts)
        lines = np.asarray(ls.lines)
        # The last line should contain index 0 (closed loop)
        assert 0 in lines[-1]

    def test_max_line_length_filters_long_segments(self):
        # Points with a long segment from (0,0,0) to (10,0,0)
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        ls_full = contour_to_lineset(pts, max_line_length=0)
        ls_filtered = contour_to_lineset(pts, max_line_length=2.0)
        assert len(ls_filtered.lines) < len(ls_full.lines)

    def test_returns_lineset_type(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        ls = contour_to_lineset(pts)
        assert isinstance(ls, o3d.geometry.LineSet)


class TestMergeLineset:
    def test_total_points_is_sum(self):
        ls1 = _make_square_lineset(z=0.0)
        ls2 = _make_square_lineset(z=3.0)
        merged = merge_lineset(ls1, ls2)
        assert len(merged.points) == len(ls1.points) + len(ls2.points)

    def test_total_lines_is_sum(self):
        ls1 = _make_square_lineset(z=0.0)
        ls2 = _make_square_lineset(z=3.0)
        merged = merge_lineset(ls1, ls2)
        assert len(merged.lines) == len(ls1.lines) + len(ls2.lines)

    def test_line_indices_are_offset_correctly(self):
        ls1 = _make_square_lineset(z=0.0)   # indices 0-3
        ls2 = _make_square_lineset(z=3.0)   # indices should become 4-7
        merged = merge_lineset(ls1, ls2)
        lines = np.asarray(merged.lines)
        n1 = len(ls1.points)
        # Lines from ls2 must reference indices >= n1
        ls2_lines = lines[len(ls1.lines):]
        assert np.all(ls2_lines >= n1)

    def test_merging_three_linesets(self):
        ls1 = _make_square_lineset(0.0)
        ls2 = _make_square_lineset(1.0)
        ls3 = _make_square_lineset(2.0)
        merged = merge_lineset(ls1, ls2, ls3)
        assert len(merged.points) == 12
        assert len(merged.lines) == 12

    def test_returns_lineset_type(self):
        ls1 = _make_square_lineset(0.0)
        ls2 = _make_square_lineset(1.0)
        merged = merge_lineset(ls1, ls2)
        assert isinstance(merged, o3d.geometry.LineSet)


class TestContourToLinesetMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_lineset_constructor_called(self):
        """o3d.geometry.LineSet() is instantiated when building the contour."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        with patch("Source.linesetTools.o3d.geometry.LineSet",
                   wraps=o3d.geometry.LineSet) as mock_ls:
            contour_to_lineset(pts)
            mock_ls.assert_called_once()

    def test_numpy_linalg_norm_not_called_when_no_filter(self):
        """np.linalg.norm is NOT called when max_line_length is 0 (no filtering)."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        with patch("numpy.linalg.norm") as mock_norm:
            contour_to_lineset(pts, max_line_length=0)
            mock_norm.assert_not_called()

    def test_numpy_diff_called_for_length_filter(self):
        """np.diff is called when max_line_length > 0 to compute segment lengths."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        with patch("numpy.diff", wraps=np.diff) as mock_diff:
            contour_to_lineset(pts, max_line_length=5.0)
            mock_diff.assert_called_once()


class TestMergeLinesetMocks:
    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_merged_lineset_constructor_called(self):
        """A new o3d.geometry.LineSet is created for the merged result."""
        ls1 = _make_square_lineset(0.0)
        ls2 = _make_square_lineset(1.0)
        with patch("Source.linesetTools.o3d.geometry.LineSet",
                   wraps=o3d.geometry.LineSet) as mock_ls:
            merge_lineset(ls1, ls2)
            mock_ls.assert_called_once()

    def test_np_vstack_called_for_points(self):
        """np.vstack is called to combine all point arrays."""
        ls1 = _make_square_lineset(0.0)
        ls2 = _make_square_lineset(1.0)
        with patch("numpy.vstack", wraps=np.vstack) as mock_vstack:
            merge_lineset(ls1, ls2)
            assert mock_vstack.call_count >= 2  # once for points, once for lines

    def test_colors_not_set_when_linesets_have_no_colors(self):
        """If no input LineSet has colors, the merged result has no colors."""
        ls1 = _make_square_lineset(0.0)
        ls2 = _make_square_lineset(1.0)
        # Ensure no colors on inputs
        ls1.colors = o3d.utility.Vector3dVector([])
        ls2.colors = o3d.utility.Vector3dVector([])
        merged = merge_lineset(ls1, ls2)
        assert not merged.has_colors()
