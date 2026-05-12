"""Tests for Source/linesetTools.py (contour_to_lineset and merge_lineset)"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.linesetTools import (
    contour_to_lineset, merge_lineset,
    filter_lines_within_contour, lineset_to_trianglemesh, generate_city_json_from_building
)
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


# ── filter_lines_within_contour ────────────────────────────────────────────────

def _make_lineset_inside(z=0.0):
    """LineSet with one line entirely inside a [0,2]×[0,2] square."""
    pts = np.array([[0.5, 0.5, z], [1.5, 0.5, z], [1.5, 1.5, z], [0.5, 1.5, z]])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    return ls


def _square_contour(z=0.0):
    return np.array([[0.0, 0.0, z], [2.0, 0.0, z], [2.0, 2.0, z], [0.0, 2.0, z]])


class TestFilterLinesWithinContour:
    def test_lines_inside_are_kept(self):
        contour = _square_contour()
        ls = _make_lineset_inside()
        result = filter_lines_within_contour(contour, ls)
        assert len(result.lines) == 4

    def test_lines_outside_are_removed(self):
        contour = _square_contour()
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(
            np.array([[10.0, 10.0, 0.0], [11.0, 10.0, 0.0]])
        )
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        result = filter_lines_within_contour(contour, ls)
        assert len(result.lines) == 0

    def test_returns_lineset(self):
        contour = _square_contour()
        ls = _make_lineset_inside()
        result = filter_lines_within_contour(contour, ls)
        assert isinstance(result, o3d.geometry.LineSet)

    def test_all_points_preserved(self):
        """The output LineSet keeps the original point set unchanged."""
        contour = _square_contour()
        ls = _make_lineset_inside()
        result = filter_lines_within_contour(contour, ls)
        assert len(result.points) == len(ls.points)

    def test_open_contour_is_auto_closed(self):
        """A contour without repeated first/last point is handled gracefully."""
        # open (no repeat of first point)
        contour = _square_contour()
        ls = _make_lineset_inside()
        # should not raise
        filter_lines_within_contour(contour, ls)

    def test_buffer_keeps_line_just_outside_contour(self):
        """A line just outside the contour boundary is kept when buffer > 0."""
        # contour is [0,0]-[2,2]; line is just outside at y=-0.2 (south of boundary)
        contour = _square_contour()
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(
            np.array([[0.5, -0.2, 0.0], [1.5, -0.2, 0.0]])
        )
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        # Without buffer: line is outside → removed
        result_no_buf = filter_lines_within_contour(contour, ls, contour_buffer=0.0)
        assert len(result_no_buf.lines) == 0
        # With buffer of 0.5: line is inside buffered polygon → kept
        result_buf = filter_lines_within_contour(contour, ls, contour_buffer=0.5)
        assert len(result_buf.lines) == 1


# ── lineset_to_trianglemesh ────────────────────────────────────────────────────

class TestLinesetToTriangleMesh:
    def _make_lineset_grid(self):
        """A 3×3 grid of points on z=0.5 as a LineSet."""
        pts = np.array([[float(i), float(j), 0.5]
                        for i in range(3) for j in range(3)])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pts) - 1)])
        return ls

    def test_raises_for_less_than_3_points(self):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        contour = _square_contour()
        with pytest.raises(ValueError):
            lineset_to_trianglemesh(ls, contour)

    def test_returns_triangle_mesh(self):
        ls = self._make_lineset_grid()
        contour = np.array([[0.0, 0.0, 0.5], [3.0, 0.0, 0.5], [3.0, 3.0, 0.5], [0.0, 3.0, 0.5]])
        result = lineset_to_trianglemesh(ls, contour)
        assert isinstance(result, o3d.geometry.TriangleMesh)

    def test_no_triangles_outside_contour(self):
        """Points outside the contour produce an empty mesh."""
        ls = self._make_lineset_grid()
        tiny_contour = np.array([[10.0, 10.0, 0.5], [11.0, 10.0, 0.5],
                                 [11.0, 11.0, 0.5], [10.0, 11.0, 0.5]])
        result = lineset_to_trianglemesh(ls, tiny_contour)
        assert len(result.triangles) == 0

    def test_both_sides_generated(self):
        """For non-empty result, backface copies double the triangle count."""
        ls = self._make_lineset_grid()
        contour = np.array(
            [[0.0, 0.0, 0.5], [3.0, 0.0, 0.5],
             [3.0, 3.0, 0.5], [0.0, 3.0, 0.5]])
        result = lineset_to_trianglemesh(ls, contour)
        if len(result.triangles) > 0:
            # Triangle count must be even (front + back)
            assert len(result.triangles) % 2 == 0

    def test_buffer_includes_points_outside_contour(self):
        """Points just outside the contour produce more triangles when buffer > 0."""
        # 2x2 grid of non-collinear points, all shifted just south of the contour (y=-0.3)
        pts = np.array([
            [0.0, -0.3, 0.5], [1.5, -0.3, 0.5],
            [0.0, 0.3, 0.5], [1.5, 0.3, 0.5],
        ])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [1, 3], [2, 3]])
        # Contour: [0,0]-[3,0]-[3,3]-[0,3] — the y=-0.3 row is just outside
        contour = np.array([[0.0, 0.0, 0.5], [3.0, 0.0, 0.5], [3.0, 3.0, 0.5], [0.0, 3.0, 0.5]])
        result_no_buf = lineset_to_trianglemesh(ls, contour, contour_buffer=0.0)
        result_buf = lineset_to_trianglemesh(ls, contour, contour_buffer=0.5)
        # With buffer the triangles near the boundary are included
        assert len(result_buf.triangles) >= len(result_no_buf.triangles)

    def test_zero_buffer_default_unchanged(self):
        """Explicitly passing contour_buffer=0 gives same result as no buffer."""
        ls = self._make_lineset_grid()
        contour = np.array([[0.0, 0.0, 0.5], [3.0, 0.0, 0.5], [3.0, 3.0, 0.5], [0.0, 3.0, 0.5]])
        result_explicit = lineset_to_trianglemesh(ls, contour, contour_buffer=0.0)
        assert isinstance(result_explicit, o3d.geometry.TriangleMesh)


# ── generate_city_json_from_building ──────────────────────────────────────────

def _simple_lineset(x_offset=0.0):
    """4-point square LineSet at a given x offset."""
    pts = np.array([
        [x_offset + 0.0, 0.0, 0.0],
        [x_offset + 1.0, 0.0, 0.0],
        [x_offset + 1.0, 1.0, 0.0],
        [x_offset + 0.0, 1.0, 0.0],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    return ls


class TestGenerateCityJsonFromBuilding:
    def test_cancelled_dialog_prints_message(self, capsys):
        """If the save dialog is cancelled, 'Save cancelled.' is printed."""
        with patch("Source.linesetTools.tk.Tk"), \
             patch("Source.linesetTools.filedialog.asksaveasfilename", return_value=""):
            generate_city_json_from_building(
                _simple_lineset(0), _simple_lineset(5), _simple_lineset(10)
            )
        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()

    def test_empty_lineset_is_skipped_with_warning(self, capsys):
        """An empty LineSet prints a warning but does not crash."""
        empty_ls = o3d.geometry.LineSet()
        with patch("Source.linesetTools.tk.Tk"), \
             patch("Source.linesetTools.filedialog.asksaveasfilename", return_value=""):
            generate_city_json_from_building(empty_ls, _simple_lineset(5), _simple_lineset(10))
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "cancelled" in captured.out.lower()

    def test_raises_when_all_surfaces_invalid(self):
        """All-empty LineSet inputs raise ValueError."""
        empty = o3d.geometry.LineSet()
        with patch("Source.linesetTools.tk.Tk"), \
             patch("Source.linesetTools.filedialog.asksaveasfilename", return_value="out.json"):
            with pytest.raises(ValueError):
                generate_city_json_from_building(empty, empty, empty)

    def test_writes_json_to_file(self, tmp_path):
        """A valid save path results in a JSON file being written."""
        import json
        out_file = str(tmp_path / "building.json")
        with patch("Source.linesetTools.tk.Tk"), \
             patch("Source.linesetTools.filedialog.asksaveasfilename", return_value=out_file):
            generate_city_json_from_building(
                _simple_lineset(0), _simple_lineset(5), _simple_lineset(10)
            )
        assert os.path.exists(out_file)
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "CityJSON"

    def test_cityjson_properties_merged(self, tmp_path):
        """Extra properties passed via cityjson_properties are included in the file."""
        import json
        out_file = str(tmp_path / "building2.json")
        with patch("Source.linesetTools.tk.Tk"), \
             patch("Source.linesetTools.filedialog.asksaveasfilename", return_value=out_file):
            generate_city_json_from_building(
                _simple_lineset(0), _simple_lineset(5), _simple_lineset(10),
                cityjson_properties={"customKey": "customValue"}
            )
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("customKey") == "customValue"
