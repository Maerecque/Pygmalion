"""Tests for Source/heightMapModule.py"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.heightMapModule import project_vertices_to_plane, create_grid, generate_height_map, create_point_cloud
import open3d as o3d


class TestProjectVerticesToPlane:
    def test_splits_into_xyz(self):
        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x, y, z = project_vertices_to_plane(verts)
        np.testing.assert_array_equal(x, [1.0, 4.0])
        np.testing.assert_array_equal(y, [2.0, 5.0])
        np.testing.assert_array_equal(z, [3.0, 6.0])

    def test_single_point(self):
        verts = np.array([[7.0, 8.0, 9.0]])
        x, y, z = project_vertices_to_plane(verts)
        assert x[0] == 7.0
        assert y[0] == 8.0
        assert z[0] == 9.0

    def test_output_shapes_match(self):
        verts = np.random.rand(50, 3)
        x, y, z = project_vertices_to_plane(verts)
        assert x.shape == (50,)
        assert y.shape == (50,)
        assert z.shape == (50,)


class TestCreateGrid:
    def test_shape_matches_nx_ny(self):
        (xx, yy), x_grid, y_grid = create_grid((0, 1), (0, 1), nx=5, ny=7)
        assert xx.shape == (7, 5)
        assert yy.shape == (7, 5)
        assert len(x_grid) == 5
        assert len(y_grid) == 7

    def test_range_boundaries(self):
        (xx, yy), x_grid, y_grid = create_grid((2, 8), (10, 20), nx=4, ny=4)
        assert x_grid[0] == pytest.approx(2.0)
        assert x_grid[-1] == pytest.approx(8.0)
        assert y_grid[0] == pytest.approx(10.0)
        assert y_grid[-1] == pytest.approx(20.0)

    def test_single_step(self):
        (xx, yy), x_grid, y_grid = create_grid((0, 10), (0, 10), nx=2, ny=2)
        assert len(x_grid) == 2
        assert len(y_grid) == 2


class TestGenerateHeightMap:
    def test_keeps_max_z_per_cell(self):
        x_grid = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        # Two points fall into cell (0,0) with z=1.0 and z=5.0 – max should win
        x = np.array([0.0, 0.0])
        y = np.array([0.0, 0.0])
        z = np.array([1.0, 5.0])
        hm = generate_height_map(x, y, z, x_grid, y_grid)
        assert hm[0, 0] == pytest.approx(5.0)

    def test_output_shape(self):
        x_grid = np.linspace(0, 1, 5)
        y_grid = np.linspace(0, 1, 6)
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([3.0])
        hm = generate_height_map(x, y, z, x_grid, y_grid)
        assert hm.shape == (5, 6)

    def test_cell_gets_correct_value(self):
        x_grid = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 2.0])
        # Point exactly at cell (2, 2)
        x = np.array([2.0])
        y = np.array([2.0])
        z = np.array([99.0])
        hm = generate_height_map(x, y, z, x_grid, y_grid)
        assert hm[2, 2] == pytest.approx(99.0)


class TestCreatePointCloud:
    # ── Pure logic ────────────────────────────────────────────────────────────

    def test_returns_point_cloud(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pcd = create_point_cloud(coords)
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_point_count_matches(self):
        coords = np.random.rand(10, 3)
        pcd = create_point_cloud(coords)
        assert len(pcd.points) == 10

    def test_color_applied(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        pcd = create_point_cloud(coords, color=(1.0, 0.0, 0.0))
        assert pcd.has_colors()
        colors = np.asarray(pcd.colors)
        assert np.allclose(colors[0], [1.0, 0.0, 0.0])

    def test_invalid_color_raises(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            create_point_cloud(coords, color=(2.0, 0.0, 0.0))

    def test_list_input_converts(self):
        coords = [np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 1.0, 1.0]])]
        pcd = create_point_cloud(coords)
        assert pcd is not None

    def test_invalid_input_returns_none(self):
        pcd = create_point_cloud("not an array")
        assert pcd is None

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_o3d_point_cloud_constructor_called(self):
        """open3d.geometry.PointCloud is instantiated exactly once."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with patch("Source.heightMapModule.o3d.geometry.PointCloud",
                   wraps=o3d.geometry.PointCloud) as mock_pcd:
            create_point_cloud(coords)
            mock_pcd.assert_called_once()

    def test_error_message_printed_on_bad_input(self, capsys):
        """An error message is printed when input is neither array nor list."""
        create_point_cloud(42)
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_tqdm_not_called_in_create_point_cloud(self):
        """create_point_cloud does not invoke tqdm (no progress bar)."""
        coords = np.random.rand(5, 3)
        with patch("Source.heightMapModule.tqdm") as mock_tqdm:
            create_point_cloud(coords)
            mock_tqdm.assert_not_called()
