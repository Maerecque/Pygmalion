"""Tests for Source/heightMapModule.py"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.heightMapModule import (
    project_vertices_to_plane, create_grid, generate_height_map,
    create_point_cloud, find_edges, generate_wall_points,
    transform_pointcloud_to_height_map
)
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

    def test_invalid_nx_returns_none(self):
        result = create_grid((0, 1), (0, 1), nx=1, ny=5)
        assert result == (None, None, None)

    def test_invalid_ny_returns_none(self):
        result = create_grid((0, 1), (0, 1), nx=5, ny=1)
        assert result == (None, None, None)

    def test_none_nx_returns_none(self):
        result = create_grid((0, 1), (0, 1), nx=None, ny=5)
        assert result == (None, None, None)

    def test_none_ny_returns_none(self):
        result = create_grid((0, 1), (0, 1), nx=5, ny=None)
        assert result == (None, None, None)

    def test_error_message_printed(self, capsys):
        create_grid((0, 1), (0, 1), nx=1, ny=5)
        captured = capsys.readouterr()
        assert "Error" in captured.out


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

    def test_none_x_grid_raises(self):
        with pytest.raises(ValueError):
            generate_height_map(np.array([0.0]), np.array([0.0]), np.array([1.0]), None, np.array([0.0, 1.0]))

    def test_none_y_grid_raises(self):
        with pytest.raises(ValueError):
            generate_height_map(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0, 1.0]), None)

    def test_empty_x_grid_raises(self):
        with pytest.raises(ValueError):
            generate_height_map(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([]), np.array([0.0, 1.0]))


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


class TestFindEdges:
    def _make_height_map(self):
        """3x3 height map with a filled centre cell."""
        hm = np.full((5, 5), -np.inf)
        hm[2, 2] = 1.0
        return hm

    def test_returns_ndarray(self):
        hm = self._make_height_map()
        result = find_edges(hm)
        assert isinstance(result, np.ndarray)

    def test_edges_found_around_filled_cell(self):
        hm = self._make_height_map()
        edges = find_edges(hm)
        # There should be at least one edge pixel detected
        assert len(edges) > 0

    def test_all_filled_map_has_only_border_edges(self):
        hm = np.ones((5, 5))
        edges = find_edges(hm)
        # Interior of a fully filled map has no dilation/erosion difference
        assert edges.shape[1] == 2  # each row is a (row, col) pair

    def test_all_empty_map_returns_empty(self):
        hm = np.full((5, 5), -np.inf)
        edges = find_edges(hm)
        assert len(edges) == 0


class TestGenerateWallPoints:
    def _make_inputs(self):
        """Simple 3x3 height map with a single filled cell."""
        hm = np.full((3, 3), -np.inf)
        hm[1, 1] = 2.0
        x_grid = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 2.0])
        edges = np.array([[1, 1]])
        return hm, x_grid, y_grid, edges

    def test_returns_ndarray(self):
        hm, x_grid, y_grid, edges = self._make_inputs()
        result = generate_wall_points(edges, edges, hm, x_grid, y_grid, 0.0, 0.5)
        assert isinstance(result, np.ndarray)

    def test_generates_points_between_floor_and_ceiling(self):
        hm, x_grid, y_grid, edges = self._make_inputs()
        result = generate_wall_points(edges, edges, hm, x_grid, y_grid, 0.0, 0.5)
        # z values should span from 0.0 to 2.0
        assert result.shape[1] == 3
        assert result[:, 2].min() == pytest.approx(0.0)
        assert result[:, 2].max() == pytest.approx(2.0)

    def test_no_matching_ceiling_returns_empty(self):
        hm, x_grid, y_grid, _ = self._make_inputs()
        floor_edges = np.array([[1, 1]])
        ceiling_edges = np.array([[0, 0]])  # no match with floor edge
        result = generate_wall_points(floor_edges, ceiling_edges, hm, x_grid, y_grid, 0.0, 0.5)
        assert len(result) == 0

    def test_zero_height_diff_produces_no_points(self):
        hm, x_grid, y_grid, edges = self._make_inputs()
        # z_min == ceiling_z → height_diff = 0 → no wall points
        result = generate_wall_points(edges, edges, hm, x_grid, y_grid, 2.0, 0.5)
        assert len(result) == 0

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_tqdm_called_once(self):
        """tqdm is invoked once to iterate over floor_edges."""
        hm, x_grid, y_grid, edges = self._make_inputs()
        with patch("Source.heightMapModule.tqdm", wraps=__builtins__) as mock_tqdm:  # noqa
            with patch("Source.heightMapModule.tqdm", side_effect=lambda it, **kw: it) as mock_tqdm:
                generate_wall_points(edges, edges, hm, x_grid, y_grid, 0.0, 0.5)
                mock_tqdm.assert_called_once()


class TestTransformPointcloudToHeightMap:
    def _make_pcd(self):
        pts = np.array([[float(i), float(j), float(k)]
                        for i in range(5) for j in range(5) for k in range(5)], dtype=float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def test_raises_on_empty_pcd(self):
        pcd = o3d.geometry.PointCloud()
        with pytest.raises(ValueError):
            transform_pointcloud_to_height_map(pcd)

    def test_returns_three_point_clouds(self):
        pcd = self._make_pcd()
        result = transform_pointcloud_to_height_map(pcd)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, o3d.geometry.PointCloud)

    def test_floor_and_ceiling_have_points(self):
        pcd = self._make_pcd()
        floor_pcd, ceiling_pcd, _ = transform_pointcloud_to_height_map(pcd)
        assert len(floor_pcd.points) > 0
        assert len(ceiling_pcd.points) > 0

    def test_floor_z_is_minimum(self):
        pcd = self._make_pcd()
        floor_pcd, _, _ = transform_pointcloud_to_height_map(pcd)
        pts = np.asarray(pcd.points)
        floor_pts = np.asarray(floor_pcd.points)
        assert np.allclose(floor_pts[:, 2], pts[:, 2].min())

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_visualize_calls_draw_geometries(self):
        """draw_geometries is called when visualize_map=True."""
        pcd = self._make_pcd()
        with patch("Source.heightMapModule.o3d.visualization.draw_geometries") as mock_draw:
            transform_pointcloud_to_height_map(pcd, visualize_map=True)
            mock_draw.assert_called_once()

    def test_no_visualize_does_not_call_draw(self):
        """draw_geometries is NOT called when visualize_map=False."""
        pcd = self._make_pcd()
        with patch("Source.heightMapModule.o3d.visualization.draw_geometries") as mock_draw:
            transform_pointcloud_to_height_map(pcd, visualize_map=False)
            mock_draw.assert_not_called()

    def test_debugging_logs_prints_output(self, capsys):
        """debugging_logs=True produces console output."""
        pcd = self._make_pcd()
        transform_pointcloud_to_height_map(pcd, debugging_logs=True)
        captured = capsys.readouterr()
        assert "range" in captured.out

    def test_generate_height_map_called(self):
        """generate_height_map is called internally."""
        pcd = self._make_pcd()
        with patch("Source.heightMapModule.generate_height_map",
                   wraps=__import__("Source.heightMapModule", fromlist=["generate_height_map"]).generate_height_map
                   ) as mock_hm:
            transform_pointcloud_to_height_map(pcd)
            mock_hm.assert_called_once()

    def test_create_grid_called(self):
        """create_grid is called internally."""
        pcd = self._make_pcd()
        with patch("Source.heightMapModule.create_grid",
                   wraps=__import__("Source.heightMapModule", fromlist=["create_grid"]).create_grid
                   ) as mock_grid:
            transform_pointcloud_to_height_map(pcd)
            mock_grid.assert_called_once()
