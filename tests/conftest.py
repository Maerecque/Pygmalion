"""
conftest.py — session-wide patches that prevent any Open3D visualization
window from opening during the test run.
"""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True, scope="session")
def block_o3d_visualization():
    """Patch every Open3D visualization entry-point for the whole test session."""
    with patch("open3d.visualization.draw_geometries"), \
         patch("open3d.visualization.draw_geometries_with_editing"), \
         patch("open3d.visualization.draw"):
        yield
