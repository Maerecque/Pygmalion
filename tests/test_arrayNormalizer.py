"""Tests for Source/arrayNormalizer.py"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.arrayNormalizer import normalize_array


class TestNormalizeArray:
    # ── Pure logic ────────────────────────────────────────────────────────────

    def test_min_max_normalization_basic(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_array(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)

    def test_min_max_normalization_negative_values(self):
        arr = np.array([-2.0, 0.0, 2.0])
        result = normalize_array(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_colour_mode_divides_by_65535(self):
        arr = np.array([0, 65535, 32767], dtype=float)
        result = normalize_array(arr, is_colour=True)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(32767 / 65535)

    def test_wrong_type_returns_none(self):
        result = normalize_array([1, 2, 3])
        assert result is None

    def test_wrong_type_dict_returns_none(self):
        result = normalize_array({"a": 1})
        assert result is None

    def test_2d_array(self):
        arr = np.array([[0.0, 1.0], [2.0, 4.0]])
        result = normalize_array(arr)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_output_type_is_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = normalize_array(arr)
        assert isinstance(result, np.ndarray)

    def test_single_element_colour(self):
        arr = np.array([65535.0])
        result = normalize_array(arr, is_colour=True)
        assert result[0] == pytest.approx(1.0)

    # ── Mock tests ───────────────────────────────────────────────────────────

    def test_print_called_on_type_error(self, capsys):
        """The function prints a message when a TypeError is raised."""
        normalize_array("not an array")
        captured = capsys.readouterr()
        assert "not the correct type" in captured.out

    def test_numpy_min_called_once(self):
        """np.min is called twice during min-max normalisation (appears twice in formula)."""
        arr = np.array([1.0, 2.0, 3.0])
        with patch("numpy.min", wraps=np.min) as mock_min:
            normalize_array(arr)
            assert mock_min.call_count == 2

    def test_numpy_max_called_once(self):
        """np.max is called once during min-max normalisation."""
        arr = np.array([1.0, 2.0, 3.0])
        with patch("numpy.max", wraps=np.max) as mock_max:
            normalize_array(arr)
            mock_max.assert_called_once()

    def test_colour_mode_skips_min_max(self):
        """In colour mode np.min / np.max are NOT called."""
        arr = np.array([0.0, 32767.0, 65535.0])
        with patch("numpy.min") as mock_min, patch("numpy.max") as mock_max:
            normalize_array(arr, is_colour=True)
            mock_min.assert_not_called()
            mock_max.assert_not_called()
