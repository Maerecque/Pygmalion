"""Tests for Source/fileHandler.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Source.fileHandler import (
    FileFormatError,
    NoFileGivenError,
    NoSaveLocationGivenError,
    readout_LAS_file,
    get_file_path,
    get_save_file_path,
)


# ── Custom exceptions ──────────────────────────────────────────────────────────

class TestCustomExceptions:
    def test_file_format_error_is_exception(self):
        assert issubclass(FileFormatError, Exception)

    def test_no_file_given_error_is_exception(self):
        assert issubclass(NoFileGivenError, Exception)

    def test_no_save_location_given_error_is_exception(self):
        assert issubclass(NoSaveLocationGivenError, Exception)

    def test_file_format_error_can_be_raised(self):
        with pytest.raises(FileFormatError):
            raise FileFormatError("bad format")

    def test_no_file_given_error_can_be_raised(self):
        with pytest.raises(NoFileGivenError):
            raise NoFileGivenError


# ── readout_LAS_file ───────────────────────────────────────────────────────────

class TestReadoutLASFile:
    def test_none_filename_returns_none(self):
        """Passing None triggers NoFileGivenError which is caught – returns None."""
        result = readout_LAS_file(None, prnt_bool=False)
        assert result is None

    def test_empty_string_filename_returns_none(self):
        """Empty string is falsy – same code path as None."""
        result = readout_LAS_file("", prnt_bool=False)
        assert result is None

    def test_nonexistent_file_calls_exit(self):
        """A path that laspy cannot find triggers exit()."""
        with patch("Source.fileHandler.laspy.read",
                   side_effect=FileNotFoundError), \
             patch("builtins.exit") as mock_exit:
            readout_LAS_file("does_not_exist.las", prnt_bool=False)
            mock_exit.assert_called_once()

    def test_laspy_exception_calls_exit(self):
        """A laspy.errors.LaspyException triggers exit()."""
        import laspy
        with patch("Source.fileHandler.laspy.read",
                   side_effect=laspy.errors.LaspyException("bad")), \
             patch("builtins.exit") as mock_exit:
            readout_LAS_file("bad.las", prnt_bool=False)
            mock_exit.assert_called_once()

    def test_memory_error_calls_exit(self):
        """MemoryError triggers exit()."""
        with patch("Source.fileHandler.laspy.read",
                   side_effect=MemoryError), \
             patch("builtins.exit") as mock_exit:
            readout_LAS_file("huge.las", prnt_bool=False)
            mock_exit.assert_called_once()

    def test_successful_read_returns_point_cloud(self):
        """A mocked valid LAS read returns an Open3D PointCloud."""
        import open3d as o3d

        mock_las = MagicMock()
        mock_las.points = list(range(100))
        mock_las.header.scales = np.array([0.001, 0.001, 0.001])
        mock_las.header.offsets = np.array([0.0, 0.0, 0.0])
        mock_las.X = np.zeros(100, dtype=np.int32)
        mock_las.Y = np.zeros(100, dtype=np.int32)
        mock_las.Z = np.zeros(100, dtype=np.int32)
        mock_las.red = np.zeros(100, dtype=np.uint16)
        mock_las.green = np.zeros(100, dtype=np.uint16)
        mock_las.blue = np.zeros(100, dtype=np.uint16)
        # Make str(las) look like the expected format
        mock_las.__str__ = lambda self: "<LasData(1.2, point fmt: <PointFormat(3,"

        with patch("Source.fileHandler.laspy.read", return_value=mock_las):
            result = readout_LAS_file("valid.las", prnt_bool=False)
        assert isinstance(result, o3d.geometry.PointCloud)
        assert len(result.points) == 100

    def test_laspy_read_called_with_filename(self):
        """laspy.read is called with the exact filename supplied."""
        import laspy
        with patch("Source.fileHandler.laspy.read",
                   side_effect=FileNotFoundError) as mock_read, \
             patch("builtins.exit"):
            readout_LAS_file("my_scan.las", prnt_bool=False)
            mock_read.assert_called_once_with("my_scan.las")

    def test_normalize_array_called_three_times_for_rgb(self):
        """normalize_array is called once per colour channel (R, G, B)."""
        mock_las = MagicMock()
        mock_las.points = list(range(10))
        mock_las.header.scales = np.array([1.0, 1.0, 1.0])
        mock_las.header.offsets = np.array([0.0, 0.0, 0.0])
        mock_las.X = np.zeros(10, dtype=np.int32)
        mock_las.Y = np.zeros(10, dtype=np.int32)
        mock_las.Z = np.zeros(10, dtype=np.int32)
        mock_las.red = np.zeros(10, dtype=np.uint16)
        mock_las.green = np.zeros(10, dtype=np.uint16)
        mock_las.blue = np.zeros(10, dtype=np.uint16)
        mock_las.__str__ = lambda self: "<LasData(1.2, point fmt: <PointFormat(3,"

        with patch("Source.fileHandler.laspy.read", return_value=mock_las), \
             patch("Source.fileHandler.normalize_array",
                   wraps=__import__("Source.arrayNormalizer",
                                    fromlist=["normalize_array"]).normalize_array
                   ) as mock_norm:
            readout_LAS_file("test.las", prnt_bool=False)
            assert mock_norm.call_count == 3

    def test_print_suppressed_when_prnt_bool_false(self, capsys):
        """No output is printed when prnt_bool=False and no error occurs."""
        mock_las = MagicMock()
        mock_las.points = list(range(10))
        mock_las.header.scales = np.array([1.0, 1.0, 1.0])
        mock_las.header.offsets = np.array([0.0, 0.0, 0.0])
        mock_las.X = np.zeros(10, dtype=np.int32)
        mock_las.Y = np.zeros(10, dtype=np.int32)
        mock_las.Z = np.zeros(10, dtype=np.int32)
        mock_las.red = np.zeros(10, dtype=np.uint16)
        mock_las.green = np.zeros(10, dtype=np.uint16)
        mock_las.blue = np.zeros(10, dtype=np.uint16)
        mock_las.__str__ = lambda self: "<LasData(1.2, point fmt: <PointFormat(3,"

        with patch("Source.fileHandler.laspy.read", return_value=mock_las):
            readout_LAS_file("test.las", prnt_bool=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ── get_file_path / get_save_file_path (dialog mocks) ─────────────────────────

class TestGetFilePath:
    def test_returns_filename_when_dialog_succeeds(self):
        """When askopenfilename returns a path, get_file_path returns that path."""
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.askopenfilename",
                   return_value="/some/path/scan.las"):
            result = get_file_path("LAS files", "*.las", print_output=False)
        assert result == "/some/path/scan.las"

    def test_returns_none_when_dialog_cancelled(self):
        """When askopenfilename returns '' (user cancels), get_file_path returns None."""
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.askopenfilename", return_value=""):
            result = get_file_path("LAS files", "*.las", print_output=False)
        assert result is None

    def test_tk_withdraw_called(self):
        """Tk().withdraw() is called to hide the root window."""
        mock_root = MagicMock()
        with patch("Source.fileHandler.Tk", return_value=mock_root), \
             patch("Source.fileHandler.askopenfilename", return_value=""):
            get_file_path("X", "*.x", print_output=False)
        mock_root.withdraw.assert_called_once()

    def test_print_output_when_file_selected(self, capsys):
        """File path is printed to stdout when print_output=True."""
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.askopenfilename",
                   return_value="/a/b/c.las"):
            get_file_path("LAS files", "*.las", print_output=True)
        captured = capsys.readouterr()
        assert "/a/b/c.las" in captured.out


class TestGetSaveFilePath:
    def test_returns_path_on_success(self):
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.asksaveasfilename",
                   return_value="/out/building.json"):
            result = get_save_file_path("JSON files", "*.json", "building")
        assert result == "/out/building.json"

    def test_returns_none_when_cancelled(self):
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.asksaveasfilename", return_value=""):
            result = get_save_file_path("JSON files", "*.json", "building")
        assert result is None

    def test_asksaveasfilename_called_with_default_name(self):
        with patch("Source.fileHandler.Tk"), \
             patch("Source.fileHandler.asksaveasfilename",
                   return_value="") as mock_save:
            get_save_file_path("JSON files", "*.json", "my_output")
        _, kwargs = mock_save.call_args
        assert kwargs.get("initialfile") == "my_output"
