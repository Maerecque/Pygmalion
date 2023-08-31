import tkinter as tk
from tkinter import filedialog, messagebox
import laspy
import open3d as o3d
import numpy as np
import os
import subprocess
import threading


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Resizer™")
        self.root.geometry("300x300")
        self.root.resizable(False, False)
        self.root.iconbitmap("logo.ico")

        self.selected_file = None
        self.point_count = None
        self.grid_subsampling_size = None

        self.file_label = tk.Label(root, text="Selected file: None")
        self.file_label.pack(pady=10)

        self.point_count_label = tk.Label(root, text="Point cloud size: N/A")
        self.point_count_label.pack(pady=5)

        self.open_button = tk.Button(root, text="Open file", command=self.open_file)
        self.open_button.pack(pady=5)

        self.float_frame = tk.Frame(root)
        self.float_frame.pack(pady=10, padx=10, fill="x")

        self.voxel_label = tk.Label(self.float_frame, text="Voxel size:")
        self.voxel_label.pack(side="left")

        self.float_var = tk.DoubleVar(value=2.0)
        self.float_entry = tk.Entry(self.float_frame, textvariable=self.float_var, width=8)
        self.float_entry.pack(side="left")

        self.decrease_button = tk.Button(self.float_frame, text="-", command=self.decrease_float)
        self.decrease_button.pack(side="right", padx=5)

        self.increase_button = tk.Button(self.float_frame, text="+", command=self.increase_float)
        self.increase_button.pack(side="right")

        self.grid_subsampling_size_label = tk.Label(root, text="Point cloud size after subsampling: N/A")
        self.grid_subsampling_size_label.pack(pady=5)

        self.subsampling_factor_label = tk.Label(root, text="Subsampling factor: N/A")
        self.subsampling_factor_label.pack(pady=5)

        # Create a frame to hold the buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.grid_subsampling_button = tk.Button(button_frame, text="Apply subsampling", command=self.apply_grid_subsampling)
        self.grid_subsampling_button.pack(side="left", padx=10, pady=10)  # Place on the left side

        self.preview_button = tk.Button(button_frame, text="Preview", command=self.preview_resized_point_cloud)
        self.preview_button.pack(side="left", padx=10, pady=10)  # Place on the left side

        self.save_button = tk.Button(button_frame, text="Save", command=self.save_to_las)
        self.save_button.pack(side="left", padx=10, pady=10)  # Place on the left side

        # If presses <Escape>, close the application
        self.root.bind("<Escape>", self.close_application)

    # Method to close the application
    def close_application(self, event):
        self.root.destroy()

    # Method to preview the downsampled point cloud
    def preview_resized_point_cloud(self):
        if self.grid_subsampling_size:
            # Start a new thread to preview the downsampled point cloud
            threading.Thread(target=self.preview_resized_point_cloud_thread).start()
        else:
            messagebox.showwarning("No point cloud", "No point cloud available for preview.")

    # Method to preview the downsampled point cloud
    def preview_resized_point_cloud_thread(self):
        try:
            point_cloud = readout_LAS_file(self.selected_file)

            voxel_size = self.float_var.get()
            downsampled_pcd = grid_subsampling(point_cloud, voxel_size)

            print("Point cloud size after subsampling:", len(downsampled_pcd.points))

            # Save the downsampled point cloud to a temporary PCD file
            temp_pcd_path = "temp_pcd.pcd"
            o3d.io.write_point_cloud(temp_pcd_path, downsampled_pcd)

            # Launch the open3d_visualization.py script as a separate process
            subprocess.Popen(["python", "open3d_visualization.py", temp_pcd_path])

        except Exception as e:
            messagebox.showerror("Error", "An error occurred while previewing the point cloud:\n" + str(e))

    # Method to open a file dialog and select a file
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS and LAZ files", ["*.las", "*.laz"])])
        if file_path:
            self.selected_file = file_path
            self.update_file_label()

            # Create a new thread for reading the LAS/LAZ file
            loading_thread = threading.Thread(target=self.read_las_laz)  # Automatically read LAS/LAZ file and update point count
            loading_thread.start()

    # Method to update the file label
    def update_file_label(self):
        if self.selected_file:
            self.short_file_name = os.path.basename(self.selected_file)
            self.file_label.config(text=f"Selected file: \n{self.short_file_name}")
        else:
            self.file_label.config(text="Selected file: None")

    # Method to update the point count label
    def update_point_count(self):
        if self.point_count is not None:
            self.point_count_label.config(text=f"Point cloud size: {self.point_count}")
        else:
            self.point_count_label.config(text="Point cloud size: N/A")

    # Method to increase the float value
    def increase_float(self):
        self.float_var.set(round(self.float_var.get() + 0.05, 2))

    # Method to decrease the float value
    def decrease_float(self):
        self.float_var.set(round(max(0.05, self.float_var.get() - 0.05), 2))

    # Method to read the LAS/LAZ file and update the point count label
    def read_las_laz(self):
        if self.selected_file:
            try:
                self.point_count = "Loading..."
                self.update_point_count()
                point_cloud = readout_LAS_file(self.selected_file)
                self.point_count = len(point_cloud.points)
                self.update_point_count()
            except Exception as e:
                print("An error occurred while reading the LAS/LAZ file:")
                print(type(e).__name__, ":", e)
        else:

            print("No file selected. Please select a file first.")

    # Method to apply grid subsampling to the point cloud in a separate thread
    def apply_grid_subsampling(self):
        if self.selected_file:
            # Start a new thread to apply grid subsampling
            threading.Thread(target=self.apply_grid_subsampling_thread).start()
        else:
            messagebox.showwarning("Warning", "No file selected or no point cloud available for subsampling.")
            print("No file selected. Please select a file first.")

    # Actual method to apply grid subsampling to the point cloud
    def apply_grid_subsampling_thread(self):
        try:
            point_cloud = readout_LAS_file(self.selected_file)
            original_point_count = len(point_cloud.points)
            self.point_count = original_point_count
            self.update_point_count()

            voxel_size = self.float_var.get()
            downsampled_pcd = grid_subsampling(point_cloud, voxel_size)
            self.grid_subsampling_size = len(downsampled_pcd.points)

            # Update the GUI with the new values
            self.root.after(0, self.update_grid_subsampling_size)
            subsampling_factor = original_point_count / self.grid_subsampling_size
            self.root.after(0, self.update_subsampling_factor, subsampling_factor)

        except Exception as e:
            print("An error occurred:")
            print(type(e).__name__, ":", e)

    # Method to update the subsampling factor label
    def update_subsampling_factor(self, factor):
        self.subsampling_factor_label.config(text=f"Subsampling factor: {factor:.2f}")

    # Method to update the grid subsampling size label
    def update_grid_subsampling_size(self):
        if self.grid_subsampling_size is not None:
            self.grid_subsampling_size_label.config(text=f"Point cloud size after subsampling: {self.grid_subsampling_size}")
        else:
            self.grid_subsampling_size_label.config(text="Point cloud size after subsampling: N/A")

    # Method to save the downsampled point cloud to a LAS file in a separate thread
    def save_to_las(self):
        if self.selected_file and self.grid_subsampling_size:
            # Start a new thread to save the point cloud to a LAS file
            threading.Thread(target=self.save_to_las_thread).start()
        else:
            messagebox.showwarning("Warning", "No file selected or no point cloud available for saving.")
            print("No file selected. Please select a file first.")

    # Actual method to save the downsampled point cloud to a LAS file
    def save_to_las_thread(self):
        try:
            point_cloud = readout_LAS_file(self.selected_file)

            voxel_size = self.float_var.get()
            downsampled_pcd = grid_subsampling(point_cloud, voxel_size)

            output_path = filedialog.asksaveasfilename(defaultextension=".las", filetypes=[("LAS files", "*.las")])
            if output_path:
                custom_header = laspy.LasHeader(version="1.2", point_format=3)
                las_header_point_format = custom_header.point_format
                las_header_file_version = custom_header.version

                # Create a new LAS file using laspy
                outfile = laspy.create(point_format=las_header_point_format, file_version=las_header_file_version)

                # Convert Open3D point cloud data to laspy format
                x, y, z = np.array(downsampled_pcd.points).T
                outfile.x = x
                outfile.y = y
                outfile.z = z
                red, green, blue = np.array(downsampled_pcd.colors).T
                outfile.red = (red * 65535).astype(np.uint16)
                outfile.green = (green * 65535).astype(np.uint16)
                outfile.blue = (blue * 65535).astype(np.uint16)

                # Close the LAS file
                outfile.write(output_path)

                # Update the GUI after the file is saved
                self.root.after(0, self.show_save_success, output_path)
        except Exception as e:
            self.root.after(0, self.show_save_error, e)

    def show_save_success(self, output_path):
        messagebox.showinfo("Success", f"Saved downsampled point cloud to:\n{output_path}")
        print(f"Saved downsampled point cloud to: {output_path}")

    def show_save_error(self, e):
        messagebox.showerror("Error", f"An error occurred while saving the LAS file:\n{type(e).__name__}: {e}")
        print("An error occurred while saving the LAS file:")
        print(type(e).__name__, ":", e)


def normalize_array(array):
    """Function to normalize a NumPy ndarray.

    Args:
        array (*): A NumPy ndarray to normalize.

    Returns:
        Array: Normalized array.
    """
    return array.astype(np.float32) / 65535


class noFileGivenError(Exception):
    pass


class FileFormatError(Exception):
    pass


def readout_LAS_file(filename: str) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Raises:
        FileNotFoundError: If a given path does not exist.

        noFileGivenError: If no file is selected.

        FileFormatError: If the format of the given file is not supported.

        laspy.errors.LaspyException: If Laspy runs into an error.


    Returns:
        o3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        if not filename:
            raise noFileGivenError

        las = laspy.read(filename)

        # check if LAS file is in the correct format
        if "<LasData(1.2, point fmt: <PointFormat(3," not in str(las):
            if len(las.points) < 10000000:
                raise FileFormatError

        geom = o3d.geometry.PointCloud()

        # Create an Open3d model that contains the points from the LAS/LAZ file.
        point_data = np.stack([
            las.X,
            las.Y,
            las.Z
        ], axis=0).transpose((1, 0))

        # With the line below the visualization will look "odd", but is needed for the export to PLY and turn back to the LAS format.
        geom.points = o3d.utility.Vector3dVector((point_data * las.header.scales) + las.header.offsets)
        # geom.points = o3d.utility.Vector3dVector(point_data)

        # Assign the colours of the points to the Open3d model.
        # Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly.
        colour_data = np.stack([
            normalize_array(las.red),
            normalize_array(las.green),
            normalize_array(las.blue)
        ], axis=0).transpose((1, 0))
        geom.colors = o3d.utility.Vector3dVector(colour_data)

        # # It seems like Open3d does not accept the use gps as a variable. Find a way to give this to the ply file.
        # gpsData = np.stack([las.gps_time], axis=0).transpose((1, 0))
        # geom.gps = o3d.utility.Vector2dVector(gpsData)

        return geom
    except FileNotFoundError:
        print("Could not find a file on the given PATH,  please check if the PATH exists.")
        exit()
    except noFileGivenError:
        print("No file was selected, script will not be stopped.")
        return
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and/or if it is a LAS/LAZ file.")
        exit()
    except FileFormatError:
        print("The chosen LAS/LAZ file is not in the correct format or correct version. This file will not be used.")
        exit()
    except Exception as e:
        print("An unforeseen error occurred. See below for details.")
        print(type(e))
        print(e)
        exit()


def grid_subsampling(pcd: o3d.cpu.pybind.geometry.PointCloud, voxel_size: float) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to normalize the points in a point cloud over a grid.
    So I found out, maybe the hard way. This method will always normalize the size of the point cloud based on how many points are in the cloud.
    A large point cloud will not be decimated by the same size as a smaller point cloud. It is all dependent on the original density of the cloud.
    NOTE:   Should research a more reliable way to normalize the point cloud, find a way to normalize based on original point distance.
            `voxel_down_sample_and_trace` exists.
            No you this doesn't work, go sit in the corner and rethink your life choices. - Marc, 30/08/2023

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud to be normalized.

        voxel_size (float, optional): Distance between points that is allowed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Down sampled point cloud with normalized point positions.
    """
    # Downsample the point cloud to a regular grid using voxel_down_sample
    downsampled_pcd = pcd.voxel_down_sample(voxel_size / 100)

    return downsampled_pcd


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
